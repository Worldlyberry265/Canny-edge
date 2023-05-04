#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "stb/stb_image.h"
#include <mpi.h>
#include <omp.h>
#include <cuda.h>


#define M_PI 3.14159265358979323846

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Step 1: Gaussian smoothing
void gaussian_smooth(unsigned char *image, int width, int height, float sigma);
// Step 2: Gradient calculation
void calculate_gradients(unsigned char *image, int width, int height, float *magnitude, float *orientation);
// Step 3: Non-maximum suppression
void non_maximum_suppression(float *magnitude, float *orientation, unsigned char *edge_map, int width, int height);
// Step 4: Double thresholding
void double_thresholding(unsigned char *edge_map, int width, int height, float low_thresh, float high_thresh);
// Step 5: Edge tracking by hysteresis
void edge_tracking(unsigned char *edge_map, int width, int height, float low_thresh);


void gaussian_smooth(unsigned char *img, int width, int height, float sigma)
{
    int size = (int) (sigma * 6) + 1;
    int half_size = size / 2;
    float *kernel = (float *) malloc(size * sizeof(float));
    float sum = 0.0f;

    // Generate Gaussian kernel
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; i++) {
        float x = i - half_size;
        kernel[i] = expf(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }

    // Normalize kernel
    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }

    // Convolve image with Gaussian kernel
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            for (int i = 0; i < size; i++) {
                int ix = x - half_size + i;
                if (ix < 0 || ix >= width) {
                    continue;
                }
                sum += kernel[i] * img[y * width + ix];
            }
            img[y * width + x] = (unsigned char) sum;
        }
    }

    free(kernel);
}

// Step 2: Gradient calculation
void calculate_gradients(unsigned char *img, int width, int height, float *magnitude, float *orientation)
{
    int kernel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int kernel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // Calculate gradient magnitude and orientation
    #pragma omp parallel for 
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float dx = 0.0f, dy = 0.0f;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    dx += kernel_x[i][j] * img[(y + i - 1) * width + (x + j - 1)];
                    dy += kernel_y[i][j] * img[(y + i - 1) * width + (x + j - 1)];
                }
            }
            magnitude[y * width + x] = sqrtf(dx * dx + dy * dy);
            orientation[y * width + x] = atan2f(dy, dx);
        }
    }
}

// Step 3: Non-maximum suppression
__global__ void non_maximum_suppression(float* input, float* output, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height)
    {
        int index = y * width + x;

        // Compute the gradient direction
        float dx = input[index + 1] - input[index - 1];
        float dy = input[index + width] - input[index - width];
        float gradient_direction = atan2(dy, dx);

        // Round the gradient direction to one of four angles
        gradient_direction = (gradient_direction < -M_PI_4) ? (M_PI + gradient_direction) : gradient_direction;
        gradient_direction = (gradient_direction >= M_PI_4) ? (gradient_direction - M_PI) : gradient_direction;
        gradient_direction = (gradient_direction >= -M_PI_4 && gradient_direction < M_PI_4) ? gradient_direction : (gradient_direction + M_PI);

        // Determine the neighboring pixels in the gradient direction
        int x1, y1, x2, y2;
        if (gradient_direction >= -M_PI_4 && gradient_direction < M_PI_4)
        {
            x1 = 1; y1 = -1;
            x2 = 1; y2 = 1;
        }
        else if (gradient_direction >= M_PI_4 && gradient_direction < 3 * M_PI_4)
        {
            x1 = -1; y1 = -1;
            x2 = 1; y2 = 1;
        }
        else if (gradient_direction >= 3 * M_PI_4 || gradient_direction < -3 * M_PI_4)
        {
            x1 = -1; y1 = 0;
            x2 = 1; y2 = 0;
        }
        else
        {
            x1 = -1; y1 = 1;
            x2 = 1; y2 = -1;
        }

        // Check if the current pixel is a local maximum in the gradient direction
        float current_pixel = input[index];
        float neighbor_pixel_1 = input[index + y1 * width + x1];
        float neighbor_pixel_2 = input[index + y2 * width + x2];
        if (current_pixel > neighbor_pixel_1 && current_pixel > neighbor_pixel_2)
        {
            output[index] = current_pixel;
        }
        else
        {
            output[index] = 0.0f;
        }
    }
}

// Step 4: double thresholding
__global__ void double_thresholding(float* input, float* output, float low_threshold, float high_threshold, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = y * width + x;

        // Apply double thresholding
        float pixel_value = input[index];
        if (pixel_value >= high_threshold)
        {
            output[index] = 255.0f;
        }
        else if (pixel_value < low_threshold)
        {
            output[index] = 0.0f;
        }
        else
        {
            // Check if the pixel is connected to a strong edge
            int is_connected_to_strong_edge = 0;
            for (int j = -1; j <= 1; j++)
            {
                for (int i = -1; i <= 1; i++)
                {
                    if (x + i >= 0 && x + i < width && y + j >= 0 && y + j < height)
                    {
                        if (input[(y + j) * width + (x + i)] >= high_threshold)
                        {
                            is_connected_to_strong_edge = 1;
                            break;
                        }
                    }
                }
                if (is_connected_to_strong_edge)
                {
                    break;
                }
            }
            output[index] = (is_connected_to_strong_edge) ? 255.0f : 0.0f;
        }
    }
}

// CUDA Code
void cudaManage(float *magnitude, float *orientation, float low_threshold, float high_threshold, unsigned char *edge_map, int width, int height)
{
    // Allocate memory on device
    float *d_magnitude, *d_orientation;
    float low_threshold = 50.0f;
    float high_threshold = 150.0f;
    unsigned char *d_edge_map;
    cudaMalloc(&d_magnitude, width * height * sizeof(float));
    cudaMalloc(&d_orientation, width * height * sizeof(float));
    cudaMalloc(&d_edge_map, width * height * sizeof(unsigned char));

    // Copy data from host to device
    cudaMemcpy(d_magnitude, magnitude, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_orientation, orientation, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Call kernel functions
    non_maximum_suppression_kernel<<<gridDim, blockDim>>>(d_magnitude, d_orientation, d_edge_map, width, height);
    double_thresholding<<<gridDim, blockDim>>>(edge_map, width, height, low_thresh, high_thresh);

    // Copy data from device to host
    cudaMemcpy(edge_map, d_edge_map, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(d_magnitude);
    cudaFree(d_orientation);
    cudaFree(d_edge_map);
}

// Step 5: Edge tracking by hysteresis
void edge_tracking(unsigned char *edge_map, int width, int height, float low_thresh)
{
    int weak = 25;
    int strong = 255;   //CHANGE BACK TO 75 OR 60 

    // Initialize MPI
    MPI_Init(NULL, NULL);

    // Get the rank of the current process and the total number of processes
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Calculate the number of rows per process
    int rows_per_proc = height / world_size;
    int remainder = height % world_size;

    // Calculate the starting and ending row for the current process
    int start_row = world_rank * rows_per_proc;
    int end_row = (world_rank + 1) * rows_per_proc;
    if (world_rank == world_size - 1) {
        end_row += remainder;
    }

    // Allocate memory for the sub-image
    int sub_height = end_row - start_row;
    unsigned char *sub_edge_map = (unsigned char*) malloc(width * sub_height * sizeof(unsigned char));

    // Copy the sub-image from the main image to the sub-image
    for (int y = start_row; y < end_row; y++) {
        for (int x = 0; x < width; x++) {
            sub_edge_map[(y - start_row) * width + x] = edge_map[y * width + x];
        }
    }

    // Edge tracking on the sub-image
    for (int y = 1; y < sub_height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            if (sub_edge_map[y * width + x] == weak) {
                // Check if any of the 8 neighboring pixels are strong
                if (sub_edge_map[(y - 1) * width + (x - 1)] == strong ||
                    sub_edge_map[(y - 1) * width + x] == strong ||
                    sub_edge_map[(y - 1) * width + (x + 1)] == strong ||
                    sub_edge_map[y * width + (x - 1)] == strong ||
                    sub_edge_map[y * width + (x + 1)] == strong ||
                    sub_edge_map[(y + 1) * width + (x - 1)] == strong ||
                    sub_edge_map[(y + 1) * width + x] == strong ||
                    sub_edge_map[(y + 1) * width + (x + 1)] == strong) {
                    sub_edge_map[y * width + x] = strong;
                } else {
                    sub_edge_map[y * width + x] = 0;
                }
            }
        }
    }

    // Gather the results from all processes to the root process
    if (world_rank == 0) {
        // Allocate memory for the final edge map
        unsigned char *final_edge_map = (unsigned char*) malloc(width * height * sizeof(unsigned char));

        // Copy the sub-image from the current process to the final edge map
        for (int y = start_row; y < end_row; y++) {
            for (int x = 0; x < width; x++) {
                final_edge_map[y * width + x] = sub_edge_map[(y - start_row) * width + x];
            }
        }       
         // Receive the sub-images from the other processes and copy them to the final edge map
    for (int i = 1; i < world_size; i++) {
        // Calculate the starting and ending row for the current sub-image
        int sub_start_row = i * rows_per_proc;
        int sub_end_row = (i + 1) * rows_per_proc;
        if (i == world_size - 1) {
            sub_end_row += remainder;
        }
        int sub_height = sub_end_row - sub_start_row;

        // Receive the sub-image from the current process
        unsigned char *recv_buffer = (unsigned char*) malloc(width * sub_height * sizeof(unsigned char));
        MPI_Recv(recv_buffer, width * sub_height, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Copy the sub-image to the final edge map
        for (int y = sub_start_row; y < sub_end_row; y++) {
            for (int x = 0; x < width; x++) {
                final_edge_map[y * width + x] = recv_buffer[(y - sub_start_row) * width + x];
            }
        }

        // Free the receive buffer
        free(recv_buffer);
    }

    // Copy the final edge map back to the original edge map
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            edge_map[y * width + x] = final_edge_map[y * width + x];
        }
    }

    // Free the memory used by the final edge map
    free(final_edge_map);
} else {
    // Send the sub-image to the root process
    MPI_Send(sub_edge_map, width * sub_height, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
}

// Free the memory used by the sub-image
free(sub_edge_map);

// Finalize MPI
MPI_Finalize();
}


int main()
{
    FILE *fp;
    unsigned char *image_data;
    int width, height;
    float sigma, low_thresh, high_thresh;
    char filename[256];

    // Load input image from file
    printf("Enter input image filename: ");
    scanf("%s", filename);

    fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: could not open %s\n", filename);
        return -1;
    }

    fscanf(fp, "%*s %d %d %*s", &width, &height);

    image_data = (unsigned char *) malloc(width * height);
    fread(image_data, sizeof(unsigned char), width * height, fp);

    fclose(fp);

    // Set algorithm parameters
    printf("Enter Gaussian smoothing parameter sigma: ");
    scanf("%f", &sigma);

    printf("Enter double thresholding low threshold: ");
    scanf("%f", &low_thresh);

    printf("Enter double thresholding high threshold: ");
    scanf("%f", &high_thresh);

    // Apply Canny edge detection algorithm
    float *magnitude = (float *) calloc(width * height, sizeof(float));
    float *orientation = (float *) calloc(width * height, sizeof(float));
    unsigned char *edge_map = (unsigned char *) calloc(width * height, sizeof(unsigned char));

    gaussian_smooth(image_data, width, height, sigma);
    calculate_gradients(image_data, width, height, magnitude, orientation);
    non_maximum_suppression(magnitude, orientation, edge_map, width, height);
    double_thresholding(edge_map, width, height, low_thresh, high_thresh);
    edge_tracking(edge_map, width, height,sigma);
    

    // Save output image to file
    printf("Enter output image filename: ");
    scanf("%s", filename);

    fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error: could not open %s\n", filename);
        return -1;
    }


fprintf(fp, "P5\n%d %d\n255\n", width, height);
    fwrite(edge_map, sizeof(unsigned char), width * height, fp);

    fclose(fp);

    free(image_data);
    free(magnitude);
}
