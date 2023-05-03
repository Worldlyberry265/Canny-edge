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
__global__ void non_maximum_suppression_kernel(float *magnitude, float *orientation, unsigned char *edge_map, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        float mag = magnitude[y * width + x];
        float angle = orientation[y * width + x];
        float q = 255;
        float r = 255;

        // Check direction of edge
        if ((angle < -M_PI / 8) || (angle >= M_PI / 8 && angle <= 3 * M_PI / 8)) {
            q = magnitude[y * width + x + 1];
            r = magnitude[y * width + x - 1];
        } else if ((angle >= -3 * M_PI / 8 && angle < -M_PI / 8) || (angle >= 3 * M_PI / 8 && angle < M_PI / 8)) {
            q = magnitude[(y - 1) * width + x];
            r = magnitude[(y + 1) * width + x];
        } else if (angle >= M_PI / 8 && angle < 3 * M_PI / 8) {
            q = magnitude[y * width + x + 1];
            r = magnitude[y * width + x - 1];
        } else if (angle >= 3 * M_PI / 8 && angle < 5 * M_PI / 8) {
            q = magnitude[(y + 1) * width + x - 1];
            r = magnitude[(y - 1) * width + x + 1];
        }

        // Check if current pixel is a local maximum
        if (mag >= q && mag >= r) {
            edge_map[y * width + x] = (unsigned char) mag;
        } else {
            edge_map[y * width + x] = 0;
        }
    }
}
// Step 3: Non-maximum suppression PART 2
void non_maximum_suppression(float *magnitude, float *orientation, unsigned char *edge_map, int width, int height)
{
    // Allocate memory on device
    float *d_magnitude, *d_orientation;
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

    // Call kernel function
    non_maximum_suppression_kernel<<<gridDim, blockDim>>>(d_magnitude, d_orientation, d_edge_map, width, height);

    // Copy data from device to host
    cudaMemcpy(edge_map, d_edge_map, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(d_magnitude);
    cudaFree(d_orientation);
    cudaFree(d_edge_map);
}

// Step 4: double thresholding
__global__ void double_thresholding_kernel(unsigned char *edge_map, int width, int height, float low_thresh, float high_thresh, int *queue_elements, int *queue_size, int *queue_front, int *queue_rear) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= width * height) {
        return;
    }

    // First pass: thresholding
    if (edge_map[i] >= high_thresh) {
        edge_map[i] = 255;
    } else if (edge_map[i] < low_thresh) {
        edge_map[i] = 0;
    }

    // Second pass: hysteresis thresholding
    if (edge_map[i] == 255) {
        int rear = atomicAdd(queue_rear, 1);
        queue_elements[rear] = i;
        atomicAdd(queue_size, 1);
    }

    __syncthreads();

    while (*queue_size > 0) {
        int idx = atomicAdd(queue_front, 1);
        if (idx >= *queue_size) {
            break;
        }
        int j = queue_elements[idx];
        atomicSub(queue_size, 1);

        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                if (y == 0 && x == 0) {
                    continue;
                }
                int ix = j % width + x;
                int iy = j / width + y;
                if (ix < 0 || ix >= width || iy < 0 || iy >= height) {
                    continue;
                }
                int idx2 = iy * width + ix;
                if (edge_map[idx2] >= low_thresh && edge_map[idx2] < 255) {
                    edge_map[idx2] = 255;
                    int rear2 = atomicAdd(queue_rear, 1);
                    queue_elements[rear2] = idx2;
                    atomicAdd(queue_size, 1);
                }
            }
        }

        __syncthreads();
    }
}

// Step 4: Non-maximum suppression PART 2
void double_thresholding(unsigned char *edge_map, int width, int height, float low_thresh, float high_thresh) {
    int *queue_elements, *queue_size, *queue_front, *queue_rear;
    cudaMallocManaged(&queue_elements, width * height * sizeof(int));
    cudaMallocManaged(&queue_size, sizeof(int));
    cudaMallocManaged(&queue_front, sizeof(int));
    cudaMallocManaged(&queue_rear, sizeof(int));
    *queue_size = 0;
    *queue_front = 0;
    *queue_rear = -1;

    int threads_per_block = 256;
    int blocks_per_grid = (width * height + threads_per_block - 1) / threads_per_block;
    double_thresholding_kernel<<<blocks_per_grid, threads_per_block>>>(edge_map, width, height, low_thresh, high_thresh, queue_elements, queue_size, queue_front, queue_rear);
    cudaDeviceSynchronize();

    cudaFree(queue_elements);
    cudaFree(queue_size);
    cudaFree(queue_front);
    cudaFree(queue_rear);
}




// Step 5: Edge tracking by hysteresis
void edge_tracking_mpi(unsigned char *edge_map, int width, int height, float low_thresh)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate sub-image dimensions
    int sub_width = width / size;
    int sub_height = height;
    int sub_start = rank * sub_width;
    int sub_end = (rank + 1) * sub_width;
    if (rank == size - 1) {
        sub_end = width;
    }

    // Allocate memory for sub-image and copy data
    unsigned char *sub_edge_map = (unsigned char *) malloc(sub_width * sub_height * sizeof(unsigned char));
    memcpy(sub_edge_map, edge_map + sub_start, sub_width * sub_height * sizeof(unsigned char));

    // Apply edge tracking on sub-image
    for (int y = 0; y < sub_height; y++) {
        for (int x = 0; x < sub_width; x++) {
            if (sub_edge_map[y * sub_width + x] >= low_thresh) {
                // Start new edge
                sub_edge_map[y * sub_width + x] = 255;

                // Perform depth-first search to find connected edge pixels
                int stack_x[100000], stack_y[100000], stack_top = 0;
                stack_x[stack_top] = x;
                stack_y[stack_top] = y;
                while (stack_top >= 0) {
                    int cx = stack_x[stack_top];
                    int cy = stack_y[stack_top];
                    stack_top--;
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (cx + dx < 0 || cx + dx >= sub_width || cy + dy < 0 || cy + dy >= sub_height) {
                                continue;
                            }
                            if (sub_edge_map[(cy + dy) * sub_width + (cx + dx)] >= low_thresh &&
                                sub_edge_map[(cy + dy) * sub_width + (cx + dx)] < 255) {
                                sub_edge_map[(cy + dy) * sub_width + (cx + dx)] = 255;
                                stack_top++;
                                stack_x[stack_top] = cx + dx;
                                stack_y[stack_top] = cy + dy;
                            }
                        }
                    }
                }
            }
        }
    }

    // Gather sub-images to rank 0
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            MPI_Recv(edge_map + i * sub_width, sub_width * sub_height, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(sub_edge_map, sub_width * sub_height, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    // Free memory
    free(sub_edge_map);
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
    
   

