#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------
// STRATEGY 1: Naive Global Memory (The "Chaos" Strategy)
// ---------------------------------------------------------
__global__ void histogram_naive(const unsigned char *d_data, unsigned int *d_bins, int num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_elements) {
        // Read 1 byte, travel all the way to global memory to add 1
        unsigned char value = d_data[tid];
        atomicAdd(&d_bins[value], 1);
    }
}

// ---------------------------------------------------------
// STRATEGY 2: Shared Memory + 4-Byte Reads (Optimized)
// ---------------------------------------------------------
__global__ void histogram_shared(const unsigned int *d_data, unsigned int *d_bins, int num_elements_u32) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Allocate 256 bins in fast, local shared memory
    __shared__ unsigned int s_bins[256];
    
    // Clear the shared memory first (using the first 256 threads of the block)
    if (threadIdx.x < 256) {
        s_bins[threadIdx.x] = 0;
    }
    __syncthreads(); // Wait for all threads to finish clearing

    // Process data: Read 4 bytes at a time (unsigned int)
    if (tid < num_elements_u32) {
        unsigned int value = d_data[tid];
        
        // Extract each byte and add to the LOCAL shared memory bins
        atomicAdd(&s_bins[value & 0x000000FF], 1);
        atomicAdd(&s_bins[(value & 0x0000FF00) >> 8], 1);
        atomicAdd(&s_bins[(value & 0x00FF0000) >> 16], 1);
        atomicAdd(&s_bins[(value & 0xFF000000) >> 24], 1);
    }
    __syncthreads(); // Wait for all threads in the block to finish counting

    // Finally, write the accumulated local totals to the main global memory
    if (threadIdx.x < 256) {
        atomicAdd(&d_bins[threadIdx.x], s_bins[threadIdx.x]);
    }
}

// ---------------------------------------------------------
// MAIN FUNCTION: Setup, Execution, and Timing
// ---------------------------------------------------------
int main() {
    // 1. Setup: Let's use 100 MB of data for a noticeable test
    int num_elements = 100 * 1024 * 1024; 
    int bin_size = 256 * sizeof(unsigned int);
    
    unsigned char *h_data = (unsigned char *)malloc(num_elements);
    unsigned int *h_bins = (unsigned int *)malloc(bin_size);
    
    // Fill the array with random numbers from 0 to 255
    for (int i = 0; i < num_elements; i++) {
        h_data[i] = rand() % 256;
    }

    // 2. Allocate memory on the GPU
    unsigned char *d_data;
    unsigned int *d_bins;
    cudaMalloc((void **)&d_data, num_elements);
    cudaMalloc((void **)&d_bins, bin_size);
    
    // Copy data to GPU
    cudaMemcpy(d_data, h_data, num_elements, cudaMemcpyHostToDevice);

    // Setup CUDA timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Setup Grid and Block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    printf("Processing %d MB of data...\n", num_elements / (1024 * 1024));
    printf("--------------------------------------------------\n");

    // --- WARM UP ---
    // GPUs need to "wake up". The first kernel call is always artificially slow.
    cudaMemset(d_bins, 0, bin_size);
    histogram_naive<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_bins, num_elements);
    cudaDeviceSynchronize();

    // --- RUN STRATEGY 1: NAIVE ---
    cudaMemset(d_bins, 0, bin_size); // Reset bins to 0
    
    cudaEventRecord(start);
    histogram_naive<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_bins, num_elements);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float time_naive = 0;
    cudaEventElapsedTime(&time_naive, start, stop);
    
    printf("Strategy 1 (Naive Global Memory) took: %.3f ms\n", time_naive);

    // --- RUN STRATEGY 2: SHARED MEMORY ---
    cudaMemset(d_bins, 0, bin_size); // Reset bins to 0
    
    // Because we read 4 bytes at a time, we need 1/4th the threads
    int num_elements_u32 = num_elements / 4; 
    int blocksPerGrid_shared = (num_elements_u32 + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    // Note: We cast d_data to (unsigned int*) so the GPU reads 4 bytes at once
    histogram_shared<<<blocksPerGrid_shared, threadsPerBlock>>>((unsigned int*)d_data, d_bins, num_elements_u32);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float time_shared = 0;
    cudaEventElapsedTime(&time_shared, start, stop);
    
    printf("Strategy 2 (Shared Memory + 4-byte reads) took: %.3f ms\n", time_shared);
    printf("--------------------------------------------------\n");
    printf("Speedup: %.2fx faster!\n", time_naive / time_shared);

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_bins);
    free(h_data);
    free(h_bins);

    return 0;
}