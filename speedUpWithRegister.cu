#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

typedef unsigned int u32;

#define KERNEL_LOOP 1024
#define NUM_ELEMENTS (1<<20)
#define THREADS_PER_BLOCK 256

__constant__ u32 packed_array[KERNEL_LOOP];

__global__ void test_gpu_register(u32 * const data, const u32 num_elements){
    const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < num_elements){
        u32 d_tmp = 0;

        for (int i = 0; i < KERNEL_LOOP; i++){
            d_tmp |= (packed_array[i] << (i & 31));
        }

        data[tid] = d_tmp;
    }
}

__device__ static u32 d_tmp[NUM_ELEMENTS];

__global__ void test_gpu_gmem(u32 * const data, const u32 num_elements){
    const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < num_elements){

        d_tmp[tid] = 0;

        for (int i = 0; i < KERNEL_LOOP; i++){
            d_tmp[tid] |= (packed_array[i] << (i & 31));
        }

        data[tid] = d_tmp[tid];
    }
}

int main(){

    u32 *d_data;
    u32 *h_data;

    size_t size = NUM_ELEMENTS * sizeof(u32);

    h_data = (u32*)malloc(size);
    cudaMalloc(&d_data,size);

    u32 h_packed[KERNEL_LOOP];
    for(int i = 0; i < KERNEL_LOOP; i++){
        h_packed[i] = i;
    }

    cudaMemcpyToSymbol(packed_array, h_packed, sizeof(h_packed));

    int blocks = (NUM_ELEMENTS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEvent_t start, stop;
    float milliseconds;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ================= REGISTER TEST =================
    cudaDeviceSynchronize();
    cudaEventRecord(start);

    test_gpu_register<<<blocks, THREADS_PER_BLOCK>>>(d_data, NUM_ELEMENTS);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Register kernel time: %f ms\n", milliseconds);

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);


    // ================= GLOBAL MEMORY TEST =================
    u32 zero = 0;
    cudaMemcpyToSymbol(d_tmp, &zero, sizeof(u32));

    cudaDeviceSynchronize();
    cudaEventRecord(start);

    test_gpu_gmem<<<blocks, THREADS_PER_BLOCK>>>(d_data, NUM_ELEMENTS);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Global memory kernel time: %f ms\n", milliseconds);

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    free(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}