#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

typedef unsigned int u32;

#define NUM_ELEM 1048576   // 1 Million elements
#define NUM_LISTS 256      // threads per block

// ================= CPU RADIX SORT =================
void cpu_sort(u32 *data, const u32 num_elements)
{
    u32 *tmp0 = (u32*)malloc(NUM_ELEM * sizeof(u32));
    u32 *tmp1 = (u32*)malloc(NUM_ELEM * sizeof(u32));

    for(u32 bit = 0; bit < 32; bit++){
        u32 base0 = 0, base1 = 0;
        u32 mask = (1 << bit);

        for(u32 i = 0; i < num_elements; i++){
            if(data[i] & mask)
                tmp1[base1++] = data[i];
            else
                tmp0[base0++] = data[i];
        }

        for(u32 i = 0; i < base0; i++)
            data[i] = tmp0[i];
        for(u32 i = 0; i < base1; i++)
            data[base0 + i] = tmp1[i];
    }

    free(tmp0);
    free(tmp1);
}

// ================= GPU VERSION 1 =================
__device__ void radix_short_v1(
    u32 *data, u32 num_lists, u32 num_elements,
    u32 tid, u32 *tmp0, u32 *tmp1)
{
    for(u32 bit = 0; bit < 32; bit++){
        u32 base0 = 0, base1 = 0;
        u32 mask = (1 << bit);

        for(u32 i = 0; i < num_elements; i += num_lists){
            u32 val = data[i + tid];

            if(val & mask){
                tmp1[base1 + tid] = val;
                base1 += num_lists;
            } else {
                tmp0[base0 + tid] = val;
                base0 += num_lists;
            }
        }

        for(u32 i = 0; i < base0; i += num_lists)
            data[i + tid] = tmp0[i + tid];

        for(u32 i = 0; i < base1; i += num_lists)
            data[base0 + i + tid] = tmp1[i + tid];
    }
    __syncthreads();
}

__global__ void gpu_v1(u32 *data, u32 num_lists, u32 num_elements)
{
    __shared__ u32 tmp0[NUM_ELEM / NUM_LISTS];
    __shared__ u32 tmp1[NUM_ELEM / NUM_LISTS];

    u32 tid = threadIdx.x;
    radix_short_v1(data, num_lists, num_elements, tid, tmp0, tmp1);
}

// ================= GPU VERSION 2 (Optimized) =================
__device__ void radix_short_v2(
    u32 *data, u32 num_lists, u32 num_elements,
    u32 tid, u32 *tmp1)
{
    for(u32 bit = 0; bit < 32; bit++){
        u32 base0 = 0, base1 = 0;
        u32 mask = (1 << bit);

        for(u32 i = 0; i < num_elements; i += num_lists){
            u32 val = data[i + tid];

            if(val & mask){
                tmp1[base1 + tid] = val;
                base1 += num_lists;
            } else {
                data[base0 + tid] = val;
                base0 += num_lists;
            }
        }

        for(u32 i = 0; i < base1; i += num_lists)
            data[base0 + i + tid] = tmp1[i + tid];
    }
    __syncthreads();
}

__global__ void gpu_v2(u32 *data, u32 num_lists, u32 num_elements)
{
    __shared__ u32 tmp1[NUM_ELEM / NUM_LISTS];
    u32 tid = threadIdx.x;
    radix_short_v2(data, num_lists, num_elements, tid, tmp1);
}

// ================= MAIN =================
int main()
{
    size_t size = NUM_ELEM * sizeof(u32);

    u32 *h_data = (u32*)malloc(size);
    u32 *h_copy = (u32*)malloc(size);
    u32 *d_data;

    // Generate random data
    for(u32 i = 0; i < NUM_ELEM; i++)
        h_data[i] = rand();

    memcpy(h_copy, h_data, size);

    cudaMalloc(&d_data, size);

    // ================= CPU Timing =================
    clock_t start_cpu = clock();
    cpu_sort(h_copy, NUM_ELEM);
    clock_t end_cpu = clock();

    double cpu_time = 1000.0 * (end_cpu - start_cpu) / CLOCKS_PER_SEC;

    // ================= GPU V1 Timing =================
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float gpu_time_v1;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gpu_v1<<<1, NUM_LISTS>>>(d_data, NUM_LISTS, NUM_ELEM);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_v1, start, stop);

    // ================= GPU V2 Timing =================
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    float gpu_time_v2;

    cudaEventRecord(start);
    gpu_v2<<<1, NUM_LISTS>>>(d_data, NUM_LISTS, NUM_ELEM);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_v2, start, stop);

    // ================= RESULTS =================
    printf("\n========== EXECUTION TIME ==========\n");
    printf("CPU Time        : %.3f ms\n", cpu_time);
    printf("GPU V1 Time     : %.3f ms\n", gpu_time_v1);
    printf("GPU V2 Time     : %.3f ms\n", gpu_time_v2);

    printf("\n========== SPEEDUP ==========\n");
    printf("GPU V1 Speedup  : %.2fx\n", cpu_time / gpu_time_v1);
    printf("GPU V2 Speedup  : %.2fx\n", cpu_time / gpu_time_v2);

    cudaFree(d_data);
    free(h_data);
    free(h_copy);

    return 0;
}


// output:
// ========== EXECUTION TIME ==========
// CPU Time        : 373.200 ms
// GPU V1 Time     : 0.000 ms
// GPU V2 Time     : 0.000 ms

// ========== SPEEDUP ==========
// GPU V1 Speedup  : infx
// GPU V2 Speedup  : 1040329763511453537861376251713311026836406272.00x