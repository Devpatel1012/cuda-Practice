#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

typedef unsigned int u32;

#define NUM_LISTS 256
#define NUM_ELEM  (1<<20)   // 1,048,576 elements
#define MAX_NUM_LISTS 1024

// ====================== CPU MERGE ======================

u32 find_min(const u32 *src_array,
             u32 *list_indexes,
             u32 num_lists,
             u32 num_elements_per_list)
{
    u32 min_val = 0xFFFFFFFF;
    u32 min_idx = 0;

    for (u32 i = 0; i < num_lists; i++)
    {
        if (list_indexes[i] < num_elements_per_list)
        {
            u32 src_idx = i + list_indexes[i] * num_lists;
            u32 data = src_array[src_idx];

            if (data <= min_val)
            {
                min_val = data;
                min_idx = i;
            }
        }
    }

    list_indexes[min_idx]++;
    return min_val;
}

void merge_array(const u32 *src_array,
                 u32 *dest_array,
                 u32 num_lists,
                 u32 num_elements)
{
    u32 num_elements_per_list = num_elements / num_lists;
    u32 *list_indexes = (u32*)calloc(num_lists, sizeof(u32));

    for (u32 i = 0; i < num_elements; i++)
    {
        dest_array[i] =
            find_min(src_array,
                     list_indexes,
                     num_lists,
                     num_elements_per_list);
    }

    free(list_indexes);
}

// ====================== GPU SECTION ======================

__device__ void merge_array_gpu(const u32 *src_array,
                                u32 *dest_array,
                                u32 num_lists,
                                u32 num_elements)
{
    __shared__ u32 list_indexes[MAX_NUM_LISTS];

    u32 tid = threadIdx.x;

    if (tid < num_lists)
        list_indexes[tid] = 0;

    __syncthreads();

    if (tid == 0)
    {
        u32 num_elements_per_list = num_elements / num_lists;

        for (u32 i = 0; i < num_elements; i++)
        {
            u32 min_val = 0xFFFFFFFF;
            u32 min_idx = 0;

            for (u32 list = 0; list < num_lists; list++)
            {
                if (list_indexes[list] < num_elements_per_list)
                {
                    u32 src_idx =
                        list + list_indexes[list] * num_lists;

                    u32 data = src_array[src_idx];

                    if (data <= min_val)
                    {
                        min_val = data;
                        min_idx = list;
                    }
                }
            }

            list_indexes[min_idx]++;
            dest_array[i] = min_val;
        }
    }
}

__global__ void gpu_kernel(u32 *data,
                           u32 num_lists,
                           u32 num_elements)
{
    merge_array_gpu(data,
                    data,
                    num_lists,
                    num_elements);
}

// =========================== MAIN ===========================

int main()
{
    const u32 num_lists = NUM_LISTS;
    const u32 num_elements = NUM_ELEM;

    size_t size = num_elements * sizeof(u32);

    u32 *h_input = (u32*)malloc(size);
    u32 *h_output = (u32*)malloc(size);

    for (u32 i = 0; i < num_elements; i++)
        h_input[i] = rand();

    // ================= CPU TIMING =================
    auto cpu_start = std::chrono::high_resolution_clock::now();

    merge_array(h_input,
                h_output,
                num_lists,
                num_elements);

    auto cpu_end = std::chrono::high_resolution_clock::now();

    double cpu_time =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // ================= GPU TIMING =================
    u32 *d_data;
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data,
               h_input,
               size,
               cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    gpu_kernel<<<1, num_lists>>>(d_data,
                                 num_lists,
                                 num_elements);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaFree(d_data);

    // ================= PRINT TIMINGS =================

    printf("CPU Merge Time  : %.3f ms\n", cpu_time);
    printf("GPU Merge Time  : %.3f ms\n", gpu_time);
    printf("Speedup         : %.2fx\n", cpu_time / gpu_time);

    free(h_input);
    free(h_output);

    return 0;
}

