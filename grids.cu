#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void whatIsMyID2DA(
    unsigned int *block_x,
    unsigned int *block_y,
    unsigned int *thread,
    unsigned int *calcThread,
    unsigned int *x_thread,
    unsigned int *y_thread,
    unsigned int *grid_dimx,
    unsigned int *block_dimx,
    unsigned int *grid_dimy,
    unsigned int *block_dimy)
{
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const unsigned int thread_idx = ((gridDim.x * blockDim.x) * idy) + idx;

    block_x[thread_idx] = blockIdx.x;
    block_y[thread_idx] = blockIdx.y;
    thread[thread_idx] = threadIdx.x;
    calcThread[thread_idx] = thread_idx;
    x_thread[thread_idx] = idx;
    y_thread[thread_idx] = idy;
    grid_dimx[thread_idx] = gridDim.x;
    block_dimx[thread_idx] = blockDim.x;
    grid_dimy[thread_idx] = gridDim.y;
    block_dimy[thread_idx] = blockDim.y;
}

#define ARRAY_SIZE_X 32
#define ARRAY_SIZE_Y 16
#define ARRAY_SIZE_IN_BYTES (ARRAY_SIZE_X * ARRAY_SIZE_Y * sizeof(unsigned int))

int main()
{
    const dim3 threads_rect(32, 4);
    const dim3 blocks_rect(1, 4);

    const dim3 thread_square(16, 8);
    const dim3 blocks_square(2, 2);

    // Host memory
    unsigned int cpuBlockX[ARRAY_SIZE_Y][ARRAY_SIZE_X];
    unsigned int cpuBlockY[ARRAY_SIZE_Y][ARRAY_SIZE_X];
    unsigned int cpuThread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
    unsigned int cpucalcThread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
    unsigned int cpuxthread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
    unsigned int cpuythread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
    unsigned int cpugriddimx[ARRAY_SIZE_Y][ARRAY_SIZE_X];
    unsigned int cpublockdimx[ARRAY_SIZE_Y][ARRAY_SIZE_X];
    unsigned int cpugriddimy[ARRAY_SIZE_Y][ARRAY_SIZE_X];
    unsigned int cpublockdimy[ARRAY_SIZE_Y][ARRAY_SIZE_X];

    // Device memory
    unsigned int *gpuBlockX, *gpuBlockY, *gpuThread, *gpuCalcThread;
    unsigned int *gpuXThread, *gpuYThread;
    unsigned int *gpuGriddimX, *gpuBlockdimX;
    unsigned int *gpuGriddimY, *gpuBlockdimY;

    cudaMalloc(&gpuBlockX, ARRAY_SIZE_IN_BYTES);
    cudaMalloc(&gpuBlockY, ARRAY_SIZE_IN_BYTES);
    cudaMalloc(&gpuThread, ARRAY_SIZE_IN_BYTES);
    cudaMalloc(&gpuCalcThread, ARRAY_SIZE_IN_BYTES);
    cudaMalloc(&gpuXThread, ARRAY_SIZE_IN_BYTES);
    cudaMalloc(&gpuYThread, ARRAY_SIZE_IN_BYTES);
    cudaMalloc(&gpuGriddimX, ARRAY_SIZE_IN_BYTES);
    cudaMalloc(&gpuBlockdimX, ARRAY_SIZE_IN_BYTES);
    cudaMalloc(&gpuGriddimY, ARRAY_SIZE_IN_BYTES);
    cudaMalloc(&gpuBlockdimY, ARRAY_SIZE_IN_BYTES);

    for (int kernel = 0; kernel < 2; kernel++)
    {
        if (kernel == 0)
        {
            whatIsMyID2DA<<<blocks_rect, threads_rect>>>(
                gpuBlockX, gpuBlockY, gpuThread, gpuCalcThread,
                gpuXThread, gpuYThread,
                gpuGriddimX, gpuBlockdimX,
                gpuGriddimY, gpuBlockdimY);
        }
        else
        {
            whatIsMyID2DA<<<blocks_square, thread_square>>>(
                gpuBlockX, gpuBlockY, gpuThread, gpuCalcThread,
                gpuXThread, gpuYThread,
                gpuGriddimX, gpuBlockdimX,
                gpuGriddimY, gpuBlockdimY);
        }

        cudaDeviceSynchronize();

        cudaMemcpy(cpuBlockX, gpuBlockX, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpuBlockY, gpuBlockY, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpuThread, gpuThread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpucalcThread, gpuCalcThread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpuxthread, gpuXThread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpuythread, gpuYThread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpugriddimx, gpuGriddimX, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpublockdimx, gpuBlockdimX, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpugriddimy, gpuGriddimY, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpublockdimy, gpuBlockdimY, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

        printf("\n================ Kernel %d ================\n", kernel);

        for (int y = 0; y < ARRAY_SIZE_Y; y++)
        {
            for (int x = 0; x < ARRAY_SIZE_X; x++)
            {
                printf("CT:%3u BKX:%u BKY:%u TID:%2u Y:%2u X:%2u GDX:%u GDY:%u\n",
                       cpucalcThread[y][x],
                       cpuBlockX[y][x],
                       cpuBlockY[y][x],
                       cpuThread[y][x],
                       cpuythread[y][x],
                       cpuxthread[y][x],
                       cpugriddimx[y][x],
                       cpugriddimy[y][x]);
            }
        }
    }

    cudaFree(gpuBlockX);
    cudaFree(gpuBlockY);
    cudaFree(gpuThread);
    cudaFree(gpuCalcThread);
    cudaFree(gpuXThread);
    cudaFree(gpuYThread);
    cudaFree(gpuGriddimX);
    cudaFree(gpuBlockdimX);
    cudaFree(gpuGriddimY);
    cudaFree(gpuBlockdimY);

    return 0;
}
