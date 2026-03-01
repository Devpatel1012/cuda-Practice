#include <stdio.h>
#include <cuda_runtime.h>

#define ARRAY_SIZE 128

__global__ void whatIsMyID(unsigned int *block,
                           unsigned int *thread,
                           unsigned int *warp,
                           unsigned int *calcThread,
                           unsigned int N)
{
    unsigned int threadInd = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadInd >= N) return;

    block[threadInd]      = blockIdx.x;
    thread[threadInd]     = threadIdx.x;
    warp[threadInd]       = threadIdx.x / warpSize;
    calcThread[threadInd] = threadInd;
}

int main()
{
    const unsigned int numBlocks  = 2;
    const unsigned int numThreads = 64;
    const unsigned int N = ARRAY_SIZE;

    unsigned int *h_block, *h_thread, *h_warp, *h_calcThread;
    unsigned int *d_block, *d_thread, *d_warp, *d_calcThread;

    size_t bytes = N * sizeof(unsigned int);

    // Host allocations
    h_block      = (unsigned int*)malloc(bytes);
    h_thread     = (unsigned int*)malloc(bytes);
    h_warp       = (unsigned int*)malloc(bytes);
    h_calcThread = (unsigned int*)malloc(bytes);

    // Device allocations
    cudaMalloc(&d_block, bytes);
    cudaMalloc(&d_thread, bytes);
    cudaMalloc(&d_warp, bytes);
    cudaMalloc(&d_calcThread, bytes);

    // Kernel launch
    whatIsMyID<<<numBlocks, numThreads>>>(
        d_block, d_thread, d_warp, d_calcThread, N
    );

    // Error checking + sync
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy back
    cudaMemcpy(h_block,      d_block,      bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_thread,     d_thread,     bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_warp,       d_warp,       bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_calcThread, d_calcThread, bytes, cudaMemcpyDeviceToHost);

    // Print results
    for (unsigned int i = 0; i < N; i++) {
        printf("GlobalThread: %3u | Block: %u | Warp: %u | Thread: %u\n",
               h_calcThread[i],
               h_block[i],
               h_warp[i],
               h_thread[i]);
    }

    // Cleanup
    cudaFree(d_block);
    cudaFree(d_thread);
    cudaFree(d_warp);
    cudaFree(d_calcThread);

    free(h_block);
    free(h_thread);
    free(h_warp);
    free(h_calcThread);

    return 0;
}

