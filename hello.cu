// The notebook expects to load this CUDA kernel from the root of your Google Drive.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__constant__ char d_message[64];

__global__ void welcome(char* msg) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    msg[idx] = d_message[idx];
}

void printErrors(const char* label) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", label, cudaGetErrorString(err));
    }
}

int main() {
    printf("Hello CUDA from CPU\n");
  
    char* d_msg;
    char* h_msg;
    const char message[] = "Hello CUDA from GPU!";
    const int length = strlen(message) + 1;

    // Allocate host and device memory
    h_msg = (char*)std::malloc(length * sizeof(char));
    cudaMalloc(&d_msg, length * sizeof(char));
    
    // Copy message to constant memory
    cudaMemcpyToSymbol(d_message, message, length);
    
    // Run CUDA kernel and wait till it's done
    welcome<<<1, length>>>(d_msg);
    printErrors("Kernel launch failed");

    // Copy result back to host
    cudaMemcpy(h_msg, d_msg, length * sizeof(char), cudaMemcpyDeviceToHost);
    h_msg[length-1] = '\0';
    printErrors("Device->Host memcpy failed");

    std::printf("%s\n", h_msg);
    std::printf("Exiting kernel\n");
    
    // Cleanup
    std::free(h_msg);
    cudaFree(d_msg);
    
    return 0;
}
