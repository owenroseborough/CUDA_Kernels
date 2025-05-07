#include <stdio.h>
#include <cuda.h>

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr, "CUDA Error: %s in %s at line %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define cudaCheckKernel() { gpuKernelCheck(__FILE__, __LINE__); } // Use this after kernel launches to check for errors
inline void gpuKernelCheck(const char *file, int line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Kernel Launch Error: %s at %s:%d\n",
                cudaGetErrorString(err), file, line);
        exit(err);
    }

    err = cudaDeviceSynchronize(); // Optional: sync to catch async errors
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Sync Error: %s at %s:%d\n",
                cudaGetErrorString(err), file, line);
        exit(err);
    }
}