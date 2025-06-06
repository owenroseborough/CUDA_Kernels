// matrix vector mul.cu
// Owen Roseborough

#include <iostream>
#include <string>
#include <cassert>
#include <ctime>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>

using namespace std;

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

__global__ void matrixVectorMulKernel(float *A_d, float *B_d, float *C_d, int N);

int main(void)
{
   int N = 1024; 
   float *A_h = (float*)malloc(N * N * sizeof(float));
   float *B_h = (float*)malloc(N * sizeof(float));
   float *C_h = (float*)malloc(N * sizeof(float));
   if (A_h == NULL || B_h == NULL || C_h == NULL){
      cout << "error allocating host memory";
      return 0;
   }
   for (int i = 0; i < N * N; i++) {
      A_h[i] = 1.0f;
   }
   for (int i = 0; i < N; i++) {
      B_h[i] = 2.0f;
   }

     
   float *A_d; float *B_d; float *C_d; // pointers to device memory

   // allocate matrix and vector arrays on device
   cudaCheckError(cudaMalloc((void **) &A_d, sizeof(float) * N * N));
   cudaCheckError(cudaMalloc((void **) &B_d, sizeof(float) * N));
   cudaCheckError(cudaMalloc((void **) &C_d, sizeof(float) * N));
   
   // transfering A_h and B_h to device
   cudaCheckError(cudaMemcpy(A_d, A_h, sizeof(float) * N * N, cudaMemcpyHostToDevice));
   cudaCheckError(cudaMemcpy(B_d, B_h, sizeof(float) * N, cudaMemcpyHostToDevice));
   
   //call matrix mul kernel
   dim3 blockSize(32);
   dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
   matrixVectorMulKernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, N);

   cudaCheckKernel();

   // retrieve out matrix from device: C_d to C_h
   cudaCheckError(cudaMemcpy(C_h, C_d, sizeof(float) * N, cudaMemcpyDeviceToHost));

   // print resultant vector
   for(int i = 0; i < N; i++){
      cout << C_h[i] << " ";
   }

   // cleanup
   free(A_h); free(B_h); free(C_h);
   cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
}

__global__  
void matrixVectorMulKernel(float *A_d, float *B_d, float *C_d, int N){
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if(index < N){
      //one thread produces an element in C_d
      float Pvalue = 0;
      for(int k = 0; k < N; k++){
         Pvalue += A_d[index*N + k] * B_d[k];
      }
      C_d[index] = Pvalue;
   }
}
