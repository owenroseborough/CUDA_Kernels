// segmented parallel scan.cu
// Owen Roseborough

#include <iostream>
#include <string>
#include <cassert>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <time.h>
#include <math.h>
#include "cuda macros.h"

using namespace std;

__global__ void firstKernel(unsigned int *input_d, unsigned int *output_d, int maxThreadsX);

#define INPUTSIZE 1000000

int main(void)
{
    unsigned int *input_h;
    input_h = (unsigned int *)mallocCheckError(malloc(sizeof(unsigned int) * INPUTSIZE));

    // randomly initialize input_h
    srand(time(NULL));
    for(unsigned int i = 0; i < INPUTSIZE; i++){
        input_h[i] = (rand() % 5) + 1;  // rand() % 5 gives 0–4, +1 makes it 1–5
    }

    unsigned int *output_h;
    output_h = (unsigned int *)mallocCheckError(malloc(sizeof(unsigned int) * INPUTSIZE));
    
    // determine size of kernel blocks and grid
    int devCount;
    cudaCheckError(cudaGetDeviceCount(&devCount));
    cudaDeviceProp devProp;
    int maxThreadsX = 0;
    int device = 0;
    for(unsigned int i = 0; i < devCount; i++){
        cudaCheckError(cudaGetDeviceProperties(&devProp, i));
        if(devProp.maxThreadsDim[0] > maxThreadsX){
            maxThreadsX = devProp.maxThreadsDim[0];
            device = i;
        }
    }
    cudaCheckError(cudaSetDevice(device)); //set device to one with max thread capability in x axis
    
    // allocate input & output arrays on device
    unsigned int *input_d, *output_d;     // pointers to device memory
    cudaCheckError(cudaMalloc((void **) &input_d, sizeof(unsigned int) * INPUTSIZE));
    cudaCheckError(cudaMalloc((void **) &output_d, sizeof(unsigned int) * INPUTSIZE));
    
    // send input_h from host to device
    cudaCheckError(cudaMemcpy(input_d, input_h, sizeof(unsigned int) * INPUTSIZE, cudaMemcpyHostToDevice));

    dim3 blockSize(maxThreadsX, 1);
    dim3 gridSize(ceil(INPUTSIZE / maxThreadsX),1);

    //call segmented parallel scan kernels
    firstKernel<<<gridSize, blockSize, 2*maxThreadsX*sizeof(unsigned int)>>>(input_d, output_d, maxThreadsX);

    cudaCheckKernel();

    // retrieve output_d from device
    cudaCheckError(cudaMemcpy(output_h, output_d, sizeof(unsigned int) * INPUTSIZE, cudaMemcpyDeviceToHost));

    // compute scan sequentially on host
    unsigned int *test_output_h;
    test_output_h = (unsigned int *)mallocCheckError(malloc(sizeof(unsigned int) * INPUTSIZE));
    test_output_h[0] = input_h[0];
    for(unsigned int i = 1; i < INPUTSIZE; i++){
        test_output_h[i] = test_output_h[i - 1] + input_h[i];
    }
    // compare sequential result to parallel to ensure they are the same
    for(unsigned int i = 0; i < INPUTSIZE; i++){
        assert(test_output_h[i] == output_h[i]);
    }

    // cleanup
    free(input_h); free(output_h); free(test_output_h);
    cudaFree(input_d); cudaFree(output_d);
}

__global__ void firstKernel(unsigned char *input_d, unsigned char *output_d, int maxThreadsX){

    extern __shared__ unsigned int buffer[];
    unsigned int *firstBuf = buffer;
    unsigned int *secondBuf = &firstBuf[maxThreadsX];

    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < INPUTSIZE){
        firstBuf[threadIdx.x] = input_d[i];
    } else{
        firstBuf[threadIdx.x] = 0.0f;
    }
    unsigned int counter = 0;
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        __syncthreads();
        if(counter % 2 == 0 && threadIdx.x >= stride)
            secondBuf[threadIdx.x] = firstBuf[threadIdx.x] + firstBuf[threadIdx.x - stride];
        else if(threadIdx.x >= stride)
            firstBuf[threadIdx.x] = secondBuf[threadIdx.x] + secondBuf[threadIdx.x - stride];
        counter += 1;
    }
    if(i < INPUTSIZE){
        if(counter % 2 == 0){ 
            output_d[i] = firstBuf[threadIdx.x];
        } else {
            output_d[i] = secondBuf[threadIdx.x];
        }
    }
}
