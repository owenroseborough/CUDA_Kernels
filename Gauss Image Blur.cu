// Gauss Image Blur.cu
// Owen Roseborough

#include <iostream>
#include <string>
#include <cassert>
#include <ctime>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

#define GAUSIZE 2

__global__ void gaussBlurKernel(unsigned char *in, unsigned char *out, int w, int h, int channels);
__constant__ float gaussian_d[25];

int main(void)
{
   int width, height, channels;
   unsigned char *img_h = stbi_load("image.png", &width, &height, &channels, 0);
   if (img_h == NULL) {
      printf("Error loading image\n");
      return 1;
   }

   //Gaussian Array
   float gaussian_h[] = {0.003, 0.013, 0.022, 0.013, 0.003,
                        0.013, 0.059, 0.097, 0.059, 0.013,
                        0.022, 0.097, 0.159, 0.097, 0.022,
                        0.013, 0.059, 0.097, 0.059, 0.013,
                        0.003, 0.013, 0.022, 0.013, 0.003};
   cudaCheckError(cudaMemcpyToSymbol(gaussian_d, gaussian_h, sizeof(float) * 25));

   //Gaussian Array

   unsigned char *img_out_h;        // pointers to host memory
   unsigned char *in_d, *out_d;     // pointers to device memory

   // allocate output image on host
   img_out_h = (unsigned char *)malloc(sizeof(unsigned char) * width * height * channels);
   // allocate input & output image arrays on device
   cudaCheckError(cudaMalloc((void **) &in_d, sizeof(unsigned char) * width * height * channels));
   cudaCheckError(cudaMalloc((void **) &out_d, sizeof(unsigned char) * width * height * channels));
   
   cout << "Sending image from host to device ..." << endl;
   // send image from host to device: img_h to in_d
   cudaCheckError(cudaMemcpy(in_d, img_h, sizeof(unsigned char) * width * height * channels, cudaMemcpyHostToDevice));
   cout << "Copying within device ..." << endl;
   
   //call Gaussian Blur Kernel
   dim3 blockSize(16, 16);
   dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);
   gaussBlurKernel<<<gridSize, blockSize>>>(in_d, out_d, width, height, channels);
   //call Gaussian Blur Kernel

   cudaCheckKernel();

   cout << "Retrieving image from device to host ..." << endl;
   // retrieve image from device: out_d to img_out_h
   cudaCheckError(cudaMemcpy(img_out_h, out_d, sizeof(unsigned char) * width * height * channels, cudaMemcpyDeviceToHost));
   
   // Save the output image
   if (stbi_write_png("output.png", width, height, channels, img_out_h, width * channels)) {
      printf("Saved output image as output.png\n");
   } else {
      fprintf(stderr, "Failed to save image.\n");
   }

   cout << "Comparing original and copied data..." << endl;
   cout << "Cleaning up ..." << endl;
   // cleanup
   stbi_image_free(img_h);
   free(img_out_h);
   cudaFree(in_d); cudaFree(out_d);
}

__global__
void gaussBlurKernel(unsigned char *in, unsigned char *out, int w, int h, int channels){
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   if(col < w && row < h){
      //gaussian kernel
      for(int ch = 0; ch < channels; ch++){ //for every channel we do the kernel
         float pixVal = 0.0f;
         for(int gauRow = 0; gauRow < GAUSIZE * 2 + 1; gauRow++){
            for(int gauCol = 0; gauCol < GAUSIZE * 2 + 1; gauCol++){
               int curRow = row - GAUSIZE + gauRow;
               int curCol = col - GAUSIZE + gauCol;
               int rgbOffset = (curRow * w + curCol) * channels;
               if(curRow >= 0 && curRow < h && curCol >= 0 && curCol < w){
                  pixVal += in[rgbOffset + ch] * gaussian_d[gauRow * (GAUSIZE * 2 + 1) + gauCol];
               }
            }
         }
         out[(row * w + col) * channels + ch] = min(max(int(pixVal), 0), 255);
      }
      //gaussian kernel
   }
}
