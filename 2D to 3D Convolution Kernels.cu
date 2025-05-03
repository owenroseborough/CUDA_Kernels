// 2D to 3D Convolution Kernels.cu
// Owen Roseborough

// Basic 2D Convolutional Kernel:
__global__ void convolution_2D_basic_kernel(float *N, float *F, float *P, int r, int width, int height){
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    float Pvalue = 0.0f;
    for(int fRow = 0; fRow < 2*r+1; fRow++){
        for(int fCol = 0; fCol < 2*r+1; fCol++){
            inRow = outRow - r + fRow;
            inCol = outCol - r + fCol;
            if(inRow >= 0 && inRow < height && inCol >= 0 && inCol < width){
                Pvalue += F[fRow][fCol] * N[inRow*width + inCol];
            }
        }
    }
    P[outRow][outCol] = Pvalue;
}
// 3D Kernel version of above:
__global__ void convolution_3D_basic_kernel(float *N, float *F, float *P, int r, int width, int height, int depth){
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outDep = blockIdx.z * blockDim.z + threadIdx.z;
    float Pvalue = 0.0f;
    for(int fRow = 0; fRow < 2*r+1; fRow++){
        for(int fCol = 0; fCol < 2*r+1; fCol++){
            for(int fDep = 0; fDep < 2*r+1; fDep++){
                inRow = outRow - r + fRow;
                inCol = outCol - r + fCol;
                inDep = outDep - r + fDep;
                if(inRow >= 0 && inRow < height && inCol >= 0 && inCol < width && inDep >= 0 && inDep < depth){
                    Pvalue += F[fDep][fRow][fCol] * N[inDep * height * width + inRow * width + inCol];
                }
            }
        }
    }
    P[outDep * height * width + outRow * width + outCol] = Pvalue;
}
// 3D Tiled Convolution Kernel with constant memory for F:
#define IN_TILE_DIM 32
#define FILTER_RADIUS
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))
__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];
__global__ void convolution_tiled_3D_const_mem_kernel(float *N, float *P, int width, int height, int depth){
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    int dep = blockIdx.z * OUT_TILE_DIM + threadIdx.z - FILTER_RADIUS;
    // loading input tile
    __shared__ N_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    if(row >= 0 && row < height && col >= 0 && col < width && dep >= 0 && dep < depth){
        N_s[threadIdx.z][threadIdx.y][threadIdx.x] = N[dep*width*height + row*width + col];
    } else {
        N_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();
    // calculating output elements
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    int tileDep = threadIdx.z - FILTER_RADIUS;
    // turning off the threads at the edges of the block
    if(row >= 0 && row < height && col >= 0 && col < width && dep >= 0 && dep < depth){
        if(tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM && tileDep >= 0 && tileDep < OUT_TILE_DIM){
            float Pvalue = 0.0f;
            for(int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++){
                for(int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++){
                    for(int fDep = 0; fDep < 2*FILTER_RADIUS+1; fDep++){
                        Pvalue += F_c[fDep][fRow][fCol] * N_s[tileDep+fDep][tileRow+fRow][tileCol+fCol];
                    }
                }
            }
            P[dep * height * width + row * width + col] = Pvalue;
        }
    }
}