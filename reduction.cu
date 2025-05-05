// reduction.cu
// Owen Roseborough

// convergent kernel written to have the threads start reducing from the second part of the block
// the final element is stored in the last element of the array
__global__ void ConvergentSumReductionKernel(float* input, float* output){
    unsigned int i = threadIdx.x + blockDim.x;
    unsigned int len = blockDim.x * 2;
    for(unsigned int stride = blockDim.x; stride >= 1; stride /= 2){
        if(i > len - stride){
            input[i] += input[i - stride];
        }
        __syncthreads();
    }
    if(i == len - 1){
        *output = input[0];
    }
}

// kernel modified to be able to take input of length not multiple of COARSE_FACTOR*2*blockDim.x
__global__ void CoarsenedSumReductionKernel(float* input, float* output, int N){
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = COARSE_FACTOR*2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    float sum = 0.0f;
    if(i < N){
        sum = input[i];
    }
    for(unsigned int tile = 1; tile < COARSE_FACTOR*2; tile++){
        if(i + tile*BLOCK_DIM < N){
            sum += input[i + tile*BLOCK_DIM];
        }
    }
    input_s[t] = sum;
    for(unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2){
        __syncthreads();
        if(t < stride){
            input_s[t] += input_s[t + stride];
        }
    }
    if(t == 0){
        atomicAdd(output, input_s[0]);
    }
}

