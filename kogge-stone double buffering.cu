__global__ void Kogge_Stone_scan_kernel(float *X, float *Y, unsigned int N){
    __shared__ float firstBuf[SECTION_SIZE];
    __shared__ float secondBuf[SECTION_SIZE];
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N){
        firstBuf[threadIdx.x] = X[i];
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
    if(i < N){
        if(counter % 2 == 0){ 
            Y[i] = firstBuf[threadIdx.x];
        } else {
            Y[i] = secondBuf[threadIdx.x];
        }
    }
}