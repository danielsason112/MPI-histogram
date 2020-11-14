#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

__global__ void init_arr(int *arr, int size) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < size) {
		arr[tid] = 0;
	}
}

__global__  void kernel_calc_hist(int *arr,int *global_hist, int numOfElements) {
    	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ int shared_hist[RANGE + 1];

    	// Initialize shared histogram values to zero 
        shared_hist[tid] = 0;

	 __syncthreads();

	// Each thread updates shared histogram by the data value.
	// atomicAdd used to avoid race condition between threads in the same block
	if (tid < numOfElements)
	{
		atomicAdd(&shared_hist[arr[tid]], 1);
	}

	__syncthreads();

	
	// Each thread updates global histogram
	if (shared_hist[tid + 1] != 0) {
		global_hist[tid + 1] += shared_hist[tid + 1];
	}
}


int cuda_task(int* histogram, int* data, int numOfElements) {
	//printf("hello world:  %d %d    |   ", numOfElements, data[1]);

	cudaError_t err = cudaSuccess;

	size_t size = numOfElements * sizeof(int);

	int *d_data;
	int *d_hist;
	int *temp_hist;

	// Allocate space on host for saving histogram value after kernel_calc_hist
	temp_hist = (int*)calloc(RANGE + 1, sizeof(int));

	// Allocate spcae on device for d_hist
	err = cudaMalloc((void**)&d_hist, (RANGE + 1) * sizeof(int));
	if (err != cudaSuccess) {
    	    fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

	// allocate space on device for inputs data
    	err = cudaMalloc((void **)&d_data, size);
    	if (err != cudaSuccess) {
    	    fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

	// Copy data from host to the GPU memory
    	err = cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);
    	if (err != cudaSuccess) {
        	fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}


    	int threadsPerBlock = RANGE;
    	int blocksPerGrid =(numOfElements + threadsPerBlock - 1) / threadsPerBlock;

	// Initialize device histogram values to zero
	init_arr<<<1, RANGE + 1>>>(d_hist, RANGE + 1);
	err = cudaDeviceSynchronize();

	// Calaulate histogram for each block
    	kernel_calc_hist<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_hist, numOfElements);
    	err = cudaGetLastError();
    	if (err != cudaSuccess) {
    	    	fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

	size = (RANGE + 1) * sizeof(int);

	// Copy d_hist from device to host
	err = cudaMemcpy(temp_hist, d_hist, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
        	fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

	// Update final histogram
	for (int i=0; i < RANGE + 1; i++)
	{
		histogram[i] += temp_hist[i];
	}


	// Free allocated memory on GPU
	if (cudaFree(d_hist) != cudaSuccess) {
        	fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}
    	if (cudaFree(d_data) != cudaSuccess) {
        	fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

	// Free allocated memory on host
	free(temp_hist);

    return 0;
}

