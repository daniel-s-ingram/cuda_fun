#include <stdio.h>
#include <time.h>

#define N 100000000
#define THREADS_PER_BLOCK 512  

__global__ void gpu_block_add(int *a, int *b, int *c)
{
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

__global__ void gpu_thread_add(int *a, int *b, int *c)
{
	c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

__global__ void gpu_both_add(int *a, int *b, int *c, int n)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n)
	{
		c[index] = a[index] + b[index];
	} 
}

void cpu_add(int *a, int *b, int *c, long n)
{
	for (int i = 0; i < n-1; i++)
	{
		c[i] = a[i] + b[i];
	}
}

void random_array(int *a, long n)
{
	for (long i = 0; i < n-1; i++)
	{
		a[i] = rand() % 1000;
	}
}

int main(void)
{
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	long size = sizeof(int)*N;
	clock_t initial, final;
	double elapsed;

	//Allocate space on host for copies of a, b, and c
	a = (int *)malloc(size);
	b = (int *)malloc(size);
	c = (int *)malloc(size);

	//Allocate space on device for copies of a, b, and c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	//Initialize a and b with random values
	random_array(a, N);
	random_array(b, N);
	
	//Copy a and b to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	//Wait for copy to finish before trying to time anything
	cudaDeviceSynchronize();

	//Comparing only the time taken for the actual addition
	//Ignoring the time taken to copy vectors to and from device

	printf("\nCPU vs. GPU: Adding Two %dx1 Vectors\n", N);
	printf("=================================================\n");

	//Add a and b on the device in parallel using only blocks
	initial = clock();
	gpu_block_add<<<N,1>>>(d_a, d_b, d_c);
	cudaDeviceSynchronize(); //Hangs up until device has finished (to get more accurate reading of time for kernel execution)
	final = clock();
	elapsed = (double)(final - initial) / CLOCKS_PER_SEC;
	printf("GPU blocks only:\t\t%.3e seconds\n", elapsed);

	//Add a and b on the device in parallel using only threads
	initial = clock();
	gpu_thread_add<<<1,N>>>(d_a, d_b, d_c);
	cudaDeviceSynchronize();
	final = clock();
	elapsed = (double)(final - initial) / CLOCKS_PER_SEC;
	printf("GPU threads only:\t\t%.3e seconds\n", elapsed);

	//Add a and b on the device in parallel using blocks and threads
	initial = clock();
	gpu_both_add<<<(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
	cudaDeviceSynchronize();
	final = clock();
	elapsed = (double)(final - initial) / CLOCKS_PER_SEC;
	printf("GPU blocks and threads:\t\t%.3e seconds\n", elapsed);

	//Add a and b on the host
	initial = clock();
	cpu_add(a, b, c, N);
	final = clock();
	elapsed = (double)(final - initial) / CLOCKS_PER_SEC;
	printf("CPU:\t\t\t\t%.3e seconds\n\n", elapsed);

	//Free memory
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	return 0;
}