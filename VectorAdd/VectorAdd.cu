#include <stdio.h>
#include <time.h>

#define N 100000000

__global__ void gpu_add(int *a, int *b, int *c)
{
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
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
	clock_t initial, final, initial2, final2;
	double elapsed, elapsed2;

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

	//Initial time for entire device operation (host -> device, kernel execution, device -> host)
	initial = clock();
	
	//Copy a and b to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	//Initial time for kernel execution
	initial2 = clock();

	//Add a and b in parallel
	gpu_add<<<N,1>>>(d_a, d_b, d_c);
	cudaThreadSynchronize(); //Hangs up until device has finished (to get more accurate reading of time for kernel execution)

	final2 = clock();
	
	//Copy result back to device
	cudaMemcpy(d_c, c, size, cudaMemcpyDeviceToHost);
	final = clock();

	elapsed = (double)(final - initial) / CLOCKS_PER_SEC;
	elapsed2 = (double)(final2 - initial2) / CLOCKS_PER_SEC;
	printf("\nCopying vectors A and B, each of size %d, to the GPU, adding them in parallel on the GPU, and copying the result back to the host took %f seconds.\n", N, elapsed);
	printf("The addition took only %f seconds.\n\n", elapsed2);

	initial = clock();
	cpu_add(a, b, c, N);
	final = clock();

	elapsed = (double)(final - initial) / CLOCKS_PER_SEC;
	printf("Adding the same vectors A and B together took %f seconds on the CPU.\n\n", elapsed);

	//Free memory
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	return 0;
}