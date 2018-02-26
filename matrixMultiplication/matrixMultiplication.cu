#include <stdio.h>
#include <time.h>
#include <cuda.h>

#define M 400
#define N 400
#define R 400

#define THREADS_PER_BLOCK 512

__global__ void gpu_matmul(int *a, int *b, int *c, int m, int n, int r)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;

	int sum = 0;

	if (i < n && j < n)
	{
		for(int k = 0; k < n; k++)
		{
			sum += a[i*n+k]*b[k*n+j];
		}
	}

	c[i*n+j] = sum;
}

void cpu_matmul(int *a, int *b, int *c, int m, int n, int r)
{
	int sum;

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < r; j++)
		{
			sum = 0;
			for (int k = 0; k < n; k++)
			{
				sum += a[i*m+k]*b[k*n+j];
			}
			c[i*m+j] = sum;
		}
	}
}

void random_matrix(int *a, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			a[i*m+j] = rand() % 100;
		}
	}
}

int main(void)
{
	int *a, *b, *c, *c2; 
	int *d_a, *d_b, *d_c;
	long a_size, b_size, c_size;
	long size_int = sizeof(int);
	double elapsed;
	clock_t initial, final;
	cudaError_t error;

	a_size = M * N * size_int;
	b_size = N * R * size_int;
	c_size = M * R * size_int;

	//Allocate memory on host for arrays a, b, and c (flattened 2D arrays)
	a = (int *)malloc(a_size);
	b = (int *)malloc(b_size);
	c = (int *)malloc(c_size);
	c2 = (int *)malloc(c_size);

	//Allocate memory on device for a, b, and c
	if ((error = cudaMalloc((void **)&d_a, a_size)) != cudaSuccess)
	{
		printf("Error allocating d_a: %s in %s on line %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	if ((error = cudaMalloc((void **)&d_b, b_size)) != cudaSuccess)
	{
		printf("Error allocating d_b: %s in %s on line %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	if ((error = cudaMalloc((void **)&d_c, c_size)) != cudaSuccess)
	{
		printf("Error allocating d_c: %s in %s on line %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	random_matrix(a, M, N);
	random_matrix(b, N, R);

	if ((error = cudaMemcpy(d_a, a, a_size, cudaMemcpyHostToDevice)) != cudaSuccess)
	{
		printf("Error copying a to d_a: %s in %s on line %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	if ((error = cudaMemcpy(d_b, b, b_size, cudaMemcpyHostToDevice)) != cudaSuccess)
	{
		printf("Error copying b to d_b: %s in %s on line %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();

	printf("\nCPU vs. GPU: Multiplying %dx%d by %dx%d Matrix\n", M, N, N, R);
	printf("====================================================\n");

	initial = clock();
	gpu_matmul<<<1,4>>>(d_a, d_b, d_c, M, N, R);
	cudaDeviceSynchronize();
	final = clock();

	elapsed = (double)(final - initial) / CLOCKS_PER_SEC;
	printf("GPU:\t\t%e seconds\n", elapsed);

	initial = clock();
	cpu_matmul(a, b, c, M, N, R);
	final = clock();

	elapsed = (double)(final - initial) / CLOCKS_PER_SEC;
	printf("CPU:\t\t%e seconds\n\n", elapsed);

	cudaMemcpy(c2, d_c, c_size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < R; j++)
		{
			printf("%d\t\t%d\t\t%d\n", c[i*M+j], c2[i*M+j], c[i*M+j] == c2[i*M+j]);
		}
	}

	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	return 0;
}