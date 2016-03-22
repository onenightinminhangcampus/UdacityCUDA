
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <stdio.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "timer.h"

// Your job is to implemment a bitonic sort. A description of the bitonic sort
// can be see at:
// http://en.wikipedia.org/wiki/Bitonic_sort
int compareFloat(const void * a, const void * b)
{
	if (*(float*)a <  *(float*)b) return -1;
	if (*(float*)a == *(float*)b) return 0;
	if (*(float*)a >  *(float*)b) return 1;
	return 0;                     // should never reach this
}


__global__ void batcherBitonicMergesort64(float * d_out, const float * d_in)
{
	// you are guaranteed this is called with <<<1, 64, 64*4>>>
	extern __shared__ float sdata[];
	int tid = threadIdx.x;
	sdata[tid] = d_in[tid];
	__syncthreads();

	int total = 64;
	if (tid < total / 2)
	{
		for (int stage = 0; stage <= 5; stage++)
		{
			for (int substage = stage; substage >= 0; substage--)
			{
				if (substage == stage)
				{
					//Example
					//stage=0
					//tid=0, ind1=0, rem=0, hop=2

					//stage=1
					//tid=0, ind1=0, rem=0, hop=4
					//tid=1, ind1=1, rem=1, hop=2
					//tid=2, ind1=2+2=4, rem=0, hop=4
					//tid=3, ind1=5

					//stage=2
					//tid=4, ind1=8, rem=0, hop=8

					//stage=3
					//tid=5, ind1=5, rem=5, hop=6

					int intermediate = powf(2, stage);

					int ind1 = tid + tid / intermediate * intermediate;
					int rem = tid%intermediate;
					int hop = intermediate * 2 - rem * 2 - 1;

					int ind2 = ind1 + hop;

					if (sdata[ind1] > sdata[ind2])
					{
						float temp = sdata[ind1];
						sdata[ind1] = sdata[ind2];
						sdata[ind2] = temp;
					}

				}

				else
				{
					//Example
					//stage=2, substage=1
					//tid=0, hop=2, ind1=0, ind2=2
					//tid=3, hop=2, ind1=5, ind2=7

					//stage=3, substage=2
					//tid=0, hop=4, ind1=0, ind2=4
					//tid=5, hop=4, ind1=9, ind2=13
				
					int hop = powf(2,substage);

					int ind1 = tid + tid / hop * hop;
					int ind2 = ind1 + hop;

					if (sdata[ind1] > sdata[ind2])
					{
						float temp = sdata[ind1];
						sdata[ind1] = sdata[ind2];
						sdata[ind2] = temp;
					}

				}

				__syncthreads();


			}
		}

		d_out[tid + 32] = sdata[tid + 32];
		d_out[tid] = sdata[tid];

	}


}



int main(int argc, char **argv)
{
	const int ARRAY_SIZE = 64;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	// generate the input array on the host
	float h_in[ARRAY_SIZE];
	float h_sorted[ARRAY_SIZE];
	float h_out[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++) {
		// generate random float in [0, 1]
		h_in[i] = (float)rand() / (float)RAND_MAX;
		h_sorted[i] = h_in[i];
	}
	qsort(h_sorted, ARRAY_SIZE, sizeof(float), compareFloat);

	// declare GPU memory pointers
	float * d_in, *d_out;

	// allocate GPU memory
	cudaMalloc((void **)&d_in, ARRAY_BYTES);
	cudaMalloc((void **)&d_out, ARRAY_BYTES);

	// transfer the input array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// launch the kernel
	GpuTimer timer;
	timer.Start();
	batcherBitonicMergesort64 << <1, ARRAY_SIZE, ARRAY_SIZE * sizeof(float) >> >(d_out, d_in);
	cudaDeviceSynchronize();
	timer.Stop();

	printf("\nYour code executed in %g ms\n", timer.Elapsed());

	// copy back the sum from GPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// compare your result against the reference solution
	//compare(h_out, h_sorted, ARRAY_SIZE);

	for (int i = 0; i < 64; i++)
	{
		if (abs(h_out[i] - h_sorted[i]) > 0.001)
		{
			std::cout << i << std::endl;
		}

		std::cout << h_out[i] << " " << h_sorted[i] << " "<<h_in[i]<<std::endl;
	}


	// free GPU memory allocation
	cudaFree(d_in);
	cudaFree(d_out);
}