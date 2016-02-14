/* Udacity Homework 3
HDR Tone-mapping
Background HDR
==============
A High Dynamic Range (HDR) image contains a wider variation of intensity
and color than is allowed by the RGB format with 1 byte per channel that we
have used in the previous assignment.
To store this extra information we use single precision floating point for
each channel.  This allows for an extremely wide range of intensity values.
In the image for this assignment, the inside of church with light coming in
through stained glass windows, the raw input floating point values for the
channels range from 0 to 275.  But the mean is .41 and 98% of the values are
less than 3!  This means that certain areas (the windows) are extremely bright
compared to everywhere else.  If we linearly map this [0-275] range into the
[0-255] range that we have been using then most values will be mapped to zero!
The only thing we will be able to see are the very brightest areas - the
windows - everything else will appear pitch black.
The problem is that although we have cameras capable of recording the wide
range of intensity that exists in the real world our monitors are not capable
of displaying them.  Our eyes are also quite capable of observing a much wider
range of intensities than our image formats / monitors are capable of
displaying.
Tone-mapping is a process that transforms the intensities in the image so that
the brightest values aren't nearly so far away from the mean.  That way when
we transform the values into [0-255] we can actually see the entire image.
There are many ways to perform this process and it is as much an art as a
science - there is no single "right" answer.  In this homework we will
implement one possible technique.
Background Chrominance-Luminance
================================
The RGB space that we have been using to represent images can be thought of as
one possible set of axes spanning a three dimensional space of color.  We
sometimes choose other axes to represent this space because they make certain
operations more convenient.
Another possible way of representing a color image is to separate the color
information (chromaticity) from the brightness information.  There are
multiple different methods for doing this - a common one during the analog
television days was known as Chrominance-Luminance or YUV.
We choose to represent the image in this way so that we can remap only the
intensity channel and then recombine the new intensity values with the color
information to form the final image.
Old TV signals used to be transmitted in this way so that black & white
televisions could display the luminance channel while color televisions would
display all three of the channels.

Tone-mapping
============
In this assignment we are going to transform the luminance channel (actually
the log of the luminance, but this is unimportant for the parts of the
algorithm that you will be implementing) by compressing its range to [0, 1].
To do this we need the cumulative distribution of the luminance values.
Example
-------
input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
min / max / range: 0 / 9 / 9
histo with 3 bins: [4 7 3]
cdf : [4 11 14]
Your task is to calculate this cumulative distribution by following these
steps.
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "device_atomic_functions.hpp"

#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

#include "utils.h"

__global__
void histogram_kernel(unsigned int* d_bins, const float* d_in, const int bin_count, const float lum_min, const float lum_max, const int size) {
	//extern __shared__ unsigned int local_bin[];

	int mid = threadIdx.x + blockDim.x * blockIdx.x;

	int index = threadIdx.x;

	if (mid >= size)
		return;

	////Initialize shared local_bin
	//local_bin[index] = 0;

	float lum_range = lum_max - lum_min;
	int bin = ((d_in[mid] - lum_min) / lum_range) * bin_count;

	////atomic operation works for both global and shared memory
	////The bottleneck of atomic operation is caused by wait time of threads trying to access same global memory
	//atomicAdd(&local_bin[bin], 1);
	////Use one thread of each block to add to the global array
	//if (index == 0)
	//{
	//	//Use reduce to add up all variables, there are blockDim.x arrays
	//	//This might be more efficient if the bin size is small
	//}

	atomicAdd(&d_bins[bin], 1);
}

//Scan is used to obtain cdf using pdf/histogram
__global__
void scan_kernel(unsigned int* d_bins, int size) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	if (x >= size)
		return;

	for (int hop = 1; hop <= size; hop *= 2) {
		//operation starts from the first entry whose index exceeds hop
		unsigned int val = 0;
		if (x >= hop)
			//Get the value 1 hop before
			val = d_bins[x-hop];
		__syncthreads();
		if (x >= hop)
			//Add the value hop before to the current value
			d_bins[x] += val;
		__syncthreads();

	}
}

__global__
void reduce_kernel(const float* const d_in, float* d_out, const size_t size, int operationType) {
	extern __shared__ float shared[];

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int index = threadIdx.x;

	if (x >= size)
		return;
	
	shared[index] = d_in[x];
	
	__syncthreads();

	//Calculate the min/max/sum of each block
	//shared[] is independent between blocks
	for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
		if (index < s) {
			//operationType==0: min
			//operationType==1: max
			if (operationType == 0) {
				shared[index] = min(shared[index], shared[index + s]);
			}
			else {
				shared[index] = max(shared[index], shared[index + s]);
			}
		}

		__syncthreads();
	}

	//The first thread of each block is in charging of writing the min/max/sum of the block into d_out
	if (index == 0) {
		d_out[blockIdx.x] = shared[0];
	}
}

int get_max_size(int n, int d) {
	return (int)ceil((float)n / (float)d) + 1;
}

float reduce(const float* const d_in, const size_t size, int operationType) {
	int BLOCK_SIZE = 32;
	// we need to keep reducing until we get to the amount that we consider 
	// having the entire thing fit into one block size
	size_t curr_size = size;
	float* d_curr_in;

	checkCudaErrors(cudaMalloc(&d_curr_in, sizeof(float) * size));
	checkCudaErrors(cudaMemcpy(d_curr_in, d_in, sizeof(float) * size, cudaMemcpyDeviceToDevice));


	float* d_curr_out;

	dim3 thread_dim(BLOCK_SIZE);
	const int shared_mem_size = sizeof(float)*BLOCK_SIZE;

	while (1) {
		checkCudaErrors(cudaMalloc(&d_curr_out, sizeof(float) * get_max_size(curr_size, BLOCK_SIZE)));

		dim3 block_dim(get_max_size(size, BLOCK_SIZE));
		reduce_kernel << <block_dim, thread_dim, shared_mem_size >> >(
			d_curr_in,
			d_curr_out,
			curr_size,
			operationType
			);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		checkCudaErrors(cudaFree(d_curr_in));
		d_curr_in = d_curr_out;

		//If curr_size already less than block_size, then there is only one block left, so the value from d_curr_out should be the final value
		if (curr_size <  BLOCK_SIZE)
			break;

		//After obtain the min/max/sum of each block, reduce the size of the next output array to size/blockWidth
		curr_size = get_max_size(curr_size, BLOCK_SIZE);
	}

	// copy value to host and return
	float h_out;
	cudaMemcpy(&h_out, d_curr_out, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_curr_out);
	return h_out;
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
	unsigned int* const d_cdf,
	float &min_logLum,
	float &max_logLum,
	const size_t numRows,
	const size_t numCols,
	const size_t numBins)
{
	const size_t size = numRows*numCols;
	min_logLum = reduce(d_logLuminance, size, 0);
	max_logLum = reduce(d_logLuminance, size, 1);

	printf("got min of %f\n", min_logLum);
	printf("got max of %f\n", max_logLum);
	printf("numBins %d\n", numBins);

	unsigned int* d_bins;
	//histo_size is the size of an unsigned int (sizer_t) array with length of numBins
	size_t histo_size = sizeof(unsigned int)*numBins;

	checkCudaErrors(cudaMalloc(&d_bins, histo_size));
	//Initialize all entries with 0
	checkCudaErrors(cudaMemset(d_bins, 0, histo_size));

	//1024*gridWidth threads operating on 1024*gridWidth array for bins[1024]
	//The array contains 1024 bins is locked for each of 1024*gridWidth threads

	//Instead, consider each block has the size of bin size, which is 1024 in this case, blockWidth=1024
	//then there are size/blockWidth blocks in total
	//Each block has a shared bin with the size of 1024
	//Threads inside of each block are updating the shared array locally
	dim3 blockWidth(numBins);
	dim3 hist_grid_dim(get_max_size(size, blockWidth.x));
	//const int shared_mem_size = sizeof(unsigned int)*blockWidth.x;
	//atomic addition kernel
	histogram_kernel << <hist_grid_dim, blockWidth>> >(d_bins, d_logLuminance, numBins, min_logLum, max_logLum, size);
	//	, shared_mem_size >> >(d_bins, d_logLuminance, numBins, min_logLum, max_logLum, size);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	//unsigned int h_out[100];
	//cudaMemcpy(&h_out, d_bins, sizeof(unsigned int) * 100, cudaMemcpyDeviceToHost);
	//for (int i = 0; i < 100; i++)
	//	printf("hist out %d\n", h_out[i]);

	//scan is acting on histogram with the size of numBins
	dim3 scan_block_dim(get_max_size(numBins, blockWidth.x));

	scan_kernel << <scan_block_dim, blockWidth >> >(d_bins, numBins);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	//cudaMemcpy(&h_out, d_bins, sizeof(unsigned int) * 100, cudaMemcpyDeviceToHost);
	//for (int i = 0; i < 100; i++)
	//	printf("cdf out %d\n", h_out[i]);


	cudaMemcpy(d_cdf, d_bins, histo_size, cudaMemcpyDeviceToDevice);


	checkCudaErrors(cudaFree(d_bins));


	//TODO
	/*Here are the steps you need to implement
	1) find the minimum and maximum value in the input logLuminance channel
	store in min_logLum and max_logLum
	2) subtract them to find the range
	3) generate a histogram of all the values in the logLuminance channel using
	the formula: bin = (lum[i] - lumMin) / lumRange * numBins
	4) Perform an exclusive scan (prefix sum) on the histogram to get
	the cumulative distribution of luminance values (this should go in the
	incoming d_cdf pointer which already has been allocated for you)       */
}
