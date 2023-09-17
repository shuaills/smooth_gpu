#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \ (((n) >> NUM_BANKS) + ((n) >> (2 * LOG_NUM_BANKS)))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

__global__
void gpu_sum_scan_blelloch(float* const d_out,
	const float* const d_in,
	float* const d_block_sums,
	const size_t numElems)
{
	extern __shared__ unsigned int s_out[];

	unsigned int glbl_tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Zero out shared memory
	// Especially important when padding shmem for
	//  non-power of 2 sized input
	s_out[threadIdx.x] = 0;
	s_out[threadIdx.x + blockDim.x] = 0;

	__syncthreads();

	// Copy d_in to shared memory per block
	unsigned int cpy_idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
	if (cpy_idx < numElems)
	{
		s_out[threadIdx.x] = d_in[cpy_idx];
		if (cpy_idx + blockDim.x < numElems)
			s_out[threadIdx.x + blockDim.x] = d_in[cpy_idx + blockDim.x];
	}

	__syncthreads();

	// Reduce/Upsweep step

	// 2^11 = 2048, the max amount of data a block can blelloch scan
	unsigned int max_steps = 11; 

	unsigned int r_idx = 0;
	unsigned int l_idx = 0;
	unsigned int sum = 0; // global sum can be passed to host if needed
	unsigned int t_active = 0;
	for (int s = 0; s < max_steps; ++s)
	{
		t_active = 0;

		// calculate necessary indexes
		// right index must be (t+1) * 2^(s+1)) - 1
		r_idx = ((threadIdx.x + 1) * (1 << (s + 1))) - 1;
		if (r_idx >= 0 && r_idx < 2048)
			t_active = 1;

		if (t_active)
		{
			// left index must be r_idx - 2^s
			l_idx = r_idx - (1 << s);

			// do the actual add operation
			sum = s_out[l_idx] + s_out[r_idx];
		}
		__syncthreads();

		if (t_active)
			s_out[r_idx] = sum;
		__syncthreads();
	}

	// Copy last element (total sum of block) to block sums array
	// Then, reset last element to operation's identity (sum, 0)
	if (threadIdx.x == 0)
	{
		d_block_sums[blockIdx.x] = s_out[r_idx];
		s_out[r_idx] = 0;
	}

	__syncthreads();

	// Downsweep step

	for (int s = max_steps - 1; s >= 0; --s)
	{
		// calculate necessary indexes
		// right index must be (t+1) * 2^(s+1)) - 1
		r_idx = ((threadIdx.x + 1) * (1 << (s + 1))) - 1;
		if (r_idx >= 0 && r_idx < 2048)
		{
			t_active = 1;
		}

		unsigned int r_cpy = 0;
		unsigned int lr_sum = 0;
		if (t_active)
		{
			// left index must be r_idx - 2^s
			l_idx = r_idx - (1 << s);

			// do the downsweep operation
			r_cpy = s_out[r_idx];
			lr_sum = s_out[l_idx] + s_out[r_idx];
		}
		__syncthreads();

		if (t_active)
		{
			s_out[l_idx] = r_cpy;
			s_out[r_idx] = lr_sum;
		}
		__syncthreads();
	}

	// Copy the results to global memory
	if (cpy_idx < numElems)
	{
		d_out[cpy_idx] = s_out[threadIdx.x];
		if (cpy_idx + blockDim.x < numElems)
			d_out[cpy_idx + blockDim.x] = s_out[threadIdx.x + blockDim.x];
	}
}


__global__ void compute_smoothed_list(float *prefix_sum, float *inlist, float *outlist, int n, int h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // start和end的计算应与Python实现一致
    int start = max(0, i - h);
    int end = min(n - 1, i + h);

    // 计算平滑窗口的和
    float sum = prefix_sum[end] - prefix_sum[start] + inlist[end];

    // 计算窗口内的元素数量
    int num_elements = end - start + 1;

    // 计算平均值
    outlist[i] = sum / num_elements;
}

std::vector<float> runSmoothListWithBlellochScan(const std::vector<float>& h_inlist, int h) {
    int n = h_inlist.size();
    std::vector<float> h_prefix_sum(n, 0.0f);  // host prefix sum
    std::vector<float> h_outlist(n);  // host smoothed list
    std::vector<float> h_block_sums((n + 2047) / 2048);  // for storing block-level sums

    float *d_inlist, *d_prefix_sum, *d_outlist, *d_block_sums;

    // Allocate device memory
    cudaMalloc((void**)&d_inlist, n * sizeof(float));
    cudaMalloc((void**)&d_prefix_sum, n * sizeof(float));
    cudaMalloc((void**)&d_outlist, n * sizeof(float));
    cudaMalloc((void**)&d_block_sums, h_block_sums.size() * sizeof(float));

    // Copy inlist from host to device
    cudaMemcpy(d_inlist, h_inlist.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // Configure and launch prefix sum kernel
    int blockSize = 1024;
    int gridSize = (n + 2 * blockSize - 1) / (2 * blockSize);
    int sharedMemSize = 2 * blockSize * sizeof(float);

    gpu_sum_scan_blelloch<<<gridSize, blockSize, sharedMemSize>>>(d_prefix_sum, d_inlist, d_block_sums, n);
    cudaDeviceSynchronize();

    // Copy prefix sum back to host
    cudaMemcpy(h_prefix_sum.data(), d_prefix_sum, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Configure and launch compute_smoothed_list kernel
    blockSize = 256;
    gridSize = (n + blockSize - 1) / blockSize;
    compute_smoothed_list<<<gridSize, blockSize>>>(d_prefix_sum, d_inlist, d_outlist, n, h);
    cudaDeviceSynchronize();

    // Copy smoothed list from device to host
    cudaMemcpy(h_outlist.data(), d_outlist, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_inlist);
    cudaFree(d_prefix_sum);
    cudaFree(d_outlist);
    cudaFree(d_block_sums);

    return h_outlist;
}



std::vector<float> runBlellochScan(const std::vector<float>& h_idata) {
    size_t numElems = h_idata.size();
    std::vector<float> h_odata(numElems);  // host output
    std::vector<float> h_block_sums((numElems + 2047) / 2048);  // for storing block-level sums

    float *d_idata, *d_odata, *d_block_sums;  // device arrays

    // Allocate device memory
    cudaMalloc((void**)&d_idata, numElems * sizeof(float));
    cudaMalloc((void**)&d_odata, numElems * sizeof(float));
    cudaMalloc((void**)&d_block_sums, h_block_sums.size() * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_idata, h_idata.data(), numElems * sizeof(float), cudaMemcpyHostToDevice);

    // Configure and launch kernel
    int blockSize = 1024;  // maximum half block size for max 2048 elements
    int gridSize = (numElems + 2 * blockSize - 1) / (2 * blockSize);  // each block handles 2 * blockSize elements
    int sharedMemSize = 2 * blockSize * sizeof(float);  // 2 * blockSize to handle 2048 elements

    gpu_sum_scan_blelloch<<<gridSize, blockSize, sharedMemSize>>>(d_odata, d_idata, d_block_sums, numElems);

    // Copy results back to host
    cudaMemcpy(h_odata.data(), d_odata, numElems * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_idata);
    cudaFree(d_odata);
    cudaFree(d_block_sums);

    return h_odata;
}





PYBIND11_MODULE(gpu_library, m) {
    //m.def("smooth_cuda", &smooth_cuda, "A function that smooths a list using CUDA");
    m.def("runSmoothListWithBlellochScan", &runSmoothListWithBlellochScan, "A function that smooths a list using Blelloch Sum Scan Algorithm");
    m.def("prescan", &runBlellochScan, "prescan");
}