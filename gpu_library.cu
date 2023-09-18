#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

__device__ double calculate_smooth_value(const double* prefix_sum, int i, int h, int N) {
    double left_sum = (i - h - 1 >= 0) ? prefix_sum[i - h - 1] : 0.0f;
    double right_sum = prefix_sum[min(i + h, N - 1)];

    double running_sum = right_sum - left_sum;
    int num_elements = min(i + h, N - 1) - max(0, i - h) + 1;

    return running_sum / num_elements;
}

struct calculate_smooth {
    const int h;
    const double* prefix_sum;
    const int N;
    calculate_smooth(int _h, const double* _prefix_sum, int _N) : h(_h), prefix_sum(_prefix_sum), N(_N) {}

    __device__ double operator()(int i) const {
        return calculate_smooth_value(prefix_sum, i, h, N);
    }
};

const int BATCH_SIZE = 1024 * 1024;  // 可根据实际硬件资源进行调整

std::vector<double> runSmoothListWithBlellochScan(const std::vector<double>& h_inlist, int h) {
    const int N = h_inlist.size();
    const int num_batches = (N + BATCH_SIZE - 1) / BATCH_SIZE;

    thrust::device_vector<double> d_inlist(h_inlist);  // 一开始就将整个列表传到设备上
    thrust::device_vector<double> d_prefix_sum(N);
    thrust::device_vector<double> d_outlist(N);

    // 计算整个列表的前缀和
    thrust::inclusive_scan(d_inlist.begin(), d_inlist.end(), d_prefix_sum.begin());

    // 执行批处理
    for (int b = 0; b < num_batches; ++b) {
        int start_idx = b * BATCH_SIZE;
        int end_idx = std::min((b + 1) * BATCH_SIZE, N);

        // 创建 calculate_smooth 对象
        calculate_smooth smooth_calculator(h, thrust::raw_pointer_cast(&d_prefix_sum[0]), N);

        // 执行平滑计算
        thrust::transform(
            thrust::counting_iterator<int>(start_idx),
            thrust::counting_iterator<int>(end_idx),
            d_outlist.begin() + start_idx,
            smooth_calculator  // 传递 calculate_smooth 对象
        );
    }

    std::vector<double> h_outlist(N);
    thrust::copy(d_outlist.begin(), d_outlist.end(), h_outlist.begin());

    return h_outlist;
}

PYBIND11_MODULE(gpu_library, m) {
    m.def("runSmoothListWithBlellochScan", &runSmoothListWithBlellochScan, "A function that smooths a list using Blelloch Sum Scan Algorithm");
}