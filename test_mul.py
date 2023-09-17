import gpu_library
import numpy as np
import time

def smooth_naive(inlist, h):
    """
    This function performs a basic smoothing of inlist using a naive approach and returns the result (outlist).
    Both lists have the same length, N. Each item in inlist is assumed to have type 'float',
    and 'h' is assumed to be an integer.
    For each i, outlist[i] will be the average of inlist[k] over all k that satisfy
    i-h <= k <= i+h and 0 <= k <= N-1.
    """
    N = len(inlist)
    if N == 0:
        return []
    
    outlist = [0.0] * N  # Initialize outlist with zeros
    
    for i in range(N):
        running_sum = 0.0
        num_elements = 0
        
        # Calculate the running sum for the window centered at i
        for k in range(i - h, i + h + 1):
            # Check if the index is within the list boundaries
            if 0 <= k < N:
                running_sum += inlist[k]
                num_elements += 1
        
        # Calculate outlist[i]
        outlist[i] = running_sum / num_elements
    
    return outlist

def smooth(inlist, h):
    """
    This function performs a basic smoothing of inlist and returns the result (outlist).
    Both lists have the same length, N. Each item in inlist is assumed to have type 'float',
    and 'h' is assumed to be an integer.
    For each i, outlist[i] will be the average of inlist[k] over all k that satisfy
    i-h <= k <= i+h and 0 <= k <= N-1.
    """
    N = len(inlist)
    if N == 0:
        return []
    
    outlist = [0.0] * N  # Initialize outlist with zeros
    
    # Initialize a running sum with the first "h" elements in inlist
    running_sum = sum(inlist[:min(h+1, N)])
    
    # Calculate outlist[0]
    num_elements = min(h + 1, N)
    outlist[0] = running_sum / num_elements
    
    # Calculate outlist[i] for 1 <= i < N
    for i in range(1, N):
        # Add the next element to the running sum if it is within the right boundary
        next_index = i + h
        if next_index < N:
            running_sum += inlist[next_index]
        
        # Remove the element that is now out of the window from the running sum if it is within the left boundary
        prev_index = i - h - 1
        if prev_index >= 0:
            running_sum -= inlist[prev_index]
        
        # Calculate the number of elements in the current window
        num_elements = min(i + h, N-1) - max(0, i - h) + 1
        
        # Calculate outlist[i]
        outlist[i] = running_sum / num_elements
    
    return outlist

import time
import numpy as np

# 假设这里有一个gpu_library模块，提供了smooth_cuda函数
# import gpu_library

def test_smooth_time(inlist, h):
    # 使用CUDA方法进行测试（请确保这个函数实际存在）
    t0 = time.time()
    print(f"lets go!")
    smoothed_cuda = gpu_library.runSmoothListWithBlellochScan(np.array(inlist), h)
    cuda_time = time.time() - t0
    
    # 使用基础Python方法进行测试
    t0 = time.time()
    smoothed_python = smooth(inlist, h)
    python_time = time.time() - t0

    # 使用前缀和方法进行测试
    #t0 = time.time()
    #smoothed_naive = smooth_naive(inlist, h)
    #prefix_naive = time.time() - t0


    # 打印结果
    print(f"Time taken by Python prefix method: {python_time} seconds")
    print(f"Time taken by Cuda method: {cuda_time} seconds")
    #print(f"Time taken by Python naive sum method: {prefix_naive} seconds")

    print(f"First 10 values by Python prefix method: {smoothed_python[:10]} ")
    #print(f"First 3 values Python naive sum method: {smoothed_naive[:10]} ")
    print(f"First 10 values CUDA method: {smoothed_cuda[:10]} ")

    print(f"Last 10 values by Python prefix method: {smoothed_python[-10:]} ")
    #print(f"First 3 values Python naive sum method: {smoothed_naive[:10]} ")
    print(f"Last 10 values CUDA method: {smoothed_cuda[-10:]} ")

    # 在这里，你还可以加入一些代码来检查三种方法得出的结果是否一致
    #assert np.allclose(smoothed_python, smoothed_naive, atol=1e-6)
    assert np.allclose(smoothed_python, smoothed_cuda, atol=1e-6)

if __name__ == "__main__":
    N = 100  # 可以设置为其他值
    h = 2  # 可以设置为其他值
    test_list = list(range(N))  # 或者其他随机列表

    test_smooth_time(test_list, h)
