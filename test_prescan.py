import gpu_library
import numpy as np
import time
from typing import List


def smooth(inlist: List[float], h: int) -> List[float]:
    """
    Smooths a list using a window size of 2*h+1.
    Time Complexity: O(N), where N is the length of inlist.

    Parameters:
    - inlist: Input list of floats.
    - h: Half-size of the smoothing window.
    Returns:
    - A list of smoothed floats of the same length as inlist.
    """
    N = len(inlist)
    if N == 0:
        return []
    
    outlist = [0.0] * N
    running_sum = sum(inlist[:min(h + 1, N)])  
    
    # Initialize the first smoothed value
    outlist[0] = running_sum / min(h + 1, N)  
    
    for i in range(1, N):
        next_boundary = i + h
        prev_boundary = i - h - 1

        # Update the running sum based on the window boundaries
        if next_boundary < N:
            running_sum += inlist[next_boundary]
        if prev_boundary >= 0:
            running_sum -= inlist[prev_boundary]
        
        window_size = min(i + h, N - 1) - max(0, i - h) + 1
        outlist[i] = running_sum / window_size
    
    return outlist

def test_smooth_time(inlist, h):
    t0 = time.time()
    smoothed_cuda = gpu_library.runSmoothListWithBlellochScan(np.array(inlist), h)
    print(len(smoothed_cuda))
    cuda_time = time.time() - t0
    
    t0 = time.time()
    smoothed_python = smooth(inlist, h)
    print(len(smoothed_python))
    python_time = time.time() - t0

    print(f"Time taken by Python prefix method: {python_time} seconds")
    print(f"Time taken by Cuda method: {cuda_time} seconds")

    print(f"First 10 values by Python prefix method: {smoothed_python[:10]} ")
    print(f"First 10 values CUDA method: {smoothed_cuda[:10]} ")

    print(f"Last 10 values by Python prefix method: {smoothed_python[-10:]} ")
    print(f"Last 10 values CUDA method: {smoothed_cuda[-10:]} ")

    start_diff_index = -1
    for i in range(len(smoothed_python)):
        if not np.isclose(smoothed_python[i], smoothed_cuda[i], atol=1e-6):
            start_diff_index = i
            break

    return start_diff_index

if __name__ == "__main__":
    N = 1000000000 
    h = 1000  
    test_list = list(range(N)) 

    index = test_smooth_time(test_list, h)
    print(index)
