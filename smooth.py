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

def smooth_prefix(inlist: List[float], h: int) -> List[float]:

    N = len(inlist)
    if N == 0:
        return []
    
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i+1] = prefix_sum[i] + inlist[i]

    outlist = [0.0] * N
    for i in range(N):
        left = max(0, i-h)
        right = min(N, i+h+1)
        window_sum = prefix_sum[right] - prefix_sum[left]
        num_elements = right - left
        outlist[i] = window_sum / num_elements

    return outlist

import random
import time

inlist = [random.random() for _ in range(10**6)]
h = 1000

def test(func):
    start = time.time()
    outlist = func(inlist, h)
    end = time.time()
    print(end - start)

print("Testing smooth...")
test(smooth) 

print("Testing smooth_prefix...")
test(smooth_prefix)