from typing import List

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

inlist = [random.random() for _ in range(10**8)]
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