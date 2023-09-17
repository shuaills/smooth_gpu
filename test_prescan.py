import gpu_library
import numpy as np
import time
import numpy as np

def find_first_mismatch_index(arr1, arr2, tol=1e-6):
    """
    Find the index of the first mismatch between two arrays within a given tolerance.

    Parameters:
        arr1, arr2 (array-like): Arrays to compare
        tol (float): Tolerance for considering two elements to be equal

    Returns:
        int: Index of the first mismatch; -1 if no mismatch within the given tolerance
    """
    for i, (a, b) in enumerate(zip(arr1, arr2)):
        if not np.isclose(a, b, atol=tol):
            return i
    return -1


def test_prescan(input_array, h_value, should_print=True):
    try:
        # Timing the GPU operation
        start_time = time.time()
        gpu_output = gpu_library.prescan(input_array)
        gpu_time = time.time() - start_time
        
        # Timing the CPU operation
        start_time = time.time()
        cpu_output = np.zeros_like(input_array, dtype=np.float32)
        for i in range(1, len(input_array)):
            cpu_output[i] = cpu_output[i-1] + input_array[i-1]
        cpu_time = time.time() - start_time

        # Find the index of the first mismatch
        mismatch_index = find_first_mismatch_index(gpu_output, cpu_output)
        
        if should_print:
            if mismatch_index == -1:
                print(f"All values match. GPU time: {gpu_time:.6f} seconds, CPU time: {cpu_time:.6f} seconds")
            else:
                print(f"First mismatch at index {mismatch_index}. GPU: {gpu_output[mismatch_index]}, CPU: {cpu_output[mismatch_index]}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Medium-scale test
N = 10**4
h_value = 5
input_array = np.arange(N, dtype=np.float32)
test_prescan(input_array, h_value, should_print=True)

# Large-scale test
N = 10**6
h_value = 5
input_array = np.arange(N, dtype=np.float32)
test_prescan(input_array, h_value, should_print=True)