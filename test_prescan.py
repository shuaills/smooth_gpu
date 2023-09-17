import gpu_library
import numpy as np
import time


# Importing the shared library built with Pybind11 and CUDA
# Note: Please replace the 'gpu_library' with the actual name of your shared library
# import gpu_library

# Sample input array and 'h' value for testing
input_array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
h_value = 3
test_list = list(range(100))  # 或者其他随机列表

# Call the 'prescan' function from the shared library
# Assuming the function takes an array of floats and an integer 'h' as parameters and returns an array of floats
out_array = gpu_library.prescan(input_array)

# Due to the environment restrictions, I can't run the actual shared library here
# But you can uncomment the import and the function call lines in your local environment for testing

# Printing the result for verification
print("Result of prescan:", out_array)

# Here is a placeholder for the expected output for manual verification
# Note: The expected output is based on a standard scan algorithm, you might need to adjust it based on your specific implementation
expected_output = [0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0]
print("Expected output for manual verification:", expected_output)

out_array = gpu_library.prescan(test_list)  # 或者其他随机列表
print("Result of prescan:", out_array)
