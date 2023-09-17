cmake_minimum_required(VERSION 3.6)
project(MyProject)

# Set C++ standard to C++14
set(CMAKE_CXX_STANDARD 14)

# Find required packages
find_package(CUDA REQUIRED)
find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)

# Set CUDA flags to use C++14
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++14")

# Include directories
include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ./pybind11/include
    )

# Link directories
link_directories(
    /usr/local/cuda/lib64
    )

# Create the shared library
cuda_add_library(gpu_library SHARED
  gpu_library.cu)

# Link the library
target_link_libraries(gpu_library
  ${PYTHON_LIBRARIES}
  cudart)

# Set library properties
set_target_properties(gpu_library PROPERTIES PREFIX "")
