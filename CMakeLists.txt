cmake_minimum_required(VERSION 3.1)
project(ASL-sign-language-classifier-)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})

set(CUDA_NVCC_FLAGS --std=c++11)
set(CMAKE_CXX_STANDARD 11)

add_subdirectory(TensorRT_Inference)

