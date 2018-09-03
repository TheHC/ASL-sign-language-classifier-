add_executable(UFF_to_ENGINE UFF_to_ENGINE.cpp)
target_link_libraries(UFF_to_ENGINE nvinfer nvparsers)
cuda_add_executable(Inference Inference.cu)
target_link_libraries(Inference ${OpenCV_LIBS} nvinfer nvparsers)
