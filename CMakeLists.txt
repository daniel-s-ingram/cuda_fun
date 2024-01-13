cmake_minimum_required(VERSION 3.5.1)

project(CudaFun)

include(FindCUDA)
find_package(CUDA REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CUDA_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED true)
#set(CMAKE_CUDA_STANDARD_REQUIRED true)
set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")
set(CUDA_NVCC_FLAGS "-G --expt-relaxed-constexpr ${CUDA_NVCC_FLAGS}")

include_directories(include)

cuda_add_library(cuda_fun 
    src/GridInterface.cu
    src/GridVisualizer.cu
    include/cuda_fun/GridInterface.hpp
    include/cuda_fun/GridVisualizer.hpp
    include/cuda_fun/Sphere.hpp
    include/cuda_fun/Vector.hpp
)

target_include_directories(cuda_fun PUBLIC include)
target_include_directories(cuda_fun PUBLIC include)
target_link_libraries(cuda_fun glfw GL)

cuda_add_executable(game_of_life examples/game_of_life.cu)
target_include_directories(game_of_life PUBLIC include)
target_link_libraries(game_of_life cuda_fun)

cuda_add_executable(tinyraytracer examples/tinyraytracer.cu)
target_include_directories(tinyraytracer PUBLIC include)
target_link_libraries(tinyraytracer cuda_fun)