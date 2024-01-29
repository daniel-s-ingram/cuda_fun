cmake_minimum_required(VERSION 3.5.1)

project(CudaFun)

include(FindCUDA)
find_package(CUDA REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++20 ${CMAKE_CXX_FLAGS}")
set(CUDA_NVCC_FLAGS "-std=c++20 --expt-relaxed-constexpr ${CUDA_NVCC_FLAGS}")

cuda_add_library(cuda_fun 
    src/GridInterface.cu
    src/GridVisualizer.cu
)
target_include_directories(cuda_fun PUBLIC include)
target_link_libraries(cuda_fun glfw GL)

cuda_add_executable(game_of_life examples/game_of_life.cu)
target_include_directories(game_of_life PUBLIC include)
target_link_libraries(game_of_life cuda_fun)

cuda_add_executable(tinyraytracer examples/tinyraytracer.cu)
target_include_directories(tinyraytracer PUBLIC include)
target_link_libraries(tinyraytracer cuda_fun)

cuda_add_executable(fractals examples/fractals.cu)
target_include_directories(fractals PUBLIC include)
target_link_libraries(fractals cuda_fun)

cuda_add_executable(nbody examples/nbody.cu)
target_include_directories(nbody PUBLIC include)
target_link_libraries(nbody cuda_fun)