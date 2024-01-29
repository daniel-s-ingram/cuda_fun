#ifndef _CUDA_FUN_CUDA_UTILS_HPP_
#define _CUDA_FUN_CUDA_UTILS_HPP_

#include <cstdio>
#include <cuda.h>

#define cudaCheckError(code) { cudaAssert(code, __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code == cudaSuccess) 
    {
        return;
    }

    printf("%s in file %s on line %d\n\n", cudaGetErrorString(code), file, line);
    exit(1);
}

#endif