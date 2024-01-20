#include <cuda_fun/GridInterface.hpp>
#include <cuda_fun/Vector.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#include <iostream>

#define cudaCheckError(code) { cudaAssert((code), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code == cudaSuccess) 
    {
        return;
    }

    printf("%s in file %s on line %d\n\n", cudaGetErrorString(code), file, line);
    exit(1);
}

namespace cuda_fun
{

template<typename GridCellType>
GridInterface<GridCellType>::GridInterface(const std::uint32_t rows, const std::uint32_t cols, GridCellType* const h_grid) :
    m_rows{rows},
    m_cols{cols},
    m_size{m_rows*m_cols*sizeof(GridCellType)},
    m_h_grid{nullptr},
    m_d_current_grid{nullptr},
    m_d_next_grid{nullptr}
{
    cudaCheckError(cudaMalloc((void**)&m_d_current_grid, m_size));
    cudaCheckError(cudaMalloc((void**)&m_d_next_grid, m_size));

    if (h_grid != nullptr)
    {
        m_h_grid = h_grid;
        cudaCheckError(cudaMemcpy(m_d_current_grid, m_h_grid, m_size, cudaMemcpyHostToDevice));
    }
}

template<typename GridCellType>
GridInterface<GridCellType>::~GridInterface()
{
    cudaCheckError(cudaFree(m_d_current_grid));
    cudaCheckError(cudaFree(m_d_next_grid));
}

template class GridInterface<std::uint8_t>;
template class GridInterface<float>;
template class GridInterface<Vec3f>;

}