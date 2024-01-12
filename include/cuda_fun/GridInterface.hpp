#ifndef _CUDA_FUN_GRID_INTERFACE_HPP_
#define _CUDA_FUN_GRID_INTERFACE_HPP_

#include <cstdint>

namespace cuda_fun
{

template<typename GridCellType>
class GridInterface
{
public:
    GridInterface(const std::size_t rows, const std::size_t cols, GridCellType* const h_grid = nullptr);

    virtual void update() = 0;

    virtual ~GridInterface();

    GridCellType* getHostPtr() { return m_h_grid; }

protected:
    std::size_t m_rows{0U};
    std::size_t m_cols{0U};
    std::size_t m_size{0U};

    GridCellType* m_h_grid{nullptr};
    GridCellType* m_d_current_grid{nullptr};
    GridCellType* m_d_next_grid{nullptr};
};

}

#endif