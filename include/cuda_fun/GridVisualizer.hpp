#ifndef _CUDA_FUN_GRID_VISUALIZER_HPP_
#define _CUDA_FUN_GRID_VISUALIZER_HPP_

#include <cstdint>
#include <memory>

class GLFWwindow;

namespace cuda_fun
{

template<typename T>
class GridInterface;

class GridVisualizer
{
public:
    GridVisualizer(const std::uint32_t rows, const std::uint32_t cols);

    template<typename T>
    void run(std::unique_ptr<GridInterface<T>> grid);

    ~GridVisualizer();

private:
    GLFWwindow* m_window{nullptr}; //hmmm
    std::uint32_t m_rows{0U};
    std::uint32_t m_cols{0U};
};

}

#endif