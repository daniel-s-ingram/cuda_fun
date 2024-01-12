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
    GridVisualizer(const std::size_t rows, const std::size_t cols);

    template<typename T>
    void run(std::unique_ptr<GridInterface<T>> grid);

    ~GridVisualizer();

private:
    GLFWwindow* m_window{nullptr}; //hmmm
    std::size_t m_rows{0U};
    std::size_t m_cols{0U};
};

}

#endif