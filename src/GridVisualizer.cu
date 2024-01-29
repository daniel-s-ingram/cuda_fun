#include <cuda_fun/GridInterface.hpp>
#include <cuda_fun/GridVisualizer.hpp>
#include <cuda_fun/Vector.hpp>

#include <GLFW/glfw3.h>

#include <iostream>

namespace cuda_fun
{

GridVisualizer::GridVisualizer(const std::uint32_t rows, const std::uint32_t cols) :
    m_window{nullptr},
    m_rows{rows},
    m_cols{cols}
{
    if (!glfwInit())
    {
        std::cout << "error initializing glfw" << std::endl;
    }

    m_window = glfwCreateWindow(m_rows, m_cols, "Conway's Game of Life", nullptr, nullptr);
    glfwMakeContextCurrent(m_window);
}

template<typename T>
void GridVisualizer::run(std::unique_ptr<GridInterface<T>> grid)
{
    // todo: figure out how to do things the "modern" OpenGL way (this works for now though)
    while (!glfwWindowShouldClose(m_window))
    {       
        glClear(GL_COLOR_BUFFER_BIT);

        // todo: figure out how to draw directly from GPU memory
        const auto* const h_grid = grid->getHostPtr();
        const auto format = sizeof(T) == 12 ? GL_RGB : GL_GREEN;
        glDrawPixels(m_rows, m_cols, format, GL_FLOAT, h_grid);

        glfwSwapBuffers(m_window);

        glfwPollEvents();

        grid->update();
    }  
}

GridVisualizer::~GridVisualizer()
{
    glfwTerminate();
}

template void GridVisualizer::run<std::uint8_t>(std::unique_ptr<GridInterface<std::uint8_t>>);
template void GridVisualizer::run<float>(std::unique_ptr<GridInterface<float>>);
template void GridVisualizer::run<Vec3f>(std::unique_ptr<GridInterface<Vec3f>>);

}