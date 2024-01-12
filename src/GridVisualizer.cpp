#include <cuda_fun/GridInterface.hpp>
#include <cuda_fun/GridVisualizer.hpp>
#include <GLFW/glfw3.h>

#include <iostream>

namespace cuda_fun
{

GridVisualizer::GridVisualizer(const std::size_t rows, const std::size_t cols) :
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
    while (!glfwWindowShouldClose(m_window))
    {       
        glClear(GL_COLOR_BUFFER_BIT);

        const auto* const h_grid = grid->getHostPtr();
        glDrawPixels(m_rows, m_cols, GL_GREEN, GL_UNSIGNED_BYTE, h_grid);

        /* Swap front and back buffers */
        glfwSwapBuffers(m_window);

        /* Poll for and process events */
        glfwPollEvents();

        grid->update();
    }  
}

GridVisualizer::~GridVisualizer()
{
    // need to free window
    glfwTerminate();
}

template void GridVisualizer::run<std::uint8_t>(std::unique_ptr<GridInterface<std::uint8_t>>);

}