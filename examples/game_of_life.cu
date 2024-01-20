#include <cuda_fun/cuda_utils.hpp>
#include <cuda_fun/GridInterface.hpp>
#include <cuda_fun/GridVisualizer.hpp>

#include <cuda.h>
#include <stdio.h>

#include <cstdint>
#include <ctime>
#include <iostream>
#include <memory>
#include <random>

namespace cuda_fun
{

// todo: lots of room for speedup here between tiling and reducing the number of branches
__global__ void doGpuGol(const std::uint8_t* const current_grid, std::uint8_t* const next_grid, const int N)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int i = by*blockDim.y + ty;
    const int j = bx*blockDim.x + tx;

    const std::uint32_t up    = (i > 0) ? (i - 1) : (N - 1);
    const std::uint32_t down  = (i < (N - 1)) ? (i + 1) : 0;
    const std::uint32_t left  = (j > 0) ? (j - 1) : (N - 1);
    const std::uint32_t right = (j < (N - 1)) ? (j + 1) : 0;

    int num_live_neighbors{0};
    num_live_neighbors += (!current_grid[up*N + left]) ? 0 : 1;
    num_live_neighbors += (!current_grid[up*N + j]) ? 0 : 1;
    num_live_neighbors += (!current_grid[up*N + right]) ? 0 : 1;
    num_live_neighbors += (!current_grid[i*N + right]) ? 0 : 1;
    num_live_neighbors += (!current_grid[down*N + right]) ? 0 : 1;
    num_live_neighbors += (!current_grid[down*N + j]) ? 0 : 1;
    num_live_neighbors += (!current_grid[down*N + left]) ? 0 : 1;
    num_live_neighbors += (!current_grid[i*N + left]) ? 0 : 1;

    if (current_grid[i*N + j])
    {
        //printf("Alive! %d\n", num_live_neighbors);
        if (num_live_neighbors < 2U || num_live_neighbors > 3U)
        {
            next_grid[i*N + j] = 0;
        }
        else
        {
            next_grid[i*N + j] = current_grid[i*N + j];
        }
        
    }
    else
    {
        //printf("Dead! %d\n", num_live_neighbors);
        if (num_live_neighbors == 3U)
        {
            next_grid[i*N + j] = 255;
        }
        else
        {
            next_grid[i*N + j] = current_grid[i*N + j];
        }
    }
}

class GameOfLife : public GridInterface<std::uint8_t>
{
public:
    GameOfLife(const std::uint32_t rows, const std::uint32_t cols, std::uint8_t* const h_grid = nullptr) : 
        GridInterface<std::uint8_t>(rows, cols, h_grid)
    {}

    void update() override
    {
        cudaDeviceSynchronize();

        const dim3 block_dim{32, 32, 1};
        const dim3 grid_dim{m_rows / block_dim.x, m_cols / block_dim.y, 1};
        doGpuGol<<<grid_dim, block_dim>>>(m_d_current_grid, m_d_next_grid, m_rows);
        cudaCheckError(cudaPeekAtLastError());

        cudaDeviceSynchronize();

        std::swap(m_d_current_grid, m_d_next_grid);

        // todo: figure out how to draw directly from GPU memory to avoid this copy
        cudaCheckError(cudaMemcpy(m_h_grid, m_d_current_grid, m_size, cudaMemcpyDeviceToHost));
    }
};

}

void populateGrid(std::uint8_t* const grid, const int N)
{
    std::srand(std::time(nullptr));
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            grid[i*N + j] = (0.4 > std::rand()/static_cast<double>(RAND_MAX)) ? 255 : 0;
        }
    }
}

int main()
{
    using namespace cuda_fun;
    constexpr std::uint32_t rows{1024};
    constexpr std::uint32_t cols{rows};

    std::uint8_t* const h_grid = new std::uint8_t[rows*cols];
    populateGrid(h_grid, rows);

    std::unique_ptr<GridInterface<std::uint8_t>> game_of_life = std::make_unique<GameOfLife>(rows, cols, h_grid);

    GridVisualizer grid_visualizer{rows, cols};
    grid_visualizer.run(std::move(game_of_life));

    return 0;
}