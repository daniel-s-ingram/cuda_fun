#include <cuda_fun/Complex.hpp>
#include <cuda_fun/cuda_utils.hpp>
#include <cuda_fun/GridInterface.hpp>
#include <cuda_fun/GridVisualizer.hpp>
#include <cuda_fun/Vector.hpp>

#include <cuda.h>
#include <stdio.h>

#include <memory>

namespace cuda_fun
{

__global__ void mandelbrot(Vec3f* const d_grid, const std::size_t N, const float power)
{
    const std::size_t i = blockIdx.y*blockDim.y + threadIdx.y;
    const std::size_t j = blockIdx.x*blockDim.x + threadIdx.x;

    const float x = 3.0F*(j - 0.5F*N)/static_cast<float>(N);
    const float y = 3.0F*(i - 0.5F*N)/static_cast<float>(N);

    const Complex c{x, y};
    std::size_t k{0U};
    Complex z{0.0F, 0.0F};
    while ((z.getReal() < 4.0) && (k++ < 16))
    {
        z = powf(z, power) + c;
    }
    
Vec3f color_map[16] = {
    Vec3f{66, 30, 15}/255.0F,
    Vec3f{25, 7, 26}/255.0F,
    Vec3f{9, 1, 47}/255.0F,
    Vec3f{4, 4, 73}/255.0F,
    Vec3f{0, 7, 100}/255.0F,
    Vec3f{12, 44, 138}/255.0F,
    Vec3f{24, 82, 177}/255.0F,
    Vec3f{57, 125, 209}/255.0F,
    Vec3f{134, 181, 229}/255.0F,
    Vec3f{211, 236, 248}/255.0F,
    Vec3f{241, 233, 191}/255.0F,
    Vec3f{248, 201, 95}/255.0F,
    Vec3f{255, 170, 0}/255.0F,
    Vec3f{204, 128, 0}/255.0F,
    Vec3f{153, 87, 0}/255.0F,
    Vec3f{106, 52, 3}/255.0F
};



    d_grid[i*N + j] = color_map[k];
}

class Mandelbrot : public GridInterface<Vec3f>
{
public:
    Mandelbrot(const std::size_t rows, const std::size_t cols, Vec3f* const h_grid) : 
        GridInterface<Vec3f>(rows, cols, h_grid),
        m_power{0.0F}
    {}

    void update() override
    {
        const dim3 block_dim{16, 16, 1};
        const dim3 grid_dim{m_rows / block_dim.x, m_cols / block_dim.y, 1};
        mandelbrot<<<grid_dim, block_dim>>>(m_d_current_grid, m_rows, m_power);
        cudaCheckError(cudaPeekAtLastError());

        //std::swap(m_d_current_grid, m_d_next_grid);

        // todo: figure out how to draw directly from GPU memory to avoid this copy
        //std::cout << m_h_grid << std::endl;
        cudaCheckError(cudaMemcpy(m_h_grid, m_d_current_grid, m_size, cudaMemcpyDeviceToHost));

        m_power += 1e-2;
    }

private:
    float m_power{0.0F};
};
}

int main()
{
    using namespace cuda_fun;
    
    constexpr std::size_t rows{1024};
    constexpr std::size_t cols{rows};

    Vec3f* const h_grid = new Vec3f[rows*cols];

    GridVisualizer grid_visualizer{rows, cols};
    std::unique_ptr<GridInterface<Vec3f>> mandelbrot = std::make_unique<Mandelbrot>(rows, cols, h_grid);

    grid_visualizer.run(std::move(mandelbrot));

    return 0; 
}
