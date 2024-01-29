#include <cuda_fun/cuda_utils.hpp>
#include <cuda_fun/Vector.hpp>

#include <cuda.h>
#include <GLFW/glfw3.h>

#include <array>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>

///@todo generalize the GridInterface/GridVisualizer classes to support particles instead of just grids, will support fluid and other sims too
namespace cuda_fun
{

template<std::size_t NumParticles>
struct Particle
{
    Vec2f pos[NumParticles];
    Vec2f vel[NumParticles];
    Vec2f acc[NumParticles];
};

template<std::size_t NumParticles>
__global__ void computeAcceleration(Particle<NumParticles>* const d_particles, const std::uint32_t num_particles, const float dt)
{
    const std::uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= num_particles)
    {
        return;
    }

    // extern __shared__ Particle shared_particles[];
    // shared_particles[i] = d_particles[i];
    // __syncthreads();

    Vec2f acc{0.0F, 0.0F};
    for (std::uint32_t j = 0U; j < num_particles; ++j)
    {
        if (i == j)
        {
            continue;
        }

        const Vec2f d = d_particles->pos[j] - d_particles->pos[i];
        const float r = d.norm() + 1.0F;
        const float a = 1.0F/(r*r);

        acc += a*d.normalized();
    }

    d_particles->acc[i] = acc;
}

template<std::size_t NumParticles>
__global__ void updateParticles(Particle<NumParticles>* const d_particles, const std::uint32_t num_particles, const float dt)
{
    const std::uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= num_particles)
    {
        return;
    }

    d_particles->vel[i] += d_particles->acc[i]*dt;
    d_particles->pos[i] += d_particles->vel[i]*dt + 0.5F*d_particles->acc[i]*dt*dt;
} 

} // namespace cuda_fun

int main()
{
    using namespace cuda_fun;

    std::srand(std::time(nullptr));
    constexpr std::uint32_t rows{1024U};
    constexpr std::uint32_t cols{rows};
    constexpr std::uint32_t num_particles{10000U};
    constexpr float dt{1e-2};

    if (!glfwInit())
    {
        std::cout << "error initializing glfw" << std::endl;
    }

    GLFWwindow* window = glfwCreateWindow(rows, cols, "N-Body", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    Particle<num_particles> particles;
    for (std::size_t i = 0U; i < num_particles; ++i)
    {
        particles.pos[i][0] = 100.0*(-0.5 + std::rand()/static_cast<float>(RAND_MAX));
        particles.pos[i][1] = 100.0*(-0.5 + std::rand()/static_cast<float>(RAND_MAX));
        particles.vel[i][0] = 0.0F;
        particles.vel[i][1] = 0.0F;
        particles.acc[i][0] = 0.0F;
        particles.acc[i][1] = 0.0F;
    }

    glOrtho(-100, 100, -100, 100, 0, 1);
    glColor4f(1.0F, 1.0F, 0.0F, 1.0F);
    glPointSize(4.0F);

    Particle<num_particles>* d_particles{nullptr};
    cudaCheckError(cudaMalloc((void**)&d_particles, sizeof(Particle<num_particles>)));
    cudaCheckError(cudaMemcpy(d_particles, &particles, sizeof(Particle<num_particles>), cudaMemcpyHostToDevice));

    std::cout << sizeof(Particle<num_particles>) << std::endl;

    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT);

        glBegin(GL_POINTS);
        for (std::size_t i = 0U; i < num_particles; ++i)
        {
            glVertex2f(particles.pos[i][0], particles.pos[i][1]);
        }
        glEnd();

        glfwSwapBuffers(window);

        glfwPollEvents();

        constexpr std::uint32_t num_threads{128U};
        constexpr std::uint32_t num_blocks = std::ceil(num_particles/static_cast<float>(num_threads));
        computeAcceleration<<<num_blocks, num_threads>>>(d_particles, num_particles, dt);
        cudaCheckError(cudaPeekAtLastError());
        updateParticles<<<num_blocks, num_threads>>>(d_particles, num_particles, dt);
        cudaCheckError(cudaPeekAtLastError());
        cudaCheckError(cudaMemcpy(&particles, d_particles, sizeof(Particle<num_particles>), cudaMemcpyDeviceToHost));
    }  

    cudaFree(d_particles);
    glfwTerminate();
}