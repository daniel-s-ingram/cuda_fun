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

struct Particle
{
    Vec2f pos{0.0F, 0.0F};
    Vec2f vel{0.0F, 0.0F};
    Vec2f acc{0.0F, 0.0F};
};

__global__ void computeAcceleration(Particle* const d_particles, const std::uint32_t num_particles, const float dt)
{
    const std::uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= num_particles)
    {
        return;
    }

    auto& particle = d_particles[i];

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

        const Vec2f d = d_particles[j].pos - particle.pos;
        const float r = d.norm() + 1.0F;
        const float a = 1.0F/(r*r);

        acc += a*d.normalized();
    }

    particle.acc = acc;
}

__global__ void updateParticles(Particle* const d_particles, const std::uint32_t num_particles, const float dt)
{
    const std::uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= num_particles)
    {
        return;
    }

    auto& particle = d_particles[i];

    particle.vel += particle.acc*dt;
    particle.pos += particle.vel*dt + 0.5F*particle.acc*dt*dt;
} 

} // namespace cuda_fun

int main()
{
    using namespace cuda_fun;

    std::srand(std::time(nullptr));
    constexpr std::uint32_t rows{1024U};
    constexpr std::uint32_t cols{rows};
    constexpr std::uint32_t num_particles{5000U};
    constexpr float dt{1e-2};

    if (!glfwInit())
    {
        std::cout << "error initializing glfw" << std::endl;
    }

    GLFWwindow* window = glfwCreateWindow(rows, cols, "N-Body", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    std::array<Particle, num_particles> particles;
    for (auto& particle : particles)
    {
        particle.pos[0] = 100.0*(-0.5 + std::rand()/static_cast<float>(RAND_MAX));
        particle.pos[1] = 100.0*(-0.5 + std::rand()/static_cast<float>(RAND_MAX));
    }

    glOrtho(-100, 100, -100, 100, 0, 1);
    glColor4f(1.0F, 1.0F, 0.0F, 1.0F);
    glPointSize(4.0F);

    Particle* d_particles{nullptr};
    cudaCheckError(cudaMalloc((void**)&d_particles, num_particles*sizeof(Particle)));
    cudaCheckError(cudaMemcpy(d_particles, particles.data(), num_particles*sizeof(Particle), cudaMemcpyHostToDevice));

    std::cout << num_particles*sizeof(Particle) << std::endl;

    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT);

        glBegin(GL_POINTS);
        for (const auto& particle : particles)
        {
            glVertex2f(particle.pos[0], particle.pos[1]);
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
        cudaCheckError(cudaMemcpy(particles.data(), d_particles, num_particles*sizeof(Particle), cudaMemcpyDeviceToHost));
    }  

    cudaFree(d_particles);
    glfwTerminate();
}