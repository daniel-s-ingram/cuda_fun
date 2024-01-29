#include <cuda_fun/cuda_utils.hpp>
#include <cuda_fun/GridInterface.hpp>
#include <cuda_fun/GridVisualizer.hpp>
#include <cuda_fun/Light.hpp>
#include <cuda_fun/Material.hpp>
#include <cuda_fun/Spheres.hpp>
#include <cuda_fun/Timer.hpp>
#include <cuda_fun/Vector.hpp>

#include <cmath>
#include <ctime>
#include <iostream>
#include <limits>
#include <vector>

namespace cuda_fun
{

__host__ __device__ Vec3f reflect(const Vec3f& I, const Vec3f& N)
{
    return I - N*2.0F*(I*N);
}

__host__ __device__ bool scene_intersect(const Vec3f& orig, const Vec3f& dir, const Spheres* const spheres, Vec3f& hit, Vec3f& N, Material& material) {
    float spheres_dist = std::numeric_limits<float>::max();
    for (std::uint32_t i = 0; i < NUM_SPHERES; ++i) {
        float dist_i{0.0F};
        if (spheres->ray_intersect(i, orig, dir, dist_i) && dist_i < spheres_dist) {
            spheres_dist = dist_i;
            hit = orig + dir*dist_i;
            N = (hit - spheres->center[i]).normalized();
            material = spheres->material[i];
        }
    }
    
    return spheres_dist<1000;
}

template<std::uint32_t depth = 0>
__host__ __device__ Vec3f cast_ray(const Vec3f &orig, const Vec3f &dir, const Spheres* const spheres, const Light* const lights, const std::uint32_t num_lights) 
{
    Vec3f point;
    Vec3f N;
    Material material;

    float sphere_dist = std::numeric_limits<float>::max();
    if (!scene_intersect(orig, dir, spheres, point, N, material))
    {
        return Vec3f{0.8F, 0.33F, 0.0F};
    }

    const Vec3f reflect_dir = reflect(dir, N).normalized();
    const Vec3f reflect_orig = reflect_dir*N < 0 ? (point - N*1e-3) : (point + N*1e-3);
    const Vec3f reflect_color = cast_ray<depth+1>(reflect_orig, reflect_dir, spheres, lights, num_lights);

    float diffuse_light_intensity{0.0F};
    float specular_light_intensity{0.0F};
    for (std::uint32_t i = 0; i < num_lights; ++i) {
        const Vec3f light_dir = (lights[i].position - point).normalized();
        const float light_distance = (lights[i].position - point).norm();

        const Vec3f shadow_orig = light_dir*N < 0.0F ? (point - N*1e-3F) : (point + N*1e-3F);
        Vec3f shadow_pt, shadow_N;
        Material tmpmaterial;
        if (scene_intersect(shadow_orig, light_dir, spheres, shadow_pt, shadow_N, tmpmaterial) && (shadow_pt - shadow_orig).norm() < light_distance)
            continue;

        diffuse_light_intensity += lights[i].intensity * std::max(0.f, light_dir*N);
        specular_light_intensity += std::pow(std::max(0.f, -reflect(-light_dir, N)*dir), material.specular_exponent)*lights[i].intensity;
    }
    
    return material.diffuse_color * diffuse_light_intensity * material.albedo[0] + (Vec3f{1.0F, 1.0F, 1.0F} * specular_light_intensity * material.albedo[1]) + (reflect_color * material.albedo[2]);
}

template<>
__host__ __device__ Vec3f cast_ray<4>(const Vec3f &orig, const Vec3f &dir, const Spheres* const spheres, const Light* const lights, const std::uint32_t num_lights) 
{
    return Vec3f{0.8, 0.33, 0.0};
}

__global__ void render(Vec3f* const current_grid, const Spheres* const spheres, const std::uint32_t N, const Light* const lights, const std::uint32_t num_lights)
{
    extern __shared__ Spheres shared_spheres;
    for (std::uint32_t i = 0; i < NUM_SPHERES; ++i)
    {
        shared_spheres.set(i, *spheres);
    }
    __syncthreads();

    constexpr float fov = 1.0F;

    const std::uint32_t i = blockIdx.y*blockDim.y + threadIdx.y;
    const std::uint32_t j = blockIdx.x*blockDim.x + threadIdx.x;

    const float x =  (2.0F*(i + 0.5F)/static_cast<float>(N) - 1.0F)*tanf(fov/2.0F)*N/static_cast<float>(N);
    const float y = -(2.0F*(j + 0.5F)/static_cast<float>(N) - 1.0F)*tanf(fov/2.0F);
    const Vec3f dir = Vec3f{x, y, -1.0F}.normalized();

    const Vec3f origin{0.0F, 0.0F, 50.0F};
    current_grid[i*N+j] = cast_ray(origin, dir, &shared_spheres, lights, num_lights);
}

__global__ void moveSpheres(Spheres* const spheres)
{
    const std::uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= NUM_SPHERES)
    {
        return;
    }

    Vec3f acceleration{0.0F, 0.0F, 0.0F};
    for (std::uint32_t j = 0; j < NUM_SPHERES; ++j)
    {
        if (i == j)
        {
            continue;
        }

        const Vec3f dist_vec = (spheres->center[i] - spheres->center[j]);
        const float r = std::max(dist_vec.norm(), 1.0F);
        acceleration += dist_vec.normalized() * (1e-1F*spheres->radius[j]/(r*r));
    }

    acceleration = -acceleration;

    __syncthreads();
    spheres->velocity[i] += acceleration;
    spheres->center[i] += spheres->velocity[i] + spheres->acceleration[i]*0.5F;
}

class TinyRayTracer : public GridInterface<Vec3f>
{
public:
    TinyRayTracer(const std::uint32_t rows, const std::uint32_t cols, Vec3f* const h_grid, const Spheres& spheres, const std::vector<Light>& lights) : 
        GridInterface<Vec3f>(rows, cols, h_grid)
    {
        m_spheres_size = sizeof(Spheres);

        m_num_lights = lights.size();
        m_lights_size = m_num_lights*sizeof(Light);

        cudaCheckError(cudaMalloc((void**)&m_d_spheres, m_spheres_size));
        cudaCheckError(cudaMalloc((void**)&m_d_lights, m_lights_size));
        
        cudaCheckError(cudaMemcpy(m_d_spheres, &spheres, m_spheres_size, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(m_d_lights, lights.data(), m_lights_size, cudaMemcpyHostToDevice));
    }

    void update() override
    {
        Timer timer{"tinyraytracer"};
        const dim3 block_dim{16, 16, 1};
        const dim3 grid_dim{m_rows / block_dim.x, m_cols / block_dim.y, 1};
        render<<<grid_dim, block_dim, m_spheres_size>>>(m_d_current_grid, m_d_spheres, m_rows, m_d_lights, m_num_lights);
        cudaCheckError(cudaPeekAtLastError());

        moveSpheres<<<1, NUM_SPHERES>>>(m_d_spheres);

        //std::swap(m_d_current_grid, m_d_next_grid);

        // todo: figure out how to draw directly from GPU memory to avoid this copy
        //std::cout << m_h_grid << std::endl;
        cudaCheckError(cudaMemcpy(m_h_grid, m_d_current_grid, m_size, cudaMemcpyDeviceToHost));
    }

    ~TinyRayTracer()
    {
        cudaCheckError(cudaFree(m_d_spheres));
        cudaCheckError(cudaFree(m_d_lights));
    }

private:
    Spheres* m_d_spheres;
    Light* m_d_lights;
    std::uint32_t m_spheres_size;
    std::uint32_t m_num_lights;
    std::uint32_t m_lights_size;
};

} // namespace cuda_fun

int main()
{
    using namespace cuda_fun;

    std::srand(std::time(nullptr));

    constexpr std::uint32_t rows{1024};
    constexpr std::uint32_t cols{rows};

    Vec3f* const h_grid{nullptr};
    cudaCheckError(cudaHostAlloc((void**)&h_grid, rows*cols*sizeof(Vec3f), cudaHostAllocDefault));

    // SoA performs ~88% better than AoS here
    Spheres spheres;
    for (std::uint32_t i = 0; i < NUM_SPHERES; ++i)
    {
        const Vec3f position = getRandomVec()*50.0F - 25.0F;
        const Vec3f velocity = getRandomVec()*0.05F- 0.025F;
        const Vec3f acceleration = getRandomVec()*0.05F - 0.025F;
        const Vec3f color = getRandomVec();
        const Vec3f albedo = getRandomVec();
        const float specular = 200.0*getRandomFloat();
        const float radius = 5.0*getRandomFloat();
        
        spheres.set(i, position, velocity, acceleration, radius, Material{color, albedo, specular});
    }

    std::vector<Light> lights;
    lights.emplace_back(Vec3f{-20, 20,  20}, 0.5);
    lights.emplace_back(Vec3f{ 30, 50, -25}, 0.8);
    lights.emplace_back(Vec3f{ 30, 20,  30}, 0.7);

    std::unique_ptr<GridInterface<Vec3f>> tiny_ray_tracer = std::make_unique<TinyRayTracer>(rows, cols, h_grid, spheres, lights);

    GridVisualizer grid_visualizer{rows, cols};
    grid_visualizer.run(std::move(tiny_ray_tracer));

    return 0;
}