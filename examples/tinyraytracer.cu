#include <cuda_fun/GridInterface.hpp>
#include <cuda_fun/GridVisualizer.hpp>
#include <cuda_fun/Vector.hpp>
#include <cuda_fun/Sphere.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#define cudaCheckError(code) { cudaAssert((code), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code == cudaSuccess) 
    {
        return;
    }

    cudaDeviceReset();
    printf("%s in file %s on line %d\n\n", cudaGetErrorString(code), file, line);
    exit(1);
}

namespace cuda_fun
{

void populateGrid(Vec3f* const grid, const int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            grid[i*N + j] = Vec3f{j/float(N), i/float(N), 0};
        }
    }
}

struct Material
{
    __host__ __device__ Material(const Vec3f& color) : 
        diffuse_color{color}
    {}

    __host__ __device__ Material() : diffuse_color{}
    {}

    Vec3f diffuse_color;
};

struct Light
{
    Light(const Vec3f& p, const float i) :
        position{p},
        intensity{i}
    {}

    Vec3f position;
    float intensity;
};

struct Sphere
{
    Vec3f center;
    Material material;
    float radius;

    Sphere(const Vec3f& c, const Material& m, const float r) : 
        center{c},
        material{m},
        radius{r}
    {}

    __host__ __device__ bool ray_intersect(const Vec3f& orig, const Vec3f& dir, float& t0) const 
    {
        const Vec3f L = center - orig;
        const float tca = L*dir;
        const float d2 = L*L - tca*tca;
        if (d2 > radius*radius)
        {
            return false;
        }

        const float thc = std::sqrt(radius*radius - d2);
        const float t1 = tca + thc;

        t0 = tca - thc;
        if (t0 < 0)
        {
            t0 = t1;
        }

        if (t0 < 0)
        {
            return false;
        }

        return true;
    }
};

__host__ __device__ bool scene_intersect(const Vec3f& orig, const Vec3f& dir, const Sphere* const spheres, const int num_spheres, Vec3f& hit, Vec3f& N, Material& material) {
    float spheres_dist = std::numeric_limits<float>::max();
    for (int i = 0; i < num_spheres; ++i) {
        float dist_i;
        if (spheres[i].ray_intersect(orig, dir, dist_i) && dist_i < spheres_dist) {
            spheres_dist = dist_i;
            hit = orig + dir*dist_i;
            N = (hit - spheres[i].center).normalized();
            material = spheres[i].material;
        }
    }
    
    return spheres_dist<1000;
}

__host__ __device__ Vec3f cast_ray(const Vec3f &orig, const Vec3f &dir, const Sphere* const spheres, const int num_spheres, const Light* const lights, const int num_lights) 
{
    Vec3f point, N;
    Material material;

    float sphere_dist = std::numeric_limits<float>::max();
    if (!scene_intersect(orig, dir, spheres, num_spheres, point, N, material))
    {
        return Vec3f(0.2, 0.7, 0.8);
    }

    float diffuse_light_intensity = 1.0;
    for (std::size_t i = 0; i < num_lights; ++i) {
        // const Vec3f light_dir = (lights[i].position - point).normalized();
        // diffuse_light_intensity += lights[i].intensity * std::max(0.f, light_dir*N);
    }
    
    return material.diffuse_color * diffuse_light_intensity;
}

__global__ void render(Vec3f* const current_grid, const Sphere* const spheres, const int N, const int num_spheres, const Light* const lights, const int num_lights)
{
    constexpr int fov = M_PI/2.;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int i = by*blockDim.y + ty;
    const int j = bx*blockDim.x + tx;

    const float x =  (2*(i + 0.5)/(float)N - 1)*std::tan(fov/2.)*N/(float)N;
    const float y = -(2*(j + 0.5)/(float)N - 1)*std::tan(fov/2.);
    const Vec3f dir = Vec3f{x, y, -1}.normalized();

    const Vec3f origin{0, 0, 1};
    current_grid[i*N+j] = cast_ray(origin, dir, spheres, num_spheres, lights, num_spheres);
}

__global__ void moveSpheres(Sphere* const spheres, const int num_spheres)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= num_spheres)
    {
        return;
    }

    spheres[i].center += Vec3f{0.01, 0.01, 0.0};
}

class TinyRayTracer : public GridInterface<Vec3f>
{
public:
    TinyRayTracer(const std::size_t rows, const std::size_t cols, Vec3f* const h_grid, const std::vector<Sphere>& spheres, const std::vector<Light>& lights) : 
        GridInterface<Vec3f>(rows, cols, h_grid)
    {    
        m_num_spheres = spheres.size();
        m_spheres_size = m_num_spheres*sizeof(Sphere);

        m_num_lights = lights.size();
        m_lights_size = m_num_lights*sizeof(Light);

        cudaCheckError(cudaMalloc((void**)&m_d_spheres, m_spheres_size));
        cudaCheckError(cudaMalloc((void**)&m_d_lights, m_lights_size));
        
        cudaCheckError(cudaMemcpy(m_d_spheres, spheres.data(), m_spheres_size, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(m_d_lights, lights.data(), m_lights_size, cudaMemcpyHostToDevice));

        std::cout << "spheres: " << m_d_spheres << " " << m_d_spheres + m_spheres_size << std::endl;
        std::cout << "lights " << m_d_lights << " " << m_d_lights + m_lights_size << std::endl;
    }

    void update() override
    {
        const dim3 block_dim{32, 32, 1};
        const dim3 grid_dim{m_rows / block_dim.x, m_cols / block_dim.y, 1};
        render<<<grid_dim, block_dim>>>(m_d_current_grid, m_d_spheres, m_rows, m_num_spheres, m_d_lights, m_num_lights);
        cudaCheckError(cudaPeekAtLastError());

        moveSpheres<<<1, m_num_spheres>>>(m_d_spheres, m_num_spheres);

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
    Sphere* m_d_spheres;
    Light* m_d_lights;
    std::size_t m_num_spheres;
    std::size_t m_spheres_size;
    std::size_t m_num_lights;
    std::size_t m_lights_size;
};

}

int main()
{
    using namespace cuda_fun;

    constexpr std::size_t rows{1024};
    constexpr std::size_t cols{rows};

    Vec3f* const h_grid = new Vec3f[rows*cols];
    populateGrid(h_grid, rows);

    std::vector<Sphere> spheres;
    spheres.emplace_back(Vec3f{-3, -3, -10}, Material{Vec3f{0.5, 0, 0}}, 1);
    spheres.emplace_back(Vec3f{-2, -8, -20}, Material{Vec3f{0.2, 0.7, 0}}, 4);
    spheres.emplace_back(Vec3f{-5, -5, -13}, Material{Vec3f{0.2, 0.2, 0.4}}, 2);

    std::vector<Light> lights;
    //lights.emplace_back(Vec3f{-20, 20, 20}, 0.5);

    GridVisualizer grid_visualizer{rows, cols};
    std::unique_ptr<GridInterface<Vec3f>> tiny_ray_tracer = std::make_unique<TinyRayTracer>(rows, cols, h_grid, spheres, lights);

    grid_visualizer.run(std::move(tiny_ray_tracer));

    return 0;
}