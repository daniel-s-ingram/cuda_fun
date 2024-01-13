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
            grid[i*N + j] = Vec3f(j/float(N), i/float(N), 0);
        }
    }
}

struct Sphere
{
    Vec3f center;
    float radius;

    Sphere(const Vec3f& c, const float r) : 
        center{c},
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

__host__ __device__ Vec3f cast_ray(const Vec3f &orig, const Vec3f &dir, const Sphere &sphere) 
{
    float sphere_dist = std::numeric_limits<float>::max();
    if (!sphere.ray_intersect(orig, dir, sphere_dist))
    {
        return Vec3f(0.2, 0.7, 0.8);
    }
    return Vec3f(0.0, 0.0, 0.0);
}

__global__ void render(Vec3f* const current_grid, const Sphere* const spheres, const int N, const int num_spheres)
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
    const Vec3f dir = Vec3f(x, y, -1).normalized();

    //for (int k = 0; k < num_spheres; ++k)
    //{
    current_grid[i*N+j] = cast_ray(Vec3f(0,0,0), dir, spheres[0]);
    //}
}

__global__ void moveSpheres(Sphere* const spheres, const int num_spheres)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= num_spheres)
    {
        return;
    }

    spheres[i].center += Vec3f{0.001, 0.001, 0.0};
}

class TinyRayTracer : public GridInterface<Vec3f>
{
public:
    TinyRayTracer(const std::size_t rows, const std::size_t cols, Vec3f* const h_grid, const std::vector<Sphere>& spheres) : 
        GridInterface<Vec3f>(rows, cols, h_grid)
    {
        m_num_spheres = spheres.size();
        m_spheres_size = m_num_spheres*sizeof(Sphere);
        cudaCheckError(cudaMalloc((void**)&m_d_spheres, m_spheres_size));
        cudaCheckError(cudaMemcpy(m_d_spheres, spheres.data(), m_spheres_size, cudaMemcpyHostToDevice));
    }

    void update() override
    {
        cudaDeviceSynchronize();

        const dim3 block_dim{32, 32, 1};
        const dim3 grid_dim{m_rows / block_dim.x, m_cols / block_dim.y, 1};
        render<<<grid_dim, block_dim>>>(m_d_current_grid, m_d_spheres, m_rows, m_spheres_size);
        cudaCheckError(cudaPeekAtLastError());

        cudaDeviceSynchronize();

        moveSpheres<<<1, m_num_spheres>>>(m_d_spheres, m_num_spheres);

        //std::swap(m_d_current_grid, m_d_next_grid);

        // todo: figure out how to draw directly from GPU memory to avoid this copy
        cudaCheckError(cudaMemcpy(m_h_grid, m_d_current_grid, m_size, cudaMemcpyDeviceToHost));
    }

private:
    Sphere* m_d_spheres;
    std::size_t m_num_spheres;
    std::size_t m_spheres_size;
};

}

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

int main()
{
    using namespace cuda_fun;

    constexpr std::size_t rows{1024};
    constexpr std::size_t cols{rows};

    Vec3f* const h_grid = new Vec3f[rows*cols];
    populateGrid(h_grid, rows);

    std::vector<Sphere> spheres;
    spheres.emplace_back(Vec3f{0, 0, -10}, 1);

    GridVisualizer grid_visualizer{rows, cols};
    std::unique_ptr<GridInterface<Vec3f>> tiny_ray_tracer = make_unique<TinyRayTracer>(rows, cols, h_grid, spheres);

    grid_visualizer.run(std::move(tiny_ray_tracer));

    return 0;
}