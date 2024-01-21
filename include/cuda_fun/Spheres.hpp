#ifndef _CUDA_FUN_SPHERES_HPP_
#define _CUDA_FUN_SPHERES_HPP_

#include <cuda_fun/Material.hpp>
#include <cuda_fun/Vector.hpp>

namespace cuda_fun
{

/// @todo make this a template parameter for Spheres as soon as I figure out how to get around the 
///       lack of partial template function specialization support since I need the specialization
///       to support the recursive cast_ray function, but I also need to pass Spheres to that function
constexpr std::uint32_t NUM_SPHERES{100U};

// AoS is ~88% faster than SoA
struct Spheres
{
    Vec3f center[NUM_SPHERES];
    Vec3f velocity[NUM_SPHERES];
    Vec3f acceleration[NUM_SPHERES];
    float radius[NUM_SPHERES];
    Material material[NUM_SPHERES];

    __host__ __device__ void set(const std::uint32_t i, const Vec3f& c, const Vec3f& v, const Vec3f& a, const float r, const Material& m)
    {
        center[i] = c;
        velocity[i] = v;
        acceleration[i] = a;
        radius[i] = r;
        material[i] = m;
    }

    __host__ __device__ void set(const std::uint32_t i, const Spheres& spheres)
    {
        center[i] = spheres.center[i];
        velocity[i] = spheres.velocity[i];
        acceleration[i] = spheres.acceleration[i];
        radius[i] = spheres.radius[i];
        material[i] = spheres.material[i];
    }

    __host__ __device__ bool ray_intersect(const std::uint32_t i, const Vec3f& orig, const Vec3f& dir, float& t0) const 
    {
        const Vec3f L = center[i] - orig;
        const float tca = L*dir;
        const float d2 = L*L - tca*tca;
        const float r = radius[i];
        if (d2 > r*r)
        {
            return false;
        }

        const float thc = sqrtf(r*r - d2);
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

} // namespace cuda_fun
#endif