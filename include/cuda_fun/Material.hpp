#ifndef _CUDA_FUN_MATERIAL_HPP_
#define _CUDA_FUN_MATERIAL_HPP_

#include <cuda_fun/Vector.hpp>

namespace cuda_fun
{

struct Material
{
    __host__ __device__ Material(const Vec3f& color, const Vec3f& a, const float s) : 
        diffuse_color{color},
        albedo{a},
        specular_exponent{s}
    {}

    __host__ __device__ Material() : 
        diffuse_color{},
        albedo{},
        specular_exponent{0.0}
    {}

    Vec3f diffuse_color;
    Vec3f albedo;
    float specular_exponent;
};

}
#endif