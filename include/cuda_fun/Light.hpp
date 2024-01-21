#ifndef _CUDA_FUN_LIGHT_HPP_
#define _CUDA_FUN_LIGHT_HPP_

#include <cuda_fun/Vector.hpp>

namespace cuda_fun
{

struct Light
{
    Light(const Vec3f& p, const float i) :
        position{p},
        intensity{i}
    {}

    Vec3f position;
    float intensity;
};

} 
#endif