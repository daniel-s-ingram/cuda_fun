#ifndef _CUDA_FUN_VECTOR_HPP_
#define _CUDA_FUN_VECTOR_HPP_

#include <cuda.h>

#include <cmath>

namespace cuda_fun
{

struct Vec3f
{
    float x;
    float y;
    float z;

    __host__ __device__ Vec3f() : x{0.f}, y{0.f}, z{0.f}
    {}

    __host__ __device__ Vec3f(float a, float b, float c) : x{a}, y{b}, z{c}
    {}
    
    __host__ __device__ Vec3f operator*(const float s) const { return {s*x, s*y, s*z}; }
    __host__ __device__ float operator*(const Vec3f& v) const { return x*v.x + y*v.y + z*v.z; }
    __host__ __device__ Vec3f operator+(const Vec3f& v) const { return {x+v.x, y+v.y, z+v.z}; }
    __host__ __device__ Vec3f operator-(const Vec3f& v) const { return {x-v.x, y-v.y, z-v.z}; }
    __host__ __device__ Vec3f operator-() const { return {-x, -y, -z}; }
    __host__ __device__ Vec3f& operator+=(const Vec3f& v)
    {
        *this = *this + v;
        return *this;
    }

    __host__ __device__ float norm() const { return std::sqrt(x*x + y*y + z*z); }
    __host__ __device__ Vec3f normalized() const { return (*this)*(1.f/norm()); }
};

}

#endif