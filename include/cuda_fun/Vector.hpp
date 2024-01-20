#ifndef _CUDA_FUN_VECTOR_HPP_
#define _CUDA_FUN_VECTOR_HPP_

#include <cuda.h>

#include <cassert>
#include <cmath>

namespace cuda_fun
{

template<std::size_t Dimension, typename ElementType>
struct Vector
{
    ElementType elements[Dimension];

    __host__ __device__ Vector()
    {
        for (std::size_t idx = 0; idx < Dimension; ++idx)
        {
            elements[idx] = ElementType();
        }
    }

    template<typename... Elements>
    __host__ __device__ Vector(Elements&&... args)
    {
        static_assert(sizeof...(Elements) == Dimension);
        std::size_t idx{0};
        for (const auto& arg : {args...})
        {
            elements[idx++] = arg;
        }
    }

    __host__ __device__ ElementType& operator[](const std::size_t idx)
    {
        assert(idx < Dimension);
        return elements[idx];
    }

    __host__ __device__ const ElementType& operator[](const std::size_t idx) const
    {
        assert(idx < Dimension);
        return elements[idx];
    }

    __host__ __device__ ElementType norm() const 
    {
        ElementType sum_of_squares{};
        for (std::size_t idx = 0; idx < Dimension; ++idx)
        {
            const auto& element = elements[idx];
            sum_of_squares += element*element;
        }

        return sqrtf(sum_of_squares);
    }

    __host__ __device__ Vector<Dimension, ElementType> normalized() const 
    { 
        return (*this) * (static_cast<ElementType>(1.0) / norm()); 
    }

    __host__ __device__ Vector<Dimension, ElementType>& operator+=(const Vector<Dimension, ElementType>& rhs);
    __host__ __device__ Vector<Dimension, ElementType>& operator-=(const Vector<Dimension, ElementType>& rhs);
};

template<std::size_t Dimension, typename ElementType>
__host__ __device__ ElementType operator*(const Vector<Dimension, ElementType>& lhs, const Vector<Dimension, ElementType>& rhs)
{
    ElementType val{};
    for (std::size_t idx = 0; idx < Dimension; ++idx)
    {
        val += lhs[idx]*rhs[idx];
    }

    return val;
}

template<std::size_t Dimension, typename ElementType, typename ScalarType>
__host__ __device__ Vector<Dimension, ElementType> operator*(const Vector<Dimension, ElementType>& lhs, const ScalarType& rhs)
{
    Vector<Dimension, ElementType> vec;
    for (std::size_t idx = 0; idx < Dimension; ++idx)
    {
        vec[idx] = lhs[idx]*rhs;
    }

    return vec;
}

template<std::size_t Dimension, typename ElementType, typename ScalarType>
__host__ __device__ Vector<Dimension, ElementType> operator-(const Vector<Dimension, ElementType>& lhs, const ScalarType& rhs)
{
    Vector<Dimension, ElementType> vec;
    for (std::size_t idx = 0; idx < Dimension; ++idx)
    {
        vec[idx] = lhs[idx] - rhs;
    }

    return vec;
}

template<std::size_t Dimension, typename ElementType, typename ScalarType>
__host__ __device__ Vector<Dimension, ElementType> operator/(const Vector<Dimension, ElementType>& lhs, const ScalarType& rhs)
{
    Vector<Dimension, ElementType> vec;
    for (std::size_t idx = 0; idx < Dimension; ++idx)
    {
        vec[idx] = lhs[idx]/rhs;
    }

    return vec;
}

template<std::size_t Dimension, typename ElementType>
__host__ __device__ Vector<Dimension, ElementType> operator+(const Vector<Dimension, ElementType>& lhs, const Vector<Dimension, ElementType>& rhs)
{
    Vector<Dimension, ElementType> vec;
    for (std::size_t idx = 0; idx < Dimension; ++idx)
    {
        vec[idx] = lhs[idx] + rhs[idx];
    }

    return vec;
}

template<std::size_t Dimension, typename ElementType>
__host__ __device__ Vector<Dimension, ElementType> operator-(const Vector<Dimension, ElementType>& orig)
{
    Vector<Dimension, ElementType> vec;
    for (std::size_t idx = 0; idx < Dimension; ++idx)
    {
        vec[idx] = -orig[idx];
    }

    return vec;
}

template<std::size_t Dimension, typename ElementType>
__host__ __device__ Vector<Dimension, ElementType> operator-(const Vector<Dimension, ElementType>& lhs, const Vector<Dimension, ElementType>& rhs)
{
    return lhs + (-rhs);
}

template<std::size_t Dimension, typename ElementType>
__host__ __device__ Vector<Dimension, ElementType> operator*(const ElementType& lhs, const Vector<Dimension, ElementType>& rhs)
{
    return rhs*lhs;
}

template<std::size_t Dimension, typename ElementType>
__host__ __device__ Vector<Dimension, ElementType>& Vector<Dimension, ElementType>::operator+=(const Vector<Dimension, ElementType>& rhs)
{
    *this = *this + rhs;
    return *this;
}

template<std::size_t Dimension, typename ElementType>
__host__ __device__ Vector<Dimension, ElementType>& Vector<Dimension, ElementType>::operator-=(const Vector<Dimension, ElementType>& rhs)
{
    *this += (-rhs);
    return *this;
}

using Vec3f = Vector<3, float>;

}

#endif