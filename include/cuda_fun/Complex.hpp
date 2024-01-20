#ifndef _CUDA_FUN_COMPLEX_HPP_
#define _CUDA_FUN_COMPLEX_HPP_

#include <cuda.h>

namespace cuda_fun
{

class Complex
{
public:
    __device__ Complex(const float real, const float imag) : m_real{real}, m_imag{imag}
    {}

    __device__ float getReal()      const noexcept { return m_real; }
    __device__ float getImaginary() const noexcept { return m_imag; }

private:
    float m_real;
    float m_imag;
};

__device__ Complex operator+(const Complex& lhs, const Complex& rhs)
{
    return Complex{lhs.getReal() + rhs.getReal(), lhs.getImaginary() + rhs.getImaginary()};
}

__device__ Complex operator-(const Complex& lhs, const Complex& rhs)
{
    return Complex{lhs.getReal() - rhs.getReal(), lhs.getImaginary() - rhs.getImaginary()};
}

__device__ Complex operator*(const Complex& lhs, const Complex& rhs)
{
    const float real = lhs.getReal()*rhs.getReal() - lhs.getImaginary()*rhs.getImaginary();
    const float imag = lhs.getReal()*rhs.getImaginary() + lhs.getImaginary()*rhs.getReal();
    return Complex{real, imag};
}

}

// CUDA math function overloads
__device__ cuda_fun::Complex powf(const cuda_fun::Complex& base, const float power)
{
    const float real = base.getReal();
    const float imag = base.getImaginary();
    const float r = sqrtf(real*real + imag*imag);
    const float theta = atan2f(imag, real);
    const float factor = powf(r, power);
    return cuda_fun::Complex{factor*cosf(power*theta), factor*sinf(power*theta)};
}

#endif