#ifndef _CUDA_FUN_COMPLEX_HPP_
#define _CUDA_FUN_COMPLEX_HPP_

#include <cuda.h>

namespace cuda_fun
{

class Complex
{
public:
    __device__ Complex(const float real) : m_real{real}, m_imag{0.0F}
    {}

    __device__ Complex(const float real, const float imag) : m_real{real}, m_imag{imag}
    {}

    __device__ float real()      const noexcept { return m_real; }
    __device__ float imag() const noexcept { return m_imag; }

private:
    float m_real{0.0F};
    float m_imag{0.0F};
};

__device__ Complex operator+(const Complex& lhs, const Complex& rhs)
{
    return Complex{lhs.real() + rhs.real(), lhs.imag() + rhs.imag()};
}

__device__ Complex operator-(const Complex& lhs, const Complex& rhs)
{
    return Complex{lhs.real() - rhs.real(), lhs.imag() - rhs.imag()};
}

__device__ Complex operator*(const Complex& lhs, const Complex& rhs)
{
    const float real = lhs.real()*rhs.real() - lhs.imag()*rhs.imag();
    const float imag = lhs.real()*rhs.imag() + lhs.imag()*rhs.real();
    return Complex{real, imag};
}

__device__ Complex operator/(const Complex& lhs, const Complex& rhs)
{
    const float a = lhs.real();
    const float b = lhs.imag();
    const float c = rhs.real();
    const float d = rhs.imag();
    const float denom = c*c + d*d;
    const float x = (a*c + b*d) / denom;
    const float y = (b*c - b*d) / denom;
    return Complex{x, y};
}

__device__ Complex operator*(const float lhs, const Complex& rhs)
{
    return Complex{lhs*rhs.real(), lhs*rhs.imag()};
}

}

// CUDA math function overloads
__device__ cuda_fun::Complex powf(const cuda_fun::Complex& base, const float power)
{
    const float real = base.real();
    const float imag = base.imag();
    const float r = sqrtf(real*real + imag*imag);
    const float theta = atan2f(imag, real);
    const float factor = powf(r, power);
    return cuda_fun::Complex{factor*cosf(power*theta), factor*sinf(power*theta)};
}

__device__ cuda_fun::Complex coshf(const cuda_fun::Complex& c)
{
    return cuda_fun::Complex{coshf(c.real())*cosf(c.imag()), sinhf(c.real())*sinf(c.imag())};
}

__device__ cuda_fun::Complex sinhf(const cuda_fun::Complex& c)
{
    return cuda_fun::Complex{(c.real())*cosf(c.imag()), coshf(c.real())*sinf(c.imag())};
}

#endif