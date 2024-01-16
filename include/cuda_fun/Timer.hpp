#ifndef _CUDA_FUN_TIMER_HPP_
#define _CUDA_FUN_TIMER_HPP_

#include <chrono>
#include <string>

namespace cuda_fun
{

class Timer
{
public:
    Timer(const std::string& name)
    {
        m_name = name;
        m_start = std::chrono::high_resolution_clock::now();
    }

    ~Timer()
    {
        const auto end = std::chrono::high_resolution_clock::now();
        printf("%s: %ld ms\n", m_name.c_str(), std::chrono::duration_cast<std::chrono::milliseconds>(end - m_start).count());
    }

private:
    std::string m_name;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

}

#endif