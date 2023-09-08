#pragma once

#include <string>
#include <chrono>
#include <functional>
#include <iostream>
#include <cassert>

struct TIMER
{
    std::function<void(double)> f;
    std::chrono::high_resolution_clock::time_point begin;
    bool stopped = false;
    double totalTime = 0;

    TIMER(std::function<void(double)> func = [](double res)
          { std::cout << res << std::endl; })
        : f(func)
    {
        start(true);
    }

    ~TIMER()
    {
        if (!stopped)
            stop(true);
    }

    double stop(bool invokeFunction)
    {
        auto end = std::chrono::high_resolution_clock::now();
        stopped = true;
        totalTime += (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.);
        if (invokeFunction)
            f(totalTime);
        return totalTime;
    }

    void start(bool resetTimer)
    {
        if (resetTimer)
            totalTime = 0;
        stopped = false;
        begin = std::chrono::high_resolution_clock::now();
    }

    double get() const { return totalTime; }
};
