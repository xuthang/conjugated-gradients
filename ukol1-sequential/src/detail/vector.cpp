#pragma once
#include <vector>
#include <cmath>
#include <iostream>

using cVector = std::vector<double>;

std::ostream& operator<<(std::ostream & out, const cVector & v)
{
    for(int i = 0; i < 5; i++)
        out << v[i] << " ";

    out << " ... ";

    for(int i = 0; i < 5; i++)
        out << v[v.size() -1 - i] << " ";
    
    return out << std::endl;
}

cVector operator+(const cVector &a, const cVector &b)
{
    cVector ret(a.size(), 0);
    for (size_t i = 0; i < a.size(); i++)
        ret[i] = a[i] + b[i];
    return ret;
}

cVector operator-(const cVector &a, const cVector &b)
{
    cVector ret(a.size(), 0);
    for (size_t i = 0; i < a.size(); i++)
        ret[i] = a[i] - b[i];
    return ret;
}

double operator*(const cVector &a, const cVector &b)
{
    double ret = 0;
    for (size_t i = 0; i < a.size(); i++)
        ret += a[i] * b[i];
    return ret;
}

cVector operator*(double alpha, const cVector &v)
{
    cVector ret = v;
    for (size_t i = 0; i < ret.size(); i++)
        ret[i] *= alpha;
    return ret;
}

double vectorMagnitude(const cVector & v)
{
    double ret = 0;
    for(size_t i = 0; i < v.size(); i++)
        ret += v[i]*v[i];
    return std::sqrt(ret);
}