#pragma once

template<typename cVector>
cVector vectorPlus(const cVector &a, const cVector &b, cVector &ret)
{
    for (size_t i = 0; i < a.size(); i++)
        ret[i] = a[i] + b[i];
    return ret;
}

template<typename cVector>
cVector vectorPlus(const cVector &a, const cVector &b)
{
    cVector ret(a.size(), 0);
    return vectorPlus(a, b, ret);
}

template<typename cVector>
cVector vectorMult(double alpha, const cVector &v, cVector &ret)
{
    for (size_t i = 0; i < ret.size(); i++)
        ret[i] *= alpha;
    return ret;
}

template<typename cVector, typename cVectorType>
cVectorType vectorMult(const cVector &a, const cVector &b)
{
    cVectorType ret = 0;
    for (size_t i = 0; i < a.size(); i++)
        ret += a[i] * b[i];
    return ret;
}

template<typename cVector>
cVector scalarVectorMult(double alpha, const cVector &v)
{
    cVector ret = v;
    return vectorMult(alpha, v, ret);
}

