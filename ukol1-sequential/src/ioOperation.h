#pragma once
#include <iostream>
#include <vector>
#include "MatrixOperation.h"
#include "VectorOperation.h"
//--------------------------------------------------------

#define deb(x) std::cerr << #x << " " << x << std::endl;

template <template<typename...> class Container, typename T>
std::ostream &operator<<(std::ostream &out, const Container<T> &v)
{
    out << "\t[ ";

    if (v.size() > 10)
    {

        for (int i = 0; i < 5; i++)
            out << v[i] << " ";

        out << " ... ";

        for (int i = 0; i < 5; i++)
            out << v[v.size() - 1 - i] << " ";
    }
    else
    {
        for (auto x : v)
            out << x << " ";
    }

    return out << " ] ";
}


template <typename Matrix>
bool loadMatrix(const std::string &fileLocation, Matrix & ret)
{
    std::ifstream in(fileLocation);
    if (!in) return false;

    int n, amount;
    if(! (in >> n >> amount)) return false;
    
    std::vector<std::pair<std::pair<int, int>, double>> elements;
    for(int i = 0; i < amount; i++)
    {
        int x, y;
        if( !(in >> x >> y)) return false;
        double val;
        if(!(in >> val)) return false;

        if(Matrix::isStoredSymmetrically)
        {
            elements.push_back({{y, x}, val});
        }
        else
        {
            if(x == y)
            {
                elements.push_back({{y, x}, val*2});
            }
            else
            {
                elements.push_back({{x, y}, val});
                elements.push_back({{y, x}, val});
            }            
        }
    }

    sort(elements.begin(), elements.end());

    amount = (int)elements.size();

    ret = Matrix();
    ret.data.reserve(amount);
    ret.colPosition.reserve(amount);
    ret.prefSumRow.reserve(n + 1);

    int Yprev = -1;
    for(int i = 0; i < amount; i++)
    {
        auto &e = elements[i];
        ret.data.push_back(e.second);
        ret.colPosition.push_back(e.first.second);
        if(e.first.first != Yprev)
        {
            ret.prefSumRow.push_back(i);
            Yprev = e.first.first;
        }
    }


    ret.prefSumRow.push_back(amount);
    ret.n = n;
    ret.nonZeros = amount;

    return true;
}


// template <typename Matrix>
// bool loadMatrix(const std::string &fileLocation, Matrix & ret)
// {
//     std::ifstream in(fileLocation);
//     if (!in) return false;

//     int n, amount;
//     if(! (in >> n >> amount)) return false;
//     ret.data.clear();
//     ret.data.reserve(amount);

//     ret.colPosition.clear();
//     ret.colPosition.reserve(amount);

//     ret.prefSumRow.clear();
//     ret.prefSumRow.reserve(n + 1);

//     int prevY = -1;
//     for (int i = 0; i < amount; i++)
//     {
//         int x, y;
//         if( !(in >> x >> y)) return false;
//         double val;
//         if(!(in >> val)) return false;

//         ret.data.push_back(val);
//         ret.colPosition.push_back(x);

//         while (y != prevY)
//         {
//             ret.prefSumRow.push_back(i);
//             prevY++;
//         }
//     }

//     ret.prefSumRow.push_back(amount);
//     ret.n = n;
//     ret.nonZeros = amount;

//     return true;
// }

// template<typename cVector, typename cVectorType>
// cVector loadVector(const std::string &fileLocation)
// {
//     std::ifstream in(fileLocation);
//     if (!in)
//         throw "couldnt open vector file";

//     cVector ret;

//     cVectorType tmp;
//     while (in >> tmp)
//         ret.push_back(tmp);

//     return ret;
// }