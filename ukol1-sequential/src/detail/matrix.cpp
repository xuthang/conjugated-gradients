#pragma once
#include "vector.cpp"
#include<vector>
#include<iostream>
using namespace std;
class SparceMatrix
{
public:
    vector<double> data;
    vector<int> colPosition;
    vector<int> prefSumRow;

    cVector operator*(const cVector & v) const
    {
        cVector ret(v.size(), 0);
        for(int i = 0; i < (int)prefSumRow.size() - 1; i++)
        {
            int n = prefSumRow[i+1] - prefSumRow[i];
            for(int j = 0; j < n; j++)
            {
                int dataOffset = prefSumRow[i] + j;
                int y = i, x = colPosition[dataOffset];

                ret[y] += v[x] * data[dataOffset];
            }
        }
        return ret;
    }
};