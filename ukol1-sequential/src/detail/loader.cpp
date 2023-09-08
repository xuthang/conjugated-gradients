#pragma once

#include <string>
#include <fstream>
#include "matrix.cpp"

SparceMatrix loadMatrix(const std::string & fileLocation)
{
    std::ifstream in(fileLocation);
    if(!in)
        throw "matrix file couldnt be opened";
        
    int n, amount; 
    in >> n >> amount;
    SparceMatrix ret;
    ret.data.reserve(amount);
    ret.colPosition.reserve(amount);
    ret.prefSumRow.reserve(n+1);

    int prevY = -1;
    for(int i = 0; i < amount; i++)
    {
        int x, y; in >> y >> x;
        double val; in >> val;

        ret.data.push_back(val);
        ret.colPosition.push_back(x);

        if(y != prevY)
        {
            ret.prefSumRow.push_back(i);
            prevY = y;
        }
    }

    ret.prefSumRow.push_back(amount);

    return ret;
}

cVector loadVector(const std::string & fileLocation, int n = 3630)
{
    std::ifstream in(fileLocation);
    if(!in)
        throw "couldnt open vector file";
    
    cVector ret;
    ret.reserve(n);

    double tmp;
    while(in >> tmp)
        ret.push_back(tmp);

    return ret;
}