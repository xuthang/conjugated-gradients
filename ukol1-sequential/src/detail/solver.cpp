#pragma once
#include "matrix.cpp"
#include "vector.cpp"

cVector cg(const SparceMatrix &A, const cVector &b,
                            int maxIterations = 1000, double eps = 1e-12)
{
    cVector x(b.size(), 0);

    cVector r = b - A * x; //parallel 2 operations
    cVector s = r;
    double tmp_rr = r*r; //reduction
    for (int i = 0; i < maxIterations; i++)
    {
        cVector tmp_As = A * s; //mult

        double alpha = tmp_rr / (s * tmp_As); //reduction

        //parallel
        x = x + alpha * s; //apply change and add to x
        r = r - alpha * tmp_As; //parallel

        double magnitude = vectorMagnitude(r); //parallel

        if (magnitude < eps) 
            break;
        
        double tmp_rnext_rnext = r * r; //reduction

        double beta = tmp_rnext_rnext / tmp_rr;
        s = r + beta*s; //apply change and add

        tmp_rr = tmp_rnext_rnext;
    }
    
    return x;
}
