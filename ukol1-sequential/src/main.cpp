#include <bits/stdc++.h>
using namespace std;

#define deb(x) std::cerr << #x << " " << x << std::endl;

#include "../../util/timer.h"
#include "VectorOperation.h"
#include "ioOperation.h"
#include "MatrixOperation.h"

//--------------------------------------------------------

using cVectorType = double;
using cVector = std::vector<cVectorType>;
// using Matrix = FullSparseMatrix_T<cVectorType, std::vector>;
using Matrix = SymmetricallyStoredSparseMatrix_T<cVectorType, std::vector>;

//--------------------------------------------------------

template<typename Matrix>
cVector cg(const Matrix &A, const cVector &b, int maxIterations = 2000, double eps = 1e-6)
{
    cVector x(b.size(), 0); // init x

    cVector r = vectorPlus(b, scalarVectorMult(-1, MatrixMult(A, x))), s = r; // residuum = b-Ax, parallel 2 operations

    cVectorType rr = vectorMult<cVector, cVectorType>(r, r);                                       // precalc r*r which can be reused, parallel reduction +r[i]^1
    cVectorType old_rr = rr;

    int i = 0;
    double magnitude = 0;
    for (; i < maxIterations; i++)
    {
        // 1st, calc direction of change and by how much
        {
            cVector As = MatrixMult(A, s);  // most important multiplication
            cVectorType sAs = vectorMult<cVector, cVectorType>(s, As); // zipped parallel reduction +s[i]*As[i]
            double alpha = rr / sAs;
            // parallel
            vectorPlus(x, scalarVectorMult(alpha, s), x);   // apply change on x, in parallel can skip vectorMult by specializing a kernel
            vectorPlus(r, scalarVectorMult(-alpha, As), r); // same as with x, important usege of -alpha
        }

        // check if the change isnt too small
        {
            old_rr = rr;
            rr = vectorMult<cVector, cVectorType>(r, r);
            magnitude = sqrt(rr);
			
            cerr << i << " " << magnitude << endl; // no printing in parallel
            if (magnitude < eps)
                break;
        }

        // update the direction for next iteration
        {
            double beta = rr / old_rr;
            s = vectorPlus(r, scalarVectorMult(beta, s)); // sameMethod as with x
        }
    }

    cout << i << " " << magnitude << endl;
    return x;
}

//--------------------------------------------------------

int main(int argc, char *argv[])
{
    if (argc < 2 || argc > 3)
    {
        cerr << "usage: ./a.out [matrix file] [optional righ hand vector file]" << endl;
        return 1;
    }

    cerr << "starting" << endl;

    Matrix A;
    if(!loadMatrix(argv[1], A)) throw "couldnt load matrix";
    cVector b = MatrixMult(A, cVector(A.n, 2));

    cVector x;
    {
        TIMER t;
        x = cg(A, b);
    }

    // cout << "A*x = " << MatrixMult(A, x) << endl;
    // cout << "b = " << b << endl;
    return 0;
}
