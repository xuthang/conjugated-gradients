#include <bits/stdc++.h>
using namespace std;

#include "../MatrixOperation.h"
#include "../ioOperation.h"

using cVectorType = float;
using cVector = std::vector<cVectorType>;
using SparceMatrix = SparceMatrix_T<cVectorType, std::vector>;

int main(int argc, char *argv[])
{
    if (argc < 2 || argc > 3)
    {
        cerr << "usage: ./a.out [matrix file] [optional righ hand vector file]" << endl;
        return 1;
    }

    cerr << "starting" << endl;

    SparceMatrix A;
    loadMatrix(argv[1], A);
    auto b = MatrixMult(A, cVector(A.n, 2));
    cout << b << endl;
    cout << A.n << endl;
}