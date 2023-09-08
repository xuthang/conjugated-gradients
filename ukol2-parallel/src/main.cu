#include "solver.cu"
#include "../../util/timer.h"

int main(int argc, char *argv[])
{
    if (argc < 2 || argc > 3)
    {
        cerr << "usage: ./a.out [matrix file] [optional righ hand vector file]" << endl;
        return 1;
    }

    cerr << "starting" << endl;

    HostSparseMatrix AHost;
    if(!loadMatrix(argv[1], AHost))
    {
        cerr<< "couldnt read matrix" << endl;
        return 1; 
    }
    
    DeviceSparseMatrix A = copyMatrixToDevice(AHost);
    deviceVector b = MatrixMult( AHost,  hostVector(A.n, 2));
    CUDA_CHECK(cudaDeviceSynchronize());
    deviceVector x(A.n);
    //check difference between float and double
    //porovnat s cusparse
    //vytahnout diagonalu a vynasobit zvlast
    {
        // TIMER t([](double val){cout << "time: " << val << "ms" << endl;});
        TIMER t([](double val){cout << val << endl;});
        x = cg(A, b.data(), b.size());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    hostVector xHost = x, bHost = b;
    // cout << "A*x = " << MatrixMult(AHost, xHost) << endl;
    // cout << "b = " << bHost << endl;

    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}
