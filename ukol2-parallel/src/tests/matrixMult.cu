#include "../solver.cu"

int main(int argc, char *argv[])
{
    if (argc < 2 || argc > 3)
    {
        cerr << "usage: ./a.out [matrix file]" << endl;
        return 1;
    }

    HostSparseMatrix AHost = loadMatrix(argv[1]);
    int n = AHost.n;
    host_vector<cVectorType> vHost(n, 2);
    auto resHost = MatrixMult(AHost, vHost);
    cout << resHost << endl;

    DeviceSparseMatrix A = copyMatrixToDevice(AHost);
    device_vector<cVectorType> v = vHost;
    device_vector<cVectorType> Av(n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaMatrixMult<<<n, 512>>>(A.getView(), v.data().get(), Av.data().get(), n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cout << Av << endl;
}