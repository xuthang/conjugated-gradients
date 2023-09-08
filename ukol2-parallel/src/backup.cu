#include <bits/stdc++.h>
using namespace std;

#define deb(x) std::cerr << #x << " " << x << std::endl;

template <typename T>
ostream &operator<<(ostream &out, const vector<T> &v)
{
    for (const auto &x : v)
        out << x << ' ';
    return out;
}

//--------------------------------------------------------
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

//--------------------------------------------------------

using cVectorType = float;
using hostVector = thrust::host_vector<cVectorType>;
using deviceVector = thrust::device_vector<cVectorType>;

class SparseMatrix
{
public:
    vector<cVectorType> data;
    vector<int> colPosition;
    int nonZeros;

    vector<int> prefSumRow;
    int n;
};

struct DeviceSparseMatrix
{
    cVectorType *data;
    int *colPosition;
    int nonZeros;

    int *prefSumRow;
    int n; //dimension of matrix
};

//--------------------------------------------------------

std::ostream &operator<<(std::ostream &out, const hostVector &v)
{
    out << "\t[ ";
    for (int i = 0; i < 5; i++)
        out << v[i] << " ";

    out << " ... ";

    for (int i = 0; i < 5; i++)
        out << v[v.size() - 1 - i] << " ";

    return out << " ] ";
}

hostVector vectorPlus(const hostVector &a, const hostVector &b, hostVector &ret)
{
    for (size_t i = 0; i < a.size(); i++)
        ret[i] = a[i] + b[i];
    return ret;
}

hostVector vectorPlus(const hostVector &a, const hostVector &b)
{
    hostVector ret(a.size(), 0);
    return vectorPlus(a, b, ret);
}

hostVector vectorMult(double alpha, const hostVector &v, hostVector &ret)
{
    for (size_t i = 0; i < ret.size(); i++)
        ret[i] *= alpha;
    return ret;
}

hostVector vectorMult(double alpha, const hostVector &v)
{
    hostVector ret = v;
    return vectorMult(alpha, v, ret);
}

cVectorType vectorMult(const hostVector &a, const hostVector &b)
{
    cVectorType ret = 0;
    for (size_t i = 0; i < a.size(); i++)
        ret += a[i] * b[i];
    return ret;
}

hostVector MatrixMult(const SparseMatrix &A, const hostVector &v)
{
    hostVector ret(v.size(), 0);
    for (size_t i = 0; i < A.prefSumRow.size() - 1; i++)
    {
        int elemInRow = A.prefSumRow[i + 1] - A.prefSumRow[i];
        for (int j = 0; j < elemInRow; j++)
        {
            int dataOffset = A.prefSumRow[i] + j;
            int y = i, x = A.colPosition[dataOffset];

            ret[y] += v[x] * A.data[dataOffset];
            ret[x] += v[y] * A.data[dataOffset];
        }
    }
    return ret;
}

//--------------------------------------------------------

SparseMatrix loadMatrix(const std::string &fileLocation)
{
    std::ifstream in(fileLocation);
    if (!in)
        throw "matrix file couldnt be opened";

    int n, amount;
    in >> n >> amount;
    SparseMatrix ret;
    ret.data.reserve(amount);
    ret.colPosition.reserve(amount);
    ret.prefSumRow.reserve(n + 1);

    int prevY = -1;
    for (int i = 0; i < amount; i++)
    {
        int x, y;
        in >> x >> y;
        cVectorType val;
        in >> val;

        ret.data.push_back(val);
        ret.colPosition.push_back(x);

        while (y != prevY)
        {
            ret.prefSumRow.push_back(i);
            prevY++;
        }
    }

    ret.prefSumRow.push_back(amount);
    ret.n = n;
    ret.nonZeros = amount;

    return ret;
}

hostVector loadVector(const std::string &fileLocation)
{
    std::ifstream in(fileLocation);
    if (!in)
        throw "couldnt open vector file";

    hostVector ret;

    cVectorType tmp;
    while (in >> tmp)
        ret.push_back(tmp);

    return ret;
}

//--------------------------------------------------------

// cuda operations

__device__ inline int getGlobalIdx() { return blockIdx.x * blockDim.x + threadIdx.x; }

// ret = A + alpha*B
__global__ void cudaVectorPlus(cVectorType *a, double alpha, cVectorType *b, cVectorType *ret, int n)
{
    int i = getGlobalIdx();
    if (i < n)
        ret[i] = a[i] + alpha * b[i];
}

__global__ void cudaVectorMult(cVectorType alpha, cVectorType *v, cVectorType *ret, int n)
{
    int i = getGlobalIdx();
    if (i < n)
        ret[i] = alpha * v[i];
}

__global__ void cudaVectorMult(cVectorType *a, cVectorType *b, cVectorType *ret, int n)
{
    __shared__ cVectorType sharedMem[1024];

    int i = getGlobalIdx();
    sharedMem[threadIdx.x] = (i < n) ? a[i] * b[i] : 0;
    __syncthreads();

    // todo: accumulate into threadIdx.x

    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(ret, sharedMem[0]);
}

__global__ void cudaMatrixMult(DeviceSparseMatrix *A, cVectorType *v, cVectorType *ret, int n)
{
    int i = blockIdx.x;

    int start = A->prefSumRow[i], end = A->prefSumRow[i + 1];
    int elemInRow = end - start;

    for (int j = threadIdx.x; j < elemInRow; j += blockDim.x)
    {
        int dataOffset = start + i;
        int y = i, x = A->colPosition[dataOffset];

        // todo write into memory
        atomicAdd(&(ret[y]), v[x] * A->data[dataOffset]);
        atomicAdd(&(ret[x]), v[y] * A->data[dataOffset]);
    }
}

//--------------------------------------------------------

hostVector cg(const SparseMatrix &A, const hostVector &b, int maxIterations = 5000, double eps = 1e-9)
{
    hostVector x(b.size(), 0); // init x

    // cg important variables
    hostVector r = vectorPlus(b, vectorMult(-1, MatrixMult(A, x))), s = r; // residuum = b-Ax, parallel 2 operations
    double rr = vectorMult(r, r);                                       // precalc r*r which can be reused, parallel reduction +r[i]^1
    double old_rr = rr;

    //Memory Allocation Management
    



    for (int i = 0; i < maxIterations; i++)
    {
        // 1st, calc direction of change and by how much
        {
            cVector As = MatrixMult(A, s);  // most important multiplication
            double sAs = vectorMult(s, As); // zipped parallel reduction +s[i]*As[i]
            double alpha = rr / sAs;

            // parallel
            vectorPlus(x, vectorMult(alpha, s), x);   // apply change on x, in parallel can skip vectorMult by specializing a kernel
            vectorPlus(r, vectorMult(-alpha, As), r); // same as with x, important usege of -alpha
        }

        // check if the change isnt too small
        {
            old_rr = rr;
            rr = vectorMult(r, r);
            double magnitude = rr;
            cerr << i << " " << sqrt(magnitude) << " " << MatrixMult(A, x) << endl; // no printing in parallel
            if (magnitude < eps * eps)
                break;
        }

        // update the direction for next iteration
        {
            double beta = rr / old_rr;
            s = vectorPlus(r, vectorMult(beta, s)); // sameMethod as with x
        }
    }

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

    SparseMatrix A = loadMatrix(argv[1]);
    hostVector b = (argc == 3) ? loadVector(argv[2]) : MatrixMult(A, hostVector(A.n, 2));

    auto x = cg(A, b);
    cout << "A*x = " << MatrixMult(A, x) << endl;
    cout << "b = " << b << endl;
    return 0;
}
