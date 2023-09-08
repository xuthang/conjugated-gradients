#pragma once

#include <bits/stdc++.h>
using namespace std;

#define deb(x) std::cerr << #x << " " << x << std::endl;

#include "util/MatrixOperation.h"
#include "util/ioOperation.h"

//--------------------------------------------------------
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
using namespace thrust;

#define CUDA_CHECK(ans)                       \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

//--------------------------------------------------------

using cVectorType = float;
using hostVector = thrust::host_vector<cVectorType>;
using deviceVector = thrust::device_vector<cVectorType>;

using cudaDataPtr = cVectorType *;
using cudaDataElem = cVectorType *;

#define baseMatrix SymmetricallyStoredSparseMatrix_T
// #define baseMatrix FullSparseMatrix_T

using HostSparseMatrix = baseMatrix<cVectorType, thrust::host_vector>;

struct cudaDataSparseMatrix
{
    cudaDataPtr data;
    int *colPosition;
    int nonZeros;

    int *prefSumRow;
    int n; // dimension of matrix
    bool isStoredSymmetrically;
};

class DeviceSparseMatrix : public baseMatrix<cVectorType, thrust::device_vector>
{
public:
    cudaDataSparseMatrix getView()
    {
        return cudaDataSparseMatrix{
            thrust::raw_pointer_cast( data.data() ), thrust::raw_pointer_cast( colPosition.data() ), nonZeros,
            thrust::raw_pointer_cast( prefSumRow.data() ), n,
            isStoredSymmetrically
            };
    }
};

//--------------------------------------------------------

DeviceSparseMatrix copyMatrixToDevice(const HostSparseMatrix &matrix)
{
    DeviceSparseMatrix ret;
    ret.data = matrix.data;
    ret.colPosition = matrix.colPosition;
    ret.nonZeros = matrix.nonZeros;

    ret.prefSumRow = matrix.prefSumRow;
    ret.n = matrix.n;

    return ret;
}

//--------------------------------------------------------
// cuda operations

__device__ inline int getGlobalIdx() { return blockIdx.x * blockDim.x + threadIdx.x; }

//--------------------------------------------------------

// ret = A + alpha*B
__global__ void cudaVectorPlus(cudaDataPtr a, int sign, cudaDataPtr b, cudaDataPtr ret, int n)
{
    int i = getGlobalIdx();
    if (i < n)
        ret[i] = a[i] + sign * b[i];
}

__global__ void cudaVectorPlus(cudaDataPtr a, int sign, cudaDataElem alpha, cudaDataElem beta, cudaDataPtr b, cudaDataPtr ret, int n)
{
    cVectorType coef = sign * (*alpha) / (*beta);
    int i = getGlobalIdx();
    if (i < n)
        ret[i] = a[i] + coef * b[i];
}

//--------------------------------------------------------

__device__ cVectorType warpReduceSum(cVectorType initVal)
{
    const unsigned int maskConstant = 0xffffffff; // not used
    for (unsigned int mask = warpSize / 2; mask > 0; mask >>= 1)
        initVal += __shfl_xor_sync(maskConstant, initVal, mask);

    return initVal;
}

__device__ cVectorType blockReduceSum(cVectorType val)
{
    static __shared__ cVectorType shared[32];
    int lane = threadIdx.x & (warpSize - 1); //== threadIdx.x % 32
    int warp_id = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0)
        shared[warp_id] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : (cVectorType)0;

    if (warp_id == 0)
        val = warpReduceSum(val);

    if (threadIdx.x == 0)
        shared[0] = val;
    __syncthreads();

    return shared[0];
}

//--------------------------------------------------------

__global__ void cudaVectorMult(cudaDataPtr a, cudaDataPtr b, cudaDataElem ret, int n)
{
    int i = getGlobalIdx();

    cVectorType data = (i < n) ? a[i] * b[i] : 0;
    data = blockReduceSum(data);

    if (threadIdx.x == 0)
        atomicAdd(ret, data);
}

//--------------------------------------------------------

__global__ void cudaMatrixMult(cudaDataSparseMatrix A, cudaDataPtr v, cudaDataPtr ret, int n)
{
    int i = blockIdx.x;

    int start = A.prefSumRow[i], end = A.prefSumRow[i + 1];
    int elemInRow = end - start;

    cVectorType res1 = 0;
    for (int j = threadIdx.x; j < elemInRow; j += blockDim.x)
    {
        int dataOffset = start + j;
        int y = i, x = A.colPosition[dataOffset];

        // row
        res1 += v[x] * A.data[dataOffset];

        // column
        //  todo write into memory for symetric case with better memory access
        //matrix is symmetrical, but diagonals are stored as 1/2
        if(A.isStoredSymmetrically)
            atomicAdd(&(ret[x]), v[y] * A.data[dataOffset]);
    }

    cVectorType totalRes = blockReduceSum(res1);
    int y = i;
    if (threadIdx.x == 0)
        atomicAdd(&(ret[y]), totalRes);
}

//--------------------------------------------------------

device_vector<cVectorType> cg(DeviceSparseMatrix &ADeviceMem, device_ptr<cVectorType> bVector, int n,
                              int maxIterations = 2000, double eps = 1e-6)
{
    auto A = ADeviceMem.getView();
    int nonzeros = ADeviceMem.prefSumRow.back();
    cudaDataPtr b = bVector.get();

    // memory allocation

    // init x0 choice
    device_vector<cVectorType> xDeviceMem(n, 0);
    cudaDataPtr x = thrust::raw_pointer_cast( xDeviceMem.data() );

    device_vector<cVectorType> rDeviceMem(n), sDeviceMem(n), AsDeviceMem(n);
    cudaDataPtr r = thrust::raw_pointer_cast( rDeviceMem.data() ), s = thrust::raw_pointer_cast( sDeviceMem.data() ), As = thrust::raw_pointer_cast( AsDeviceMem.data() );

    // used in order to store result, which can be copied back to host later
    device_vector<cVectorType> rrDeviceMem(maxIterations + 7, 0);
    cudaDataPtr rrCudaElems = thrust::raw_pointer_cast( rrDeviceMem.data() );

    device_vector<cVectorType> sAsDeviceMem(maxIterations + 7, 0);
    cudaDataPtr sAsCudaElems = thrust::raw_pointer_cast( sAsDeviceMem.data() );
    //-----------------------------------------------------------------------
    // calculate how many blocks and threads are needed

    const int threadCnt = 512;
    // CC = vector vector operation
    const int VV_blocksCnt = n / threadCnt + (n % threadCnt != 0);
    const int MV_blocksCnt = n;
    // MV = Matrix vector operation
    // const int MV_blocksCnt = nonzeros / threadCnt + (nonzeros % threadCnt != 0);

    //-----------------------------------------------------------------------

    auto &Ax = As;
    // Ax = A*x
    cudaMatrixMult<<<MV_blocksCnt, threadCnt>>>(A, x, Ax, n);
    // r = b - Ax;
    cudaVectorPlus<<<VV_blocksCnt, threadCnt>>>(b, -1, Ax, r, n);

    // s = r
    sDeviceMem = rDeviceMem;

    // rr = r*r; precalc r*r which can be reused
    cudaVectorMult<<<VV_blocksCnt, threadCnt>>>(r, r, rrCudaElems, n);

    //-----------------------------------------------------------------------
    double magnitude = 0;
    int i = 0;
    for (; i < maxIterations; i++)
    {
        cudaDataElem rr = rrCudaElems + i;
        cudaDataElem rrNext = rrCudaElems + i + 1;
        cudaDataElem sAs = sAsCudaElems + i;

        // 1st, calc direction of change and by how much
        {
            // prepare As array, need 0 on all positions to use += operator
            // A*s, remember As to be reused later
            thrust::fill(thrust::device, AsDeviceMem.begin(), AsDeviceMem.end(), 0);
            cudaMatrixMult<<<MV_blocksCnt, threadCnt>>>(A, s, As, n);

            // s*As, zipped parallel reduction +s[i]*As[i]
            cudaVectorMult<<<VV_blocksCnt, threadCnt>>>(s, As, sAs, n);

            // a = rr / sAs;
            // x = x + a*s, r = r - a*As
            cudaVectorPlus<<<VV_blocksCnt, threadCnt>>>(x, 1, rr, sAs, s, x, n);
            cudaVectorPlus<<<VV_blocksCnt, threadCnt>>>(r, -1, rr, sAs, As, r, n);
        }

        // check if the residuum isnt too small
        {
            cudaVectorMult<<<VV_blocksCnt, threadCnt>>>(r, r, rrNext, n);

            // CUDA_CHECK(cudaDeviceSynchronize());
            double rr = rrDeviceMem[i + 1]; // copy is synchronous, no need to synchronize manually

            magnitude = sqrt(rr);
            cerr << i << " " << magnitude << endl;
            if (magnitude < eps)
            {
                CUDA_CHECK(cudaGetLastError());
                break; // breaks for-loop and returns the result
            }
        }

        // update the direction for next iteration
        {
            // b = rrnext/rr
            //  s = r + b*s
            cudaVectorPlus<<<VV_blocksCnt, threadCnt>>>(r, 1, rrNext, rr, s, s, n);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    cout << i << " " << magnitude << endl;
    return xDeviceMem;
}

//--------------------------------------------------------
