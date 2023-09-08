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

using HostSparseMatrix = SparseMatrix_T<cVectorType, thrust::host_vector>;

struct cudaDataSparseMatrix
{
    cudaDataPtr data;
    int *colPosition;
    int nonZeros;

    int *prefSumRow;
    int n; // dimension of matrix
};

class DeviceSparseMatrix : public SparseMatrix_T<cVectorType, thrust::device_vector>
{
public:
    cudaDataSparseMatrix getView()
    {
        return cudaDataSparseMatrix{
            data.data().get(), colPosition.data().get(), nonZeros,
            prefSumRow.data().get(), n};
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

// ret = A + alpha*B
__global__ void cudaVectorPlus(cudaDataPtr a, cVectorType alpha, cudaDataPtr b, cudaDataPtr ret, int n)
{
    int i = getGlobalIdx();
    if (i < n)
        ret[i] = a[i] + alpha * b[i];
}

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

__global__ void cudaVectorMult(cudaDataPtr a, cudaDataPtr b, cudaDataElem ret, int n)
{
    int i = getGlobalIdx();

    cVectorType data = (i < n) ? a[i] * b[i] : 0;
    __syncthreads();
    data = blockReduceSum(data);
    __syncthreads();

    if (threadIdx.x == 0)
        atomicAdd(ret, data);
}

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
        atomicAdd(&(ret[x]), v[y] * A.data[dataOffset]);
    }

    __syncthreads();
    cVectorType totalRes = blockReduceSum(res1);
    __syncthreads();
    int y = i;
    if (threadIdx.x == 0)
        atomicAdd(&(ret[y]), totalRes);
}

//--------------------------------------------------------

device_vector<cVectorType> cg(DeviceSparseMatrix &ADeviceMem, device_ptr<cVectorType> bVector, int n,
                              int maxIterations = 5000, double eps = 1e-9)
{
    auto A = ADeviceMem.getView();
    int nonzeros = ADeviceMem.prefSumRow.back();
    cudaDataPtr b = bVector.get();

    // memory allocation
    device_vector<cVectorType> xDeviceMem(n, 0), rDeviceMem(n), sDeviceMem(n), AsDeviceMem(n);
    cudaDataPtr x = xDeviceMem.data().get(), r = rDeviceMem.data().get(), s = sDeviceMem.data().get(), As = AsDeviceMem.data().get();

    // used in order to store result, which can be copied back to host later
    device_vector<cVectorType> elemResultDeviceMem(1);
    cudaDataElem elemResult = elemResultDeviceMem.data().get();

    //-----------------------------------------------------------------------
    // calculate how many blocks and threads are needed

    const int threadCnt = 512;
    // CC = vector vector operation
    const int VV_blocksCnt = n / threadCnt + (n % threadCnt != 0);
    // MV = Matrix vector operation
    // const int MV_blocksCnt = nonzeros / threadCnt + (nonzeros % threadCnt != 0);

    //-----------------------------------------------------------------------

    auto &Ax = As;
    // Ax = A*x
    cudaMatrixMult<<<n, threadCnt>>>(A, x, Ax, n);
    // r = b - Ax;
    cudaVectorPlus<<<VV_blocksCnt, threadCnt>>>(b, -1, Ax, r, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    // s = r
    sDeviceMem = rDeviceMem;

    // rr = r*r; precalc r*r which can be reused
    cudaVectorMult<<<VV_blocksCnt, threadCnt>>>(r, r, elemResult, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    double rr = elemResultDeviceMem[0];
    CUDA_CHECK(cudaDeviceSynchronize());
    double old_rr;

    //-----------------------------------------------------------------------

    for (int i = 0; i < maxIterations; i++)
    {
        // 1st, calc direction of change and by how much
        {
            // prepare As array, need 0 on all positions to use += operator
            thrust::fill(thrust::device, AsDeviceMem.begin(), AsDeviceMem.end(), 0);
            // A*s, remember As to be reused later
            cudaMatrixMult<<<n, threadCnt>>>(A, s, As, n);
            CUDA_CHECK(cudaDeviceSynchronize());

            // s*As
            thrust::fill(thrust::device, elemResultDeviceMem.begin(), elemResultDeviceMem.end(), 0);
            cudaVectorMult<<<VV_blocksCnt, threadCnt>>>(s, As, elemResult, n); // zipped parallel reduction +s[i]*As[i]
            CUDA_CHECK(cudaDeviceSynchronize());
            double sAs = elemResultDeviceMem[0];
            CUDA_CHECK(cudaDeviceSynchronize());

            double alpha = rr / sAs;

            // x = x + a*s, r = r - a*As
            cudaVectorPlus<<<VV_blocksCnt, threadCnt>>>(x, alpha, s, x, n);
            cudaVectorPlus<<<VV_blocksCnt, threadCnt>>>(r, -alpha, As, r, n);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // check if the change isnt too small
        {
            old_rr = rr;

            // precalc r*r which can be reused
            thrust::fill(thrust::device, elemResultDeviceMem.begin(), elemResultDeviceMem.end(), 0);
            cudaVectorMult<<<VV_blocksCnt, threadCnt>>>(r, r, elemResult, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            rr = elemResultDeviceMem[0];
            CUDA_CHECK(cudaDeviceSynchronize());

            double magnitude = sqrt(rr);

            cerr << i << " " << magnitude << endl;

            if (magnitude < eps)
                break;
        }

        // update the direction for next iteration
        {
            double beta = rr / old_rr;
            // s = r - b*s
            cudaVectorPlus<<<VV_blocksCnt, threadCnt>>>(r, beta, s, s, n);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    return xDeviceMem;
}

//--------------------------------------------------------
