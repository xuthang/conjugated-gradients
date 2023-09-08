#pragma once
#include <cstddef>


template <bool symmetric, typename T, template <typename...> class cContainer>
class TemplatedSparseMatrix_T
{
public:
    cContainer<T> data;
    int nonZeros;
    cContainer<int> colPosition;
    cContainer<int> prefSumRow;
    int n;
    const static bool isStoredSymmetrically = symmetric;
};

template <typename T, template <typename...> class cContainer>
class SymmetricallyStoredSparseMatrix_T: public TemplatedSparseMatrix_T<true, T, cContainer>{};

template <typename T, template <typename...> class cContainer>
class FullSparseMatrix_T: public TemplatedSparseMatrix_T<false, T, cContainer>{};

template <typename Matrix, typename cVector>
cVector MatrixMult(const Matrix &A, const cVector &v)
{
    cVector ret(v.size(), 0);
    for (size_t i = 0; i < A.prefSumRow.size() - 1; i++)
    {
        int n = A.prefSumRow[i + 1] - A.prefSumRow[i];
        for (int j = 0; j < n; j++)
        {
            int dataOffset = A.prefSumRow[i] + j;
            int y = i, x = A.colPosition[dataOffset];

            ret[y] += v[x] * A.data[dataOffset];

            // diagonal is saved as 1/2
            if(A.isStoredSymmetrically)
                ret[x] += v[y] * A.data[dataOffset];
        }
    }
    return ret;
}
