/***************************************************************************
 *            test_procedure.cpp
 *
 *  Copyright  2010-20  Pieter Collins
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne.
 *
 *  Ariadne is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <stdexcept>

#include "config.hpp"

#include "function/procedure.hpp"
#include "function/procedure.tpl.hpp"

#include "../test.hpp"

#include <cuda.h>

using namespace std;
using namespace Ariadne;

const int BLOCK_SIZE_X = 16;
const int BLOCK_SIZE_Y = 16;

template<class X> decltype(auto) mag(Covector<X> const& u) { return norm(transpose(u)); }

class TestCudaProcedure
{
    DoublePrecision pr;
  public:
    TestCudaProcedure();
    Void test();
  private:
    Void test_matrix_moltiplication();
};

TestCudaProcedure::TestCudaProcedure()
{
}

Void TestCudaProcedure::test()
{
    ARIADNE_TEST_CALL(test_matrix_moltiplication());
}

__global__
void matrixMultiplicationKernel(const int* d_matrixA,
                                const int* d_matrixB,
                                int        N,
                                int*       d_matrixC) {
    int Row = blockIdx.y*blockDim.y + threadIdx.y;
    int Col = blockIdx.x*blockDim.x + threadIdx.x;
    
    int Pvalue = 0;
    if (Row < N && Col < N) {
        for (int k = 0; k < N; ++k)
            Pvalue += d_matrixA[Row*N+k] * d_matrixB[Col+k*N];        

        d_matrixC[Row*N+Col] = Pvalue;
    }
}

Void TestCudaProcedure::test_matrix_moltiplication()
{
    int N = 5;
    int* h_matrixA = new int[N * N];
    int* h_matrixB = new int[N * N];
    int* h_matrixC = new int[N * N];

    for (int i = 0; i < N * N; i++) {
        h_matrixA[i] = i;
        h_matrixB[i] = i+1;
    }

//---------------- GPU
    int *d_matrixA, *d_matrixB, *d_matrixC;
    cudaMalloc( &d_matrixA, N*N * sizeof(int) );
    cudaMalloc( &d_matrixB, N*N * sizeof(int) );
    cudaMalloc( &d_matrixC, N*N * sizeof(int) );

    cudaMemcpy( d_matrixA, h_matrixA, N*N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_matrixB, h_matrixB, N*N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 DimGrid(N/BLOCK_SIZE_X, N/BLOCK_SIZE_Y, 1);
    if (N%BLOCK_SIZE_X) DimGrid.x++;
    if (N%BLOCK_SIZE_Y) DimGrid.y++;
    dim3 DimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    
    matrixMultiplicationKernel<<< DimGrid,DimBlock>>> (d_matrixA, d_matrixB, N, d_matrixC);

    cudaMemcpy( h_matrixC, d_matrixC, N*N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "GPU: " << std::endl;
    for (int i = 0; i < N * N; i++){
        if (i % N == 0){
            std::cout << std::endl;
        }
        std::cout << h_matrixC[i] << " ";
    }
    std::cout << std::endl;

    cudaDeviceReset();

//-------------- CPU
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++)
                 sum += h_matrixA[i * N + k] * h_matrixB[k * N + j];
            h_matrixC[i * N + j] = sum;
        }
    }
    std::cout << "CPU: " << std::endl;
    for (int i = 0; i < N * N; i++){
        if (i % N == 0){
            std::cout << std::endl;
        }
        std::cout << h_matrixC[i] << " ";
    }
    std::cout << std::endl;
}



Int main() {
    TestCudaProcedure().test();
    return ARIADNE_TEST_FAILURES;
}

