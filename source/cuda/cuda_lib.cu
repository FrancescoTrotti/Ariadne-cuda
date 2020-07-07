#include <iostream>
#include <cuda.h>
#include "./cuda_lib.hpp"

const int BLOCK_SIZE_X = 16;
const int BLOCK_SIZE_Y = 16;

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

void function(const int N, int * h_matrixA, int * h_matrixB, int * h_matrixC) {
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
}
