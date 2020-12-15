#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"
#include <immintrin.h>
#include <pthread.h>

#include <cuda_runtime.h>


int threads_per_block = 256;
int max_blocks_per_grid = 4096;

//Threads per block = tpb, mbpg = max blocks per grid
int set_grid_size(int tpb, int mbpg){
	if(tpb > 1024 || mbpg > 65535){
		return 0;
	}
	else{
		threads_per_block = tpb;
		max_blocks_per_grid = mbpg;
		return 1;
	}

}

//scalar multi gpu
__global__
void escalar(int n, float *d_x, float escalar)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;


    for (int i = index; i < n; i += stride) {
        d_x[i] = d_x[i] * escalar;
    }
}

//matrix matrix multi
__global__
void matrix_mult(int n, float *d_A, float *d_B, float *d_C, int widthA, int widthB, int widthC)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;


    for (int i = index; i < n; i += stride) {
		//linha de C = int(index/C.width)
		//linha de A = (linha de C)*A.width 
		//coluna de C = index%d_C->width
		int cLine = i/widthC;
		int aLine = cLine*widthA;
		int colC = i%widthC;
		
		//Zerando c
		d_C[i] = 0.0;
		
		//Iterando pela linha de A e achando resultado em um C
		int k = 0;
		for(int j = aLine; j < aLine + widthA; j++){
			d_C[i] += d_A[j] * d_B[k*widthB + colC];
			k++;
		}
    }
}

void preenche_matrix(Matrix *matrix, float val){
	unsigned long int m = matrix->height, n = matrix->width;
	int i=0;


	for(i=0;i<m*n; i++){
		matrix->h_rows[i] = val;
	}
}


int scalar_matrix_mult(float scalar_value, Matrix* matrix){
	unsigned long int m = matrix->height, n = matrix->width;
	
	if((m%8!=0)||(n%8!=0)){
		return 0;
	}


	int blockSize = threads_per_block;
	int numBlocks = (m*n + blockSize - 1) / blockSize;
	if (numBlocks > max_blocks_per_grid) numBlocks = max_blocks_per_grid;

	escalar<<<numBlocks, blockSize>>>(m*n, matrix->d_rows, scalar_value);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	return 1;
};


int matrix_matrix_mult(Matrix *matrixA, Matrix *matrixB, Matrix *matrixC){

	unsigned long int m = matrixA->height, q = matrixB->width, n = matrixA->width;


	if((m%8!=0)||(q%8!=0)||(n%8!=0)||(matrixA->width!=matrixB->height)){
		return 0;
	}

	int blockSize = threads_per_block;
	int numBlocks = (matrixC->width*matrixC->height + blockSize - 1) / blockSize;
	if (numBlocks > max_blocks_per_grid) numBlocks = max_blocks_per_grid;



	matrix_mult<<<numBlocks, blockSize>>>(matrixC->width*matrixC->height, matrixA->d_rows, matrixB->d_rows, matrixC->d_rows, matrixA->width, matrixB->width, matrixC->width);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();


	return 1;
}

void mostra_matrix(Matrix *matrix){

	unsigned long int m = matrix->height, n = matrix->width;
	int i=0,j=0;

	printf("[ ");
	for(i=0;i<m; i++){
		for(j=0; j<n; j++){
			printf(" %f ",matrix->h_rows[i*n + j]);
		}
		printf("\n");
	}
	printf("]\n");

}
