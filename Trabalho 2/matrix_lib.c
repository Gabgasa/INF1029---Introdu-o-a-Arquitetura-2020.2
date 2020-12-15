#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"
#include <immintrin.h>


void preenche_matrix(Matrix *matrix, float val){
	unsigned long int m = matrix->height, n = matrix->width;
	int i=0,j=0;
	__m256 v = _mm256_set1_ps(val);
	for(i=0;i<m*n; i += 8){
		_mm256_store_ps(&matrix->rows[i], v);
	}
}


int scalar_matrix_mult(float scalar_value, Matrix* matrix){
	unsigned long int m = matrix->height, n = matrix->width;
	int i=0,j=0;
	__m256 v1 = _mm256_set1_ps(scalar_value);
	__m256 v2;
	__m256 res;
	if((m%8!=0)||(n%8!=0)){
		return 0;
	}

	for(i=0;i<m*n; i+=8){
		v2 = _mm256_load_ps(&matrix->rows[i]);
		res	= _mm256_mul_ps(v2, v1);
		_mm256_store_ps(&matrix->rows[i], res);			
	}
	return 1;

};

// int matrix_matrix_mult(Matrix *matrixA, Matrix *matrixB, Matrix *matrixC){
// 	unsigned long int m = matrixA->height, q = matrixB->width, n = matrixA->width;
// 	float multiplicacao;
// 	if((m%8!=0)||(q%8!=0)||(n%8!=0)||(matrixA->width!=matrixB->height)){
// 		return 0;
// 	}

//     for(int i=0; i<m; i++){
//         for(int k=0;k<q;k++){
//             for(int j=0; j<n; j++){
//             	multiplicacao = matrixA->rows[i*matrixA->width + j] * matrixB->rows[j*matrixB->width + k];
//             	matrixC->rows[i*matrixC->width + k] += multiplicacao;                            
//             }
//         }
        
//     }

//     return 1;
// }

int matrix_matrix_mult(Matrix *matrixA, Matrix *matrixB, Matrix *matrixC){

	unsigned long int m = matrixA->height, q = matrixB->width, n = matrixA->width;
	__m256 a;
	__m256 b;
	__m256 c;
	__m256 escalar_a_b;

	if((m%8!=0)||(q%8!=0)||(n%8!=0)||(matrixA->width!=matrixB->height)){
		return 0;
	}
	
	 float *nxtA = matrixA->rows;
	 float *nxtB = matrixB->rows;
	 float *nxtC = matrixC->rows;
    
	//Linhas de A
	for(int i = 0; i < m; i++, nxtA+=8){
		//Selecionando a linha de C para ser a mesma de A
		nxtC = matrixC->rows+(i*q);
		
		//Selecionando elementos de A
		for(int j = 0; j<n; j++){
			a = _mm256_set1_ps(nxtA[j]);
			
			for(int k = 0; k < q; k+=8, nxtB+=8, nxtC+=8){
				if(j==0){
					//Zerando a linha de C
					c = _mm256_set1_ps(0);
				}
				else
				{
					c = _mm256_load_ps(nxtC);
				}
				
				b = _mm256_load_ps(nxtB);
				
				escalar_a_b = _mm256_fmadd_ps(a, b, c);
				_mm256_store_ps(nxtC, escalar_a_b);
			}
			nxtC = matrixC->rows+(i*q);
		}
		
		nxtB = matrixB->rows;
		
		
    }

    return 1;
}

void mostra_matrix(Matrix *matrix){

	unsigned long int m = matrix->height, n = matrix->width;
	int i=0,j=0;

	printf("[ ");
	for(i=0;i<m; i++){
		for(j=0; j<n; j++){
			printf(" %f ",matrix->rows[i*n + j]);
		}
		printf("\n");
	}
	printf("]\n");

}
