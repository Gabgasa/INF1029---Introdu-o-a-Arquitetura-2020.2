#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"
#include <immintrin.h>
#include <pthread.h>

int num_threads = 1;



void set_number_thread(int n){
	printf("Number of threads: %d\n", n);
	num_threads = n;
}

void *escalar(void *t)
{
    int i;
    
	struct thread_data_scalar *my_data;
	my_data = (struct thread_data_scalar*)t;
	int tid = my_data->thread_id;
	int numLinhas = my_data->numLinhas;
	int numCol = my_data->numCol;
	float *start = my_data->matrixStart + (my_data->buffer_begin * numCol);
	float scalar_value = my_data->scalar_value;
	// printf("start: %f \n", start[0]);
	// printf("bufferBegin: %d \n", my_data->buffer_begin);
	// printf("numCol: %d \n", numCol);
	// printf("numLinha: %d \n", numLinhas);


	__m256 v1 = _mm256_set1_ps(scalar_value);
	__m256 v2;
	__m256 res;
	
	for(i=0;i<numLinhas*numCol; i+=8, start+=8){

		v2 = _mm256_load_ps(start);
		res	= _mm256_mul_ps(v2, v1);
		_mm256_store_ps(start, res);			
	}
	
    pthread_exit((void*) t);
}

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
	int rc;
	long i;
	
	void *status;
	struct thread_data_scalar thread_data_array[num_threads];
	unsigned long int buffer_chunk = m/num_threads;

	pthread_t tEscalar[num_threads];

	__m256 v1 = _mm256_set1_ps(scalar_value);
	__m256 v2;
	__m256 res;


	if((m%8!=0)||(n%8!=0)){
		return 0;
	}

	if(m%num_threads != 0){
		printf("ERRO: O numero de linhas nao eh multiplo do numero de threads.\n");
		return 0;
	}

	for(i = 0; i<num_threads; i++){
		thread_data_array[i].buffer_begin = i * buffer_chunk;
		thread_data_array[i].numLinhas = buffer_chunk;
		thread_data_array[i].numCol = n;
		thread_data_array[i].scalar_value = scalar_value;
		thread_data_array[i].thread_id = i;
		thread_data_array[i].matrixStart = matrix->rows;
		rc = pthread_create(&tEscalar[i], NULL, escalar, (void *) &thread_data_array[i]);
		if (rc){
			printf("ERROR; return code from pthread_create() is %d\n", rc);
	  		exit(-1);
		}

	}

	for(i = 0; i<num_threads; i++){
		rc = pthread_join(tEscalar[i], &status);

		if(rc){
			printf("ERROR; return code from pthread_join() is %d\n", rc);
      		exit(-1);
		}
	}	
	return 1;

};

void *matrix_mult(void *t){
	__m256 a;
	__m256 b;
	__m256 c;
	__m256 escalar_a_b;

	struct thread_data_matrix_mult *my_data;
	my_data = (struct thread_data_matrix_mult*)t;


	Matrix *matrixA = my_data->matrixA;
	Matrix *matrixB = my_data->matrixB;
	Matrix *matrixC = my_data->matrixC;

	
	float *nxtA =  matrixA->rows + (my_data->buffer_begin * matrixA->width);
	float *nxtB = matrixB->rows;
	float *nxtC = matrixC->rows + (my_data->buffer_begin * matrixC->width);
	float *iniC = nxtC;

	int numLinhas = my_data->numLinhas;

	//Linhas de A
	for(int i = 0; i < numLinhas; i++, nxtA+=8){
		//Selecionando a linha de C para ser a mesma de A
		nxtC = iniC+(i*matrixC->width);
		
		//Selecionando elementos de A
		for(int j = 0; j<matrixA->width; j++){
			a = _mm256_set1_ps(nxtA[j]);
			
			for(int k = 0; k < matrixB->width; k+=8, nxtB+=8, nxtC+=8){
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
			nxtC = iniC+(i*matrixB->width);
		}
		
		nxtB = matrixB->rows;		
    }

    pthread_exit((void*) t);

}

int matrix_matrix_mult(Matrix *matrixA, Matrix *matrixB, Matrix *matrixC){

	unsigned long int m = matrixA->height, q = matrixB->width, n = matrixA->width;

	int rc;
	long i;	
	void *status;
	struct thread_data_matrix_mult thread_data_array[num_threads];


	unsigned long int buffer_chunk = m/num_threads;

	pthread_t tMatrixMult[num_threads];

	if((m%8!=0)||(q%8!=0)||(n%8!=0)||(matrixA->width!=matrixB->height)){
		return 0;
	}
	if(m%num_threads != 0){
		printf("ERRO: O numero de linhas nao eh multiplo do numero de threads.\n");
		return 0;
	}

    for(i = 0; i<num_threads; i++){
		thread_data_array[i].buffer_begin = i * buffer_chunk;
		thread_data_array[i].thread_id = i;
		thread_data_array[i].numLinhas = buffer_chunk;
		thread_data_array[i].matrixA = matrixA;
		thread_data_array[i].matrixB = matrixB;
		thread_data_array[i].matrixC = matrixC;
		rc = pthread_create(&tMatrixMult[i], NULL, matrix_mult, (void *) &thread_data_array[i]);
		if (rc){
			printf("ERROR; return code from pthread_create() is %d\n", rc);
	  		exit(-1);
		}

	}

	for(i = 0; i<num_threads; i++){
		rc = pthread_join(tMatrixMult[i], &status);

		if(rc){
			printf("ERROR; return code from pthread_join() is %d\n", rc);
      		exit(-1);
		}
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
