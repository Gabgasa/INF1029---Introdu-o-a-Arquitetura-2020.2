#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
extern "C" {
#include "timer.h"
}



void writeMatrixFile(Matrix matrix, char *name){
	FILE *outfile;
	outfile = fopen (name, "wb");
	if (outfile == NULL) 
    { 
        fprintf(stderr, "\nError openning write file 1\n"); 
        exit (1); 
    }
	


    fwrite (&matrix.width, sizeof(unsigned long int), 1, outfile);
	fwrite (&matrix.height, sizeof(unsigned long int), 1, outfile);
	//fwrite (&B, 2*sizeof(unsigned long int), 1, outfile2);  

	for(int i=0;i<matrix.height; i++){
		for(int j=0; j<matrix.width; j++){
			fwrite (&matrix.h_rows[i*matrix.width + j], sizeof(float), 1, outfile);
		}
	}
	
   	fclose(outfile);
}

void readMatrixFile(Matrix matrix, char *name){
	FILE *infile;
	infile = fopen (name, "rb"); 
    if (infile == NULL) 
    { 
        fprintf(stderr, "\nError opening read file 1\n"); 
        exit (1); 
    }

    fread(&matrix.width, sizeof(unsigned long int), 1, infile);
    fread(&matrix.height, sizeof(unsigned long int), 1, infile);
    for(int i=0;i<matrix.height; i++){
		for(int j=0; j<matrix.width; j++){
			fread (&matrix.h_rows[i*matrix.width + j], sizeof(float), 1, infile);
		}
	}
	fclose(infile);

}

int check_errors(Matrix *matrix, float scalar_value) {
  unsigned long int i;
  unsigned long int N;

  /* Check the numbers of the elements of the matrix */
  N = matrix->height * matrix->width;

  /* Check the integrity of the matrix */
  if (N == 0 || matrix->h_rows == NULL) return 0;

  /* Check for errors (all values should be equal to scalar_value) */
  float maxError = 0.0f;
  float diffError = 0.0f;
  for (i = 0; i < N; i++)
    maxError = (maxError > (diffError=fabs(matrix->h_rows[i]-scalar_value)))? maxError : diffError;
  printf("Max error: %f\n", maxError);

  return 1;
}



int main(int argc, char *argv[]){

	struct timeval start, stop, overall_start, overall_stop;
	cudaError_t cudaError;
	gettimeofday(&overall_start, NULL);
	int num_threads = atoi(argv[6]);
	int max_blocks = atoi(argv[7]);
	set_grid_size(num_threads, max_blocks);

	float escalar = atof(argv[1]);
	unsigned long int linhasA = atoi(argv[2]);
	unsigned long int colunasA = atoi(argv[3]);
	unsigned long int linhasB = atoi(argv[4]);
	unsigned long int colunasB = atoi(argv[5]);
	char* nomeMatrizA = argv[8];
	char* nomeMatrizB = argv[9];
	char* result1 = argv[10];
	char* result2 = argv[11];

	
 	Matrix A, B, C, D, Dres;
	// Open matrixes for reading
	
	//Contem a matriz com 8 linhas e 16 colunas
	A.height = linhasA;
	A.width = colunasA;
	A.h_rows = (float*)aligned_alloc(32,A.height*A.width*sizeof(float));

	cudaError = cudaMalloc(&A.d_rows, A.height*A.width*sizeof(float));

	// check cudaMalloc memory allocation
	if (cudaError != cudaSuccess) {
		printf("cudaMalloc A.d_rows returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
			return 1;
	}

	//Contem a matriz com 16 linhas e 8 colunas
	B.height = linhasB;
	B.width = colunasB;
	B.h_rows = (float*)aligned_alloc(32,B.height*B.width*sizeof(float));

	cudaError = cudaMalloc(&B.d_rows, B.height*B.width*sizeof(float));

	// check cudaMalloc memory allocation
	if (cudaError != cudaSuccess) {
		printf("cudaMalloc B.d_rows returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
			return 1;
	}

	//Vai conter o resultado da multiplicacao de matrizes
	C.height = linhasA;
	C.width = colunasB;
	C.h_rows = (float*)aligned_alloc(32,C.height*C.width*sizeof(float));

	cudaError = cudaMalloc(&C.d_rows, C.height*C.width*sizeof(float));

	// check cudaMalloc memory allocation
	if (cudaError != cudaSuccess) {
		printf("cudaMalloc C.d_rows returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
			return 1;
	}

	//preenche_matrix(&C, 0);

	D.height = 1024;
	D.width = 1024;
	D.h_rows = (float*)aligned_alloc(32,D.height*D.width*sizeof(float));

	cudaError = cudaMalloc(&D.d_rows, D.height*D.width*sizeof(float));

	// check cudaMalloc memory allocation
	if (cudaError != cudaSuccess) {
		printf("cudaMalloc D.d_rows returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
			return 1;
	}

	Dres.height = 1024;
	Dres.width = 1024;
	Dres.h_rows = (float*)aligned_alloc(32,Dres.height*Dres.width*sizeof(float));

	cudaError = cudaMalloc(&Dres.d_rows, Dres.height*Dres.width*sizeof(float));

	// check cudaMalloc memory allocation
	if (cudaError != cudaSuccess) {
		printf("cudaMalloc Dres.d_rows returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
			return 1;
	}
	

	
	
	readMatrixFile(A, nomeMatrizA);
	readMatrixFile(B, nomeMatrizB);

	cudaError = cudaMemcpy(A.d_rows, A.h_rows, A.height*A.width*sizeof(float), cudaMemcpyHostToDevice);

	if (cudaError != cudaSuccess) {
		printf("cudaMemcpy (A_.h_rows -> A_.d_rows) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
			return 1;
	}

	cudaError = cudaMemcpy(B.d_rows, B.h_rows, B.height*B.width*sizeof(float), cudaMemcpyHostToDevice);

	if (cudaError != cudaSuccess) {
		printf("cudaMemcpy (B_.h_rows -> B_.d_rows) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
			return 1;
	}

	


	gettimeofday(&start, NULL);
	//mostra_matrix(&A);
	scalar_matrix_mult(escalar, &A);
	//mostra_matrix(&A);
	gettimeofday(&stop, NULL);
	cudaError = cudaMemcpy(A.h_rows, A.d_rows, A.height*A.width*sizeof(float), cudaMemcpyDeviceToHost);

	if (cudaError != cudaSuccess) {
		printf("cudaMemcpy (A.d_rows -> A.h_rows) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
			return 1;
	}
	printf("Scalar matrix mult: %f ms\n", timedifference_msec(start, stop));

	printf("Checking matrixA scalar for errors...\n");
  	check_errors(&A, 20.0f);



	writeMatrixFile(A, result1);


	//mostra_matrix(&result2);
	gettimeofday(&start, NULL);	
	matrix_matrix_mult(&A, &B, &C);
	gettimeofday(&stop, NULL);
	printf("Matrix matrix mult: %f ms\n", timedifference_msec(start, stop));
	
	cudaError = cudaMemcpy(C.h_rows, C.d_rows, C.height*C.width*sizeof(float), cudaMemcpyDeviceToHost);

	if (cudaError != cudaSuccess) {
		printf("cudaMemcpy (C.d_rows -> C.h_rows) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
			return 1;
	}

	mostra_matrix(&C);
	printf("Checking matrixC for errors...\n");
  	check_errors(&C, 640.0f);

	//mostra_matrix(&C);
	
	writeMatrixFile(C, result2);

	
	// //TESTE COM MATRIZ 1024X1024
	preenche_matrix(&D, 3);
	preenche_matrix(&Dres, 3);

	cudaError = cudaMemcpy(D.d_rows, D.h_rows, D.height*D.width*sizeof(float), cudaMemcpyHostToDevice);

	if (cudaError != cudaSuccess) {
		printf("cudaMemcpy (D_.h_rows -> D_.d_rows) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
			return 1;
	}

	gettimeofday(&start, NULL);
	matrix_matrix_mult(&D, &D, &Dres);
	gettimeofday(&stop, NULL);
	printf("Checking matrixD multi 1024x1024 for errors...\n");
	cudaError = cudaMemcpy(Dres.h_rows, Dres.d_rows, Dres.height*Dres.width*sizeof(float), cudaMemcpyDeviceToHost);

	if (cudaError != cudaSuccess) {
		printf("cudaMemcpy (Dres.d_rows -> Dres.h_rows) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
			return 1;
	}
  	check_errors(&Dres, 9216.0f);
	printf("Time for matrix multi with 1024x1024 matrix: %f ms\n", timedifference_msec(start,stop));

	// //FIM TESTE MATRIZ 1024X1024

	free(Dres.h_rows);
	free(D.h_rows);
	
	free(C.h_rows);
	free(B.h_rows);
	free(A.h_rows);

	cudaFree(A.d_rows);
	cudaFree(B.d_rows);
	cudaFree(C.d_rows);
	cudaFree(D.d_rows);
	cudaFree(Dres.d_rows);


   	gettimeofday(&overall_stop, NULL);
   	printf("Overall time: %f ms\n", timedifference_msec(overall_start, overall_stop));
    return 0; 
} 
