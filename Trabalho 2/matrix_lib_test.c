#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"
#include <string.h>
#include "timer.h" 






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
			fwrite (&matrix.rows[i*matrix.width + j], sizeof(float), 1, outfile);
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
			fread (&matrix.rows[i*matrix.width + j], sizeof(float), 1, infile);
		}
	}
	fclose(infile);

}



int main(int argc, char *argv[]){

	struct timeval start, stop, overall_start, overall_stop;
	gettimeofday(&overall_start, NULL);

	float escalar = atof(argv[1]);
	unsigned long int linhasA = atoi(argv[2]);
	unsigned long int colunasA = atoi(argv[3]);
	unsigned long int linhasB = atoi(argv[4]);
	unsigned long int colunasB = atoi(argv[5]);
	char* nomeMatrizA = argv[6];
	char* nomeMatrizB = argv[7];
	char* result1 = argv[8];
	char* result2 = argv[9];

	
 	Matrix A, B, C, D, Dres;
	// Open matrixes for reading
	
	//Contem a matriz com 8 linhas e 16 colunas
	A.height = linhasA;
	A.width = colunasA;
	A.rows = (float*)aligned_alloc(32,A.height*A.width*sizeof(float));

	//Contem a matriz com 16 linhas e 8 colunas
	B.height = linhasB;
	B.width = colunasB;
	B.rows = (float*)aligned_alloc(32,B.height*B.width*sizeof(float));

	//Vai conter o resultado da multiplicacao de matrizes
	C.height = linhasA;
	C.width = colunasB;
	C.rows = (float*)aligned_alloc(32,C.height*C.width*sizeof(float));

	preenche_matrix(&C, 0);

	D.height = 1024;
	D.width = 1024;
	D.rows = (float*)aligned_alloc(32,D.height*D.width*sizeof(float));

	Dres.height = 1024;
	Dres.width = 1024;
	Dres.rows = (float*)aligned_alloc(32,Dres.height*Dres.width*sizeof(float));
	

	
	
	readMatrixFile(A, nomeMatrizA);
	readMatrixFile(B, nomeMatrizB);


	gettimeofday(&start, NULL);
	scalar_matrix_mult(escalar, &A);
	gettimeofday(&stop, NULL);
	printf("Scalar matrix mult: %f ms\n", timedifference_msec(start, stop));


	writeMatrixFile(A, result1);


	//mostra_matrix(&result2);
	gettimeofday(&start, NULL);	
	matrix_matrix_mult(&A, &B, &C);
	gettimeofday(&stop, NULL);
	printf("Matrix matrix mult: %f ms\n", timedifference_msec(start, stop));
	mostra_matrix(&C);

	//mostra_matrix(&C);
	
	writeMatrixFile(C, result2);

	
	//TESTE COM MATRIZ 1024X1024
	preenche_matrix(&D, 3);
	preenche_matrix(&Dres, 3);

	gettimeofday(&start, NULL);
	matrix_matrix_mult(&D, &D, &Dres);
	gettimeofday(&stop, NULL);
	printf("Time for matrix multi with 1024x1024 matrix: %f ms\n", timedifference_msec(start,stop));

	//FIM TESTE MATRIZ 1024X1024

	free(Dres.rows);
	free(D.rows);
	
	free(C.rows);
	free(B.rows);
	free(A.rows);

   	gettimeofday(&overall_stop, NULL);
   	printf("Overall time: %f ms\n", timedifference_msec(overall_start, overall_stop));
    return 0; 
} 
