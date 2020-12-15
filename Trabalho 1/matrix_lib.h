
typedef struct matrix{
	unsigned long int height;
	unsigned long int width;
	float *rows;
} Matrix;

void preenche_matrix(Matrix *matrix);

int scalar_matrix_mult(float scalar_value, Matrix *matrix);

int matrix_matrix_mult(Matrix *matrixA, Matrix *matrixB, Matrix *matrixC);

void mostra_matrix(Matrix *matrix);

