typedef struct matrix{
	unsigned long int height;
	unsigned long int width;
	float *rows;
} Matrix;

struct thread_data_scalar {
	int thread_id;
	float scalar_value;
	int numLinhas;
	int numCol;
	float *matrixStart;
	long unsigned int buffer_begin;

};

struct thread_data_matrix_mult {
	int thread_id;
	int numLinhas;
	Matrix *matrixA;
	Matrix *matrixB;
	Matrix *matrixC;
	long unsigned int buffer_begin;

};

void preenche_matrix(Matrix *matrix, float val);

int scalar_matrix_mult(float scalar_value, Matrix *matrix);

int matrix_matrix_mult(Matrix *matrixA, Matrix *matrixB, Matrix *matrixC);

void mostra_matrix(Matrix *matrix);

void set_number_thread(int n);
