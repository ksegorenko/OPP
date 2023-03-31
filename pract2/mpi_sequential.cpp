#include <iostream>
#include <cmath>
#include <time.h>

using namespace std;

const double eps = 0.000000001;
const double taul = 0.00001;
const int N = 6000;

void firstInit(double* A, double* b, double* x) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			A[i * N + j] = i == j ? 2 : 1; // утяжеляем диагональ двойками, остальные значения в матрице 1 
		}
		b[i] = N + 1;
		x[i] = 0; // при первой инициализации заподняем вектор x нулями
	}
}

void countVecMatrMult (double* matr, double* vec, double* result) {
	for (int i = 0; i < N; ++i) {
		result[i] = 0;
		for (int j = 0; j < N; ++j) {
			result[i] += matr[i * N + j] *  vec[j];
		}
	}
}

void countSubOfVectors(double* result, double* left, double* right) {
	for (int i = 0; i < N; ++i) {
		result[i] = left[i] - right[i];
	}
}

void countScalarMatrMult(double* result, double* vec) {
	for (int i = 0; i < N; ++i) {
		result[i] = vec[i] * taul;
	}
}

double countAbs(double* vec) {
	double abs = 0;
	for (int i = 0; i < N; ++i) {
		abs += pow(vec[i], 2);
	}
	return abs;
}

int main(int argc, char** argv[]) {

	MPI_Init(&argc, &argv);
	double startTime = MPI_Wtime();
//	time_t timeStart= time(NULL);

	double* A = new double[N * N];
	double* b = new double[N];
	double* x = new double[N];

	// создадим вектора для хранения промежуточных вычислений
	double* Ax = new double[N];
	double* subAx_b = new double[N];
	double* multTaulAx_b = new double[N];

	// инициализируем переменные для проверки конца итераций
	double absAx_b = 0, abs_b = 0;

	firstInit(A, b, x);

	countVecMatrMult(A, x, Ax);
	countSubOfVectors(subAx_b, Ax, b);
	absAx_b = countAbs(subAx_b);
	abs_b = countAbs(b);

	long double sqrEps = pow(eps, 2);
	double g_x = absAx_b / abs_b;
	int itearationsNum = 0;

	while (g_x >= sqrEps && itearationsNum < 10000) {
		countScalarMatrMult(multTaulAx_b, subAx_b);
		countSubOfVectors(x, x, multTaulAx_b);
		countVecMatrMult(A, x, Ax);
		countSubOfVectors(subAx_b, Ax, b);

		absAx_b = countAbs(subAx_b);
		g_x = absAx_b / abs_b;
		itearationsNum++;
	}
	double endTime = MPI_Wtime();
//	time_t timeEnd = time(NULL);

	for (int i = 0; i < N; ++i) {
		cout << x[i] << " ";
	}

	printf("Time spend: %lf", endTime - startTime);
//	double totalTime = timeEnd - timeStart;
//	printf("Time spend: %lf", totalTime);
	MPI_Finallize();

	delete[] A;
	delete[] x;
	delete[] Ax;
	delete[] b;
	delete[] subAx_b;
	delete[] multTaulAx_b;

	return 0;
}
