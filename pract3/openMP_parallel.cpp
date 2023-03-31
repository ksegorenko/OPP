#include <iostream>
#include <cmath>
#include <time.h>
#include <omp.h>

using namespace std;

const double eps = 0.00001;
const double taul = 0.00001;
const int N = 7000;

void firstInit(double* A, double* b, double* x) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			A[i * N + j] = i == j ? 2 : 1;
		}
		b[i] = N + 1;
		x[i] = 0;
	}
}

void countVecMatrMult(double* matr, double* vec, double* result) {
#pragma omp for schedule(auto)
	for (int i = 0; i < N; ++i) {
		result[i] = 0;
		for (int j = 0; j < N; ++j) {
			result[i] += matr[i * N + j] * vec[j];
		}
	}
}

void countSubOfVectors(double* result, double* left, double* right) {
#pragma omp for schedule(auto)
	for (int i = 0; i < N; ++i) {
		result[i] = left[i] - right[i];
	}
}

void countScalarMatrMult(double* result, double* vec) {
#pragma omp for schedule(auto)
	for (int i = 0; i < N; ++i) {
		result[i] = vec[i] * taul;
	}
}

int main(int argc, char** argv) {
	double* A = new double[N * N];
	double* b = new double[N];
	double* x = new double[N];
	double* Ax = new double[N];
	double* subAx_b = new double[N];
	double* multTaulAx_b = new double[N];

	double absAx_b = 0;
	double abs_b = 0;
	double startTime = 0;
	double endTime = 0;
	long double sqrEps = pow(eps, 2);
	int itearationsNum = 0;
	double g_x = 0.0;


	firstInit(A, b, x);
	startTime = omp_get_wtime();
	omp_set_num_threads(1);
#pragma omp parallel
	{
		countVecMatrMult(A, x, Ax);
		countSubOfVectors(subAx_b, Ax, b);
#pragma omp for schedule(auto) reduction(+ : abs_b)
		for (int i = 0; i < N; i++) {
			abs_b += pow(b[i], 2);
		}
#pragma omp for schedule(auto) //reduction(+ : absAx_b)
		for (int i = 0; i < N; i++) {
#pragma omp atomic
			absAx_b += pow(subAx_b[i], 2);
		}
		
#pragma omp single
		{
			g_x = absAx_b / abs_b;
		}

		while (g_x >= sqrEps && itearationsNum < 10000) {
			countScalarMatrMult(multTaulAx_b, subAx_b);
			countSubOfVectors(x, x, multTaulAx_b);
			countVecMatrMult(A, x, Ax);
			countSubOfVectors(subAx_b, Ax, b);

#pragma omp single
			{
				absAx_b = 0;
			}
#pragma omp for schedule(auto) reduction(+:absAx_b)
			for (int i = 0; i < N; i++) {
				absAx_b += pow(subAx_b[i], 2);
			}
#pragma omp single
			{
				g_x = absAx_b / abs_b;
				itearationsNum++;
			}
		}

	}
	endTime = omp_get_wtime();
	std::cout << "Time spent: " << endTime - startTime << std::endl;

	delete[] A;
	delete[] x;
	delete[] Ax;
	delete[] b;
	delete[] subAx_b;
	delete[] multTaulAx_b;

	return 0;
}
