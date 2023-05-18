#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

using namespace std;

int COLS = 10; // чсило столбцов в матрице
int ROWS = 10; // число строк в матрице
int maxIterationsNum = 1000; // ограничение на число итераций которые можем сделать

void printMatrix(bool* field) {
	for (int i = 0; i < ROWS; ++i) {
		for (int j = 0; j < COLS; ++j) {
			cout << field[i * ROWS + j] << " ";
		}
		cout << endl;
	}
}

void createField(bool* field) {
	memset(field, false, COLS * ROWS);
	field[2 * ROWS + 0] = true;
	field[2 * ROWS + 1] = true;
	field[2 * ROWS + 2] = true;
	field[1 * ROWS + 2] = true;
	field[0 * ROWS + 1] = true;
}

void setFieldPartsData(int* countOfSendedElements, int* offsets, int processesNum) {
	int offset = 0;
	for (int i = 0; i < processesNum; ++i) {
		int linesCount = ROWS / processesNum;
		if (i < ROWS % processesNum) {
			++linesCount;
		}

		offsets[i] = offset * COLS;
		offset += linesCount;
		countOfSendedElements[i] = linesCount * COLS;
	}
}

int main(int argc, char** argv) {
	int size;
	int rank;
	double startTime;
	double endTime;
	int countOfLinesForProcess;
	int countOfLinesForProcWithBoundaries;
	int iter;

	bool* field;
	int* countsOfSendedElements;
	int* offsets;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	countsOfSendedElements = new int[size];
	offsets = new int[size];

	setFieldPartsData(countsOfSendedElements, offsets, size);

	countOfLinesForProcess = countsOfSendedElements[rank] / COLS;
	countOfLinesForProcWithBoundaries = countOfLinesForProcess + 2;

	if (rank == 0) {
		field = new bool[COLS * ROWS];
		createField(field);
	}

	bool* current = new bool[countOfLinesForProcWithBoundaries * COLS];
	bool* next = new bool[countOfLinesForProcWithBoundaries * COLS];

	MPI_Scatterv(field, countsOfSendedElements, offsets, MPI_C_BOOL, (current + COLS), countsOfSendedElements[rank], MPI_C_BOOL, 0, MPI_COMM_WORLD);

	startTime = MPI_Wtime();

	for (iter = 0; iter < maxIterationsNum; ++iter) {
		MPI_Request requests[5];
		// MPI_Isend(current + X, X, )
	}

	endTime = MPI_Wtime();

	if (rank == 0) {
		cout << "Time spent: " << endTime - startTime << endl;
	}

	MPI_Finalize();
	return 0;
}
