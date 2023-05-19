#include <iostream>
#include <vector>
#include <iterator>
#include <mpi.h>

using namespace std;

const int ROWS = 100;
const int COLS = 100;

void createField(bool* field) {
	memset(field, false, COLS * ROWS);
	field[0 * COLS + 1] = true;
	field[1 * COLS + 2] = true;
	field[2 * COLS + 0] = true;
	field[2 * COLS + 1] = true;
	field[2 * COLS + 2] = true;
}

int* countNumOfElemsForProc(int size) {
	int* result = new int[size];
	int linesCount = ROWS / size;
	int extraLinesCount = ROWS % size;

	for (int i = 0; i < size; ++i) {
		result[i] = linesCount * COLS;
		if (extraLinesCount > 0) {
			result[i] += COLS;
			--extraLinesCount;
		}
	}
	return result;
}

int* countNumOfLinesForProc(int* numOfElemsForProc, int size) {
	int* result = new int[size];
	for (int i = 0; i < size; ++i) {
		result[i] = numOfElemsForProc[i] / COLS;
	}
	return result;
}

int* createElemOffsetsVector(int* numOfElemsForProc, int size) {
	int* result = new int[size];
	int offset = 0;
	for (int i = 0; i < size; ++i) {
		result[i] = offset;
		offset += numOfElemsForProc[i];
	}
	return result;
}

int* createLinesOffsetsVector(int* elemOffsets, int size) {
	int* result = new int[size];
	for (int i = 0; i < size; ++i) {
		result[i] = elemOffsets[i] / ROWS;
	}
	return result;
}

bool isEqualFields(bool* last, bool* current, int countOfLines) {
	for (int i = COLS; i < COLS * (countOfLines + 1); ++i) {
		if (last[i] != current[i])
			return false;
	}
	return true;
}

void countStopVector(vector<bool*> evolution, bool* stopVector, bool* partOfFieldWithBoundaries, int countOfLines) {
	int vecSize = evolution.size() - 1; // длина вектора останова
	auto iterator = evolution.begin();
	for (int i = 0; i < vecSize; ++i) {
		stopVector[i] = isEqualFields(*iterator, partOfFieldWithBoundaries, countOfLines);
		++iterator;
	}
}

int countNumOfNeighbours(bool* field, int i, int j) {
	int result = field[i * COLS + (j + 1) % COLS] + 
		field[i * COLS + (j + COLS - 1) % COLS] + 
		field[(i + 1) * COLS + (j + 1) % COLS] + 
		field[(i + 1) * COLS + (j + COLS - 1) % COLS] + 
		field[(i - 1) * COLS + (j + 1) % COLS] + 
		field[(i - 1) * COLS + (j + COLS - 1) % COLS] + 
		field[(i + 1) * COLS + j] + 
		field[(i - 1) * COLS + j];

	return result;
}

void createNextGeneration(bool* currentField, bool* nextField, int countOfLines) {
	for (int i = 1; i < countOfLines - 1; ++i) {
		for (int j = 0; j < COLS; ++j) {
			int cellState = currentField[i * COLS + j];
			int numOfNeighbours = countNumOfNeighbours(currentField, i, j);

			if (cellState == 0 && numOfNeighbours == 3) {
				nextField[i * COLS + j] = 1;
			}
			else if (cellState == 1 && (numOfNeighbours < 2 || numOfNeighbours > 3)) {
				nextField[i * COLS + j] = 0;
			}
			else {
				nextField[i * COLS + j] = currentField[i * COLS + j] = cellState;
			}
		}
	}
}

bool checkEndOfGame(int size, int sizeOfVector, bool* matrixOfStopVectors) {
	for (int i = 0; i < sizeOfVector; ++i) {
		bool stop = true;
		for (int j = 0; j < size; ++j) {
			stop &= matrixOfStopVectors[j * sizeOfVector + j];
		}
		if (stop)
			return true;
	}
	return false;
}

void start(int size, int rank, bool* field) {
	int iteration = 0;
	int maxIterationsNum = 10000;

	bool isEnd = false;
	// считаем число элементов для каждого процесса
	int* numOfElemsForProc = countNumOfElemsForProc(size);
	// считаем число строк для каждого процесса
	int* numOfLinesForProc = countNumOfLinesForProc(numOfElemsForProc, size);
	// строим массив смещений по элементам для отправки сообщений между процессами
	int* elemOffsets = createElemOffsetsVector(numOfElemsForProc, size);
	// строим массив смещений по строкам для отправки сообщений между процессами
	int* linesOffsets = createLinesOffsetsVector(elemOffsets, size);
	
	bool* partOfFieldWithBoundaries = new bool[numOfElemsForProc[rank] + 2 * COLS];
	bool* partOfField = partOfFieldWithBoundaries + COLS;

	MPI_Scatterv(field, numOfElemsForProc, elemOffsets, MPI_C_BOOL, partOfField, numOfElemsForProc[rank], MPI_C_BOOL, 0, MPI_COMM_WORLD);

	int lastRank = (rank + size - 1) % size;
	int nextRank = (rank + 1) % size;
	vector<bool*> evolution;

	while (!isEnd || !(iteration > maxIterationsNum)) {
		++iteration;
		bool* nextPartOfFieldWithBoundaries = new bool[numOfElemsForProc[rank] + 2 * COLS];
		bool* nextPartOfField = nextPartOfFieldWithBoundaries + COLS;
		evolution.push_back(partOfFieldWithBoundaries);

		// инициализация отправки первой строки предыдущему ядру
		MPI_Request topLineSendReq;
		MPI_Isend(partOfField, COLS, MPI_C_BOOL, lastRank, 1, MPI_COMM_WORLD, &topLineSendReq);

		// инициализация отправки последней строки последующему ядру
		MPI_Request bottomLineSendReq;
		MPI_Isend(partOfField + numOfElemsForProc[rank] - COLS, COLS, MPI_C_BOOL, nextRank, 0, MPI_COMM_WORLD, &bottomLineSendReq);

		// инициирование получения от предыдущего ядра его последней строки
		MPI_Request bottomLineRecvReq;
		MPI_Irecv(partOfFieldWithBoundaries, COLS, MPI_C_BOOL, lastRank, 0, MPI_COMM_WORLD, &bottomLineRecvReq);

		// инициирование получения от последующего ядра его первой строки
		MPI_Request topLineRecvReq;
		MPI_Irecv(partOfField + numOfElemsForProc[rank], COLS, MPI_C_BOOL, nextRank, 1, MPI_COMM_WORLD, &topLineRecvReq);

		// Вычисление флага останова
		MPI_Request stopFlagsReq;
		int sizeOfVector = evolution.size() - 1; // по условию длина вектора останова на единицу меньше текущей итерации
		bool* stopVector = new bool[sizeOfVector]; // вектор останова
		bool* matrixOfStopVectors = new bool[sizeOfVector * size]; // матрица с векторами останова по которой проверяем конец итераций
		
		if (sizeOfVector > 1) {
			countStopVector(evolution, stopVector, partOfFieldWithBoundaries, numOfLinesForProc[rank]);
			// инициализация обмена векторами флагов останова со всеми ядрами
			MPI_Iallgather(stopVector, sizeOfVector, MPI_C_BOOL, matrixOfStopVectors, sizeOfVector, MPI_C_BOOL, MPI_COMM_WORLD, &stopFlagsReq);
		}

		// вычисление состояний клеток в строках, кроме первой и последней
		createNextGeneration(partOfField, nextPartOfField, numOfLinesForProc[rank]);

		// дожидаемся освобождения буфера отправки первой строки предыдущему ядру
		MPI_Status status;
		MPI_Wait(&topLineSendReq, &status);

		// дожидаемся получения отпредыдущего ядра его последней строки
		MPI_Wait(&bottomLineRecvReq, &status);

		// вычисляем состояние клеток в первой строке
		createNextGeneration(partOfFieldWithBoundaries, nextPartOfFieldWithBoundaries, 3);

		// дожидаемся освобождения буфера отправки последней строки последующему ядру
		MPI_Wait(&bottomLineSendReq, &status);

		// дожидаемся получения от последующего ядра его первой строки
		MPI_Wait(&topLineRecvReq, &status);

		// вычисляем состояния клеток в последней строке
		createNextGeneration(partOfField + (numOfLinesForProc[rank] - 2) * COLS, nextPartOfField + (numOfLinesForProc[rank] - 2) * COLS, 3);

		if (sizeOfVector > 1) {
			// дожидаемся завершения обмена векторами флагов останова со всеми ядрами
			MPI_Wait(&stopFlagsReq, &status);

			// сравиниваем вектора флагов останова, полученные от всех ядер, если для какой-то итерации все флаги равны единице, завершаем выполнение
			isEnd = checkEndOfGame(size, sizeOfVector, matrixOfStopVectors);
			delete[] stopVector;
			delete[] matrixOfStopVectors;
		}
		if (isEnd) {
			break;
		}
		partOfFieldWithBoundaries = nextPartOfFieldWithBoundaries;
		partOfField = nextPartOfField;
	}
	if (rank == 0) {
		cout << "Num of made iterations: " << iteration << endl;
	}
	for (auto matrix : evolution) {
		delete[] matrix;
	}
	evolution.clear();
	delete[] elemOffsets;
	delete[] numOfElemsForProc;
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	int size;
	int rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	bool* field = nullptr;
	if (rank == 0) {
		field = new bool[ROWS * COLS];
		createField(field);
	}
	double startTime = MPI_Wtime();
	start(size, rank, field);
	double endTime = MPI_Wtime();
	
	if (rank == 0) {
		cout << "Time spent: " << endTime - startTime << endl;
		delete[] field;
	}

	MPI_Finalize();

	return 0;
}
