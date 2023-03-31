#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

const int N = 7000;
const double eps = 0.00001;
const int maxIterationsCount = 5000;
const double taul = 0.00001;

void createMatrixPartForProcesses(int* countOfLinesForProcess, int* offsetArray, int* countOfMatrElem_s, int* countOfSkipedElem, int N, int size) {
    int currentOffset = 0;
    for (int i = 0; i < size; ++i) {
        countOfLinesForProcess[i] = N / size; // Гарантированно каждый получит число строк матрицы, равное целой части N / size

        //Докидываем процессам от 1 до N % size по одной строке для равномерного распределения
        //Если конечно же число N не кратно size
        if (i < N % size) {
            ++countOfLinesForProcess[i];
        }

        //Начало каждого процесcа i содержится в offsets[i]
        offsetArray[i] = currentOffset;
        currentOffset += countOfLinesForProcess[i];
        countOfMatrElem_s[i] = countOfLinesForProcess[i] * N;
        countOfSkipedElem[i] = offsetArray[i] * N;
    }
}

double countAbsSqr(const double* vec, const int N) {
    double absSqr = 0.0;
    for (int i = 0; i < N; ++i) {
        absSqr += pow(vec[i], 2);
    }
    return absSqr;
}

void firstInit(double* A, double* b, double* x) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = 1.0;
        }
        A[i * N + i] = 2.0;
        x[i] = 0.0;
        b[i] = N + 1;
    }
}

void printMatrix(const double* matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    double startTime = MPI_Wtime();

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int* countLinesForProc = new int[size];
    int* offsetArray = new int[size];
    int* countOfMatrElem_s = new int[size];
    int* countOfSkipedElem = new int[size];
    createMatrixPartForProcesses(countLinesForProc, offsetArray, countOfMatrElem_s, countOfSkipedElem, N, size);

    double* partOfA = new double[countLinesForProc[rank] * N];
    double* A = new double[N * N];
    double* x = new double[N];
    double* b = new double[N];

    double g_x = 0.0;
    double abs_b = 0.0;
    if (rank == 0) {
        firstInit(A, b, x);
        g_x = 1;
        //abs_b = sqrt(countAbsSqr(b, N));
    }

    // рассылаем вектор x, b, g(x)
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&g_x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    abs_b = sqrt(countAbsSqr(b, N));
    /*
    здесь &g_x - адрес начала буфера для приема сообщения
    1 - число элементов принимаемого массива
    MPI_DOUBLE - тип элемента принимающего массива
    0 - номер процесса-отправителя
    MPI_COMM_WORLD - коммуникатор
    */
    // разрезаем матрицу А на части
    MPI_Scatterv(A, countOfMatrElem_s, countOfSkipedElem, MPI_DOUBLE, partOfA, countOfMatrElem_s[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /*
    здесь A - адрес с которого начинается пересылаемый участок информации
    countOfMatrElem_s - адрес массива размеров отправляемых сообщений
    countOfSkipedElem - адрес массива смещений отправляемых сообщений
    MPI_DOUBLE - тип элемента данных отправляемого сообщения
    partOfA - адрес начала буфера для приема сообщения
    countOfMatrElem_s[rank] - число элементов принимаемого сообщения
    MPI_DOUBLE - тип элемента данных принимаемого сообщения
    0 - номер процесса-распылителя
    MPI_COMM_WORLD - коммуникатор
    */

    double* Ax_b = new double[countLinesForProc[rank]];
    double* partOfX = new double[countLinesForProc[rank]];

    double absOfSum = 0;
    int iterationCount = 0;

    while (g_x > eps && iterationCount < maxIterationsCount) {
        // заполняем Ax-b для каждого процесса по частям
        for (int i = 0; i < countLinesForProc[rank]; ++i) {
            Ax_b[i] = 0;
            for (int j = 0; j < N; ++j) {
                Ax_b[i] += partOfA[i * N + j] * x[j];
            }
            Ax_b[i] = Ax_b[i] - b[offsetArray[rank] + i];
        }
        // заполняем x для каждого процесса по частям
        for (int i = 0; i < countLinesForProc[rank]; ++i) {
            partOfX[i] = x[offsetArray[rank] + i] - taul * Ax_b[i];
        }

        // собираем по кусочкам вектор x
        MPI_Allgatherv(partOfX, countLinesForProc[rank], MPI_DOUBLE, x, countLinesForProc, offsetArray, MPI_DOUBLE, MPI_COMM_WORLD);
        /*
        здесь partOfX - адрес с которого начинается пересылаемый участок информации
        countLinesForProc[rank] - число элементов отправляемого сообщения
        MPI_DOUBLE - тип элементов отправляемого сообщения
        x - адрес начала буфера для приема сообщения
        countLinesForProc - адрес массива размеров принимаемых сообщений
        offsetArray - адрес массива смещений принимаемых сообщений
        MPI_DOUBLE - тип элемента данных принимаемых сообщений
        MPI_COMM_WORLD - коммуникатор
        */

        // считаем норму по частям, 
        double partAbs = countAbsSqr(Ax_b, countLinesForProc[rank]);
        //не будем отсылать нулевому процессу absOfSum а сделаем так чтобы каждый процесс сам считал g_x
       MPI_Allreduce(&partAbs, &g_x, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        /*
        здесь &partAbs - адрес с которого начинается пересылаемый участок информации
        &absOfSum - адрес начала буфера для приема сообщения
        1 - кол-во элементов отправляемого сообщения
        MPI_DOUBLE - тип элемента данных пересылаемого сообщения
        MPI_SUM - операция суммы
        0 - номер принимающего процесса
        MPI_COMM_WORLD - коммуникатор
        */
        if (rank == 0) {
            iterationCount++;
        }
        // после каждой итерации Bcast рассылает обновленные значения переменных
        MPI_Bcast(&iterationCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
        g_x = sqrt(g_x) / abs_b;
    }

    if (rank == 0) {
        if (iterationCount < maxIterationsCount) {
            double endTime = MPI_Wtime();
            printf("Time taken %f\n", endTime - startTime);

        }
        else {
            //Нужно поменять знак, если метод не сошелся с заданным таул
            printf("No converge, change taul sign\n");
        }
        for (int i = 0; i < N; ++i) {
            printf("%f ", x[i]);
        }
    }

    delete[] countLinesForProc;
    delete[] offsetArray;
    delete[] x;
    delete[] b;
    delete[] A;
    delete[] partOfA;
    delete[] Ax_b;
    delete[] partOfX;

    MPI_Finalize();

    return 0;
}
