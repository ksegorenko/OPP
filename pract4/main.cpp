#include <stdio.h>
#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <mpi.h>

#define N1 4
#define N2 2
#define N3 4

#define X_DIMS 2 // размерность декартовой топологии по оси x
#define Y_DIMS 2 // размерность декартовой топологии по оси y

#define COORDS_MSG_TAG 1 // тэг сообщений с данными о декартовых координатах процесса
#define PART_B_MSG_TAG 2 // тэг сообщений с кусочками матрицы B
#define MATR_C_MSG_TAG 3 // тэг сообщений с кусочками матрицы C

void createMatrix(double* matrix, int n1, int n2) {
    for (size_t i = 0; i < n1; ++i) {
        for (size_t j = 0; j < n2; ++j) {
            matrix[i * n2 + j] = 1;
        }
    }
}

void CreateFirstMatrixBPart(double* partOfB, double* B, int n2, int n3) {
    for (int i = 0; i < n2; ++i) { // итерация по строкам
        for (int j = 0; j < n3 / X_DIMS; ++j) { // итерация по блокам (по частям матрицы B)
            partOfB[i * n3 / X_DIMS + j] = B[i * n3 + j]; // заполняем partOfB соответствующими элементами из матрицы B
        }
    }
}

void MultCreateCMatrixPart(double* partOfA, double* partOfB, double* partOfC, int n1, int n2, int n3) {
    for (int i = 0; i < n1; ++i) {
        double* ba = partOfC + i * n3;
        for (int j = 0; j < n3; ++j)
            ba[j] = 0;
        for (int k = 0; k < n2; ++k) {
            const double* b = partOfB + k * n3;
            double a = partOfA[i * n2 + k];
            for (int j = 0; j < n3; ++j) {
                ba[j] += a * b[j];
            }
        }
    }
}

void PutFirstPartOfMatrixC(double* C, double* partOfC, int n1, int n2) {
    for (int i = 0; i < n1 / Y_DIMS; ++i) {
        for (int j = 0; j < n2 / X_DIMS; ++j) {
            C[i * n2 + j] = partOfC[i * n2 / X_DIMS + j];
        }
    }
}

int main(int argc, char** argv) {
    int size = 0;// общее сило процессов в коммуникаторе MPI_COMM_WORLD
    int rank = 0;// номер конкретного процесса в коммуникаторе MPI_COMM_WORLD
    int coordsOfProcesses[X_DIMS * Y_DIMS] = { 0 };// массив с декартовыми координатами процессов
    double startTime;
    double endTime;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size); // находим общее число процессов в коммуникаторе MPI_COMM_WORLD 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // находим номер каждого процесса в коммуникаторе MPI_COMM_WORLD

    double* A = new double[N1 * N2]; 
    double* B = new double[N2 * N3];
    double* C = new double[N1 * N3];

    double* partOfA = new double[(N1 * N2) / Y_DIMS];
    double* partOfB = new double[(N2 * N3) / X_DIMS];
    double* partOfC = new double[(N1 * N3) / X_DIMS * Y_DIMS];

    if (rank == 0) {
        createMatrix(A, N1, N2);
        createMatrix(B, N2, N3);
        startTime = MPI_Wtime();
    }

    MPI_Comm CartTopology;
    int dims[2] = { X_DIMS, Y_DIMS }; //массив с размерностью нашей топологии, который мы подадим в качестве аргумента функции
    int periods[2] = { 0, 0 }; // массив флагов, задающих периодические граничные условия для каждого измерения, тк у нас никаких условий нет, заполняем его нулями
    int coords[2] = { 0 , 0 }; // массив в котором будем хранить координаты процесса в декартовой топологии

    MPI_Dims_create(size, 2, dims); // создание коммуникатора по заданным размерам решетки
    /*  1. nnodes(size) - общее количество процессов, которые нужно разместить на процессорной сетке.
        2. ndims(тк у нас декартова топология их 2) - количество измерений (размерность) процессорной сетки.
        3. dims - массив целых чисел, который будет заполнен размерами процессорной сетки в каждом измерении. 
    */
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &CartTopology); // создание декартовой топологии (решетки процессов)
    /*  1. comm_old(MPI_COMM_WORLD) - существующий коммуникатор, в котором нужно создать топологию сетки.
        2. ndims(для декартовой топологии 2) - количество измерений (размерность) сетки.
        3. dims - массив целых чисел, содержащий количество процессов в каждом измерении.
        4. periods - массив флагов, задающих периодические граничные условия для каждого измерения.
        5. reorder - флаг, указывающий на возможность переупорядочивания процессов в новом коммуникаторе.
        6. ссылка на новый коммуникатор, который мы создаем
    */
    MPI_Comm_rank(CartTopology, &rank); // получаем ранг нашего процесса в новом коммуникаторе CartTopology
    MPI_Cart_coords(CartTopology, rank, 2, coords); // определение декартовых координат процесса из его rank в коммуникаторе CartTopology
    /*  1. comm(CartTopology) — коммуникатор с топологией решетки
        2. rank — ранг процесса, для которого определяются декартовы координаты;
        3. ndims(2) — размерность решетки;
        4. coords — возвращаемые функцией декартовы координаты процесса.
    */

    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            MPI_Recv(coords, 2, MPI_INT, 0, COORDS_MSG_TAG, CartTopology, MPI_STATUS_IGNORE); // нулевой процесс получает от других процессов их декартовы координаты
            coordsOfProcesses[i] = coords[0] * 10 + coords[1]; // декартовы координаты каждого процесса кроме нулевого преобразуются в двузначное число и кладутся в общий массив
        }
        // зануляем значения для следующего процесса
        coords[0] = 0;
        coords[1] = 0;
    }
    else {
        MPI_Send(coords, 2, MPI_INT, 0, COORDS_MSG_TAG, CartTopology); // все отправляют нулевому процессу свои декартовы координаты
    }

    int subDims[2] = { 0, 0 }; // массив для размерности подрешеток
    MPI_Comm RowComm;
    subDims[1] = 0;
    subDims[0] = 1;
    MPI_Cart_sub(CartTopology, subDims, &RowComm); // разбиение решетки на подрешетки меньшей размерности (разбиение на строки)
   /* 1. comm(CartTopology) — исходный коммуникатор с топологией решетки
   *  2. subDims — массив для указания, какие измерения должны остаться в создаваемой подрешетке;
   *  3. newcomm(RowComm) — создаваемый коммуникатор с подрешеткой
   */

    MPI_Comm ColumnComm;
    subDims[1] = 1;
    subDims[0] = 0;
    MPI_Cart_sub(CartTopology, subDims, &ColumnComm); // разбиение решетки на подрешетки меньшей размерности (разбиение на столбцы)
    /* 1. comm(CartTopology) — исходный коммуникатор с топологией решетки
    *  2. subDims — массив для указания, какие измерения должны остаться в создаваемой подрешетке;
    *  3. newcomm(RowComm) — создаваемый коммуникатор с подрешеткой
    */

    if (coords[0] == 0) {
        //разрезаем матрицу А по строкам и распределяем между процессами первого столбца
         MPI_Scatter(A, (N1 * N2) / Y_DIMS, MPI_DOUBLE, partOfA, (N1 * N2) / Y_DIMS, MPI_DOUBLE, 0, ColumnComm);
         /* 1. A - дескриптор буфера, который содержит данные, отправляемые нулевым процессом
         *  2. (N1 * N2) / Y_DIMS - количество элементов в буфере отправки
         *  3. MPI_DOUBLE - тип данных каждого элемента в буфере
         *  4. partOfA - дескриптор буфера, содержащий данные, полученные в каждом процессе
         *  5. (N1 * N2) / Y_DIMS - количество элементов в буфере получения
         *  6. MPI_DOUBLE - тип элементов в буфере получения
         *  7. 0 - номер процесса-отправителя
         *  8. коммуникатор в котором происходит отправка и получение
         */
    }
    // рассылаем полученный кусок матрицы A всем процессам строки
    MPI_Bcast(partOfA, (N1 * N2) / Y_DIMS, MPI_DOUBLE, 0, RowComm); // в качестве отправителя указываем нулевой процесс подрешетки RowComm - производного коммуникатора

    MPI_Datatype MATRIX_B_COLUMN; // создаем производный тип данных для работы со столбцами матрицы B
    MPI_Type_vector(N2, N3 / X_DIMS, N3, MPI_DOUBLE, &MATRIX_B_COLUMN); // создает новый тип данных посредством дублирования существующего, при этом позволяет учитывать промежутки в смещении
    /* 1. N2 - входная перемнная, определяющая число блоков
    *  2. N3 / X_DIMS - входная переменная, определяющая число элементов в каждом болке
    *  3. N3 - входная переменная, определяющая смещение(число элементов междк началом последовательных блоков)
    *  4. MPI_DOUBLE - тип элементов, на основе которого создаем свой производный тип
    *  5. &MATRIX_B_COLUMN - имя типа ссылка на созданный производный тип данных
    */
    MPI_Type_commit(&MATRIX_B_COLUMN); // созданный тип должен быть размещен в системе перед использованием, размещаем его

    // нулевой процесс разделяет матрицу B между процессами первой строки решетки
    if (rank == 0) {
        // берем кусок матрицы B для нулевого процесса
        CreateFirstMatrixBPart(partOfB, B, N2, N3);
        // формируем куски матрицы B для остальных процессов, нулевой процесс рассылает их с помощью Send()
        for (int i = 1; i < X_DIMS; ++i) {
            MPI_Send(B + i * (N3 / X_DIMS), 1, MATRIX_B_COLUMN, i, PART_B_MSG_TAG, RowComm);
            /* 1. B + i * (N3 / X_DIMS) - указатель на буфер, содержащий отправляемые данные
            *  2. 1 - количество элементов в буфере
            *  3. MATRIX_B_COLUMN - тип данных элементов в буфере (используем созданный нами производный тип данных для блоков матрицы B)
            *  4. i - принимающий процесс
            *  5. PART_B_MSG_TAG - тэг сообщения для коммуникации
            *  6. RowComm - дескриптор коммуникатора, в котором происходит коммуникация
            */
        }
    }

    // процессы из первой строки декартовой топологии принимают свои куски матрицы B
    if (coords[1] == 0 && rank != 0) {
        MPI_Recv(partOfB, N2 * (N3 / X_DIMS), MPI_DOUBLE, 0, PART_B_MSG_TAG, RowComm, MPI_STATUS_IGNORE); // принимаем уже типом данных MPI_DOUBLE
    }
    // рассылаем полученный кусок матрицы B всем процессам столбца
    MPI_Bcast(partOfB, (N2 * N3) / X_DIMS, MPI_DOUBLE, 0, ColumnComm);

    // умножаем нужные части матриц в процессах и получаем кусок матрицы C
    MultCreateCMatrixPart(partOfA, partOfB, partOfC, N1 / Y_DIMS, N2, N3 / X_DIMS);

    MPI_Datatype PART_OF_C;
    MPI_Type_vector(N1 / Y_DIMS, N3 / X_DIMS, N3, MPI_DOUBLE, &PART_OF_C); // создаем производный тип данных для передачи части C от каждого процесса нулевому
    /* 1. N1 / Y_DIMS - входная перемнная, определяющая число блоков
    *  2. N3 / X_DIMS - входная переменная, определяющая число элементов в каждом болке
    *  3. N3 - входная переменная, определяющая смещение(число элементов междк началом последовательных блоков)
    *  4. MPI_DOUBLE - тип элементов, на основе которого создаем свой производный тип
    *  5. &PART_OF_C - имя типа ссылка на созданный производный тип данных
    */
    MPI_Type_commit(&PART_OF_C); //  созданный тип должен быть размещен в системе перед использованием, размещаем его

    //собираем матрицу C по частям от каждого процесса

    if (rank != 0) { // ненулевые процессы отправляют нулевому свои части матрицы C с помощью Send()
        MPI_Send(partOfC, (N1 * N3) / (X_DIMS * Y_DIMS), MPI_DOUBLE, 0, MATR_C_MSG_TAG, CartTopology);
    }
    else {
        for (int i = 1; i < size; ++i) {
            coords[0] = coordsOfProcesses[i] / 10; // вычисляем декартову координату x для конкретного процесса
            coords[1] = coordsOfProcesses[i] - coords[0] * 10; // вычисляем декартову координату y для конкретного процесса
            // принимаем кусок матрицы C с помощью созданного нами ранее типа PART_OF_C в соответствии с вычисленными координатами процесса
            MPI_Recv(C + coords[0] * (N3 / X_DIMS) + coords[1] * (N3 * (N1 / Y_DIMS)), 1, PART_OF_C, i, MATR_C_MSG_TAG, CartTopology, MPI_STATUS_IGNORE);
        }
        PutFirstPartOfMatrixC(C, partOfC, N1, N3); // нулевой процесс ставит свой кусок матрицы C на место
    }

    if (rank == 0) { // нулевой процесс считает и выводит время
        endTime = MPI_Wtime();
        printf("Time: %f\n", (endTime - startTime));
        for (int i = 0; i < N1; ++i) {
            for (int j = 0; j < N3; ++j) {
                printf("%lf ", C[i * N1 + j]);
            }
            printf("\n");
        }
    }

    // освобождаем выделенную память
    if (rank == 0) {
        free(A);
        free(B);
        free(C);
    }
    free(partOfA);
    free(partOfB);
    free(partOfC);

    MPI_Type_free(&MATRIX_B_COLUMN);
    MPI_Type_free(&PART_OF_C);
    MPI_Finalize();

    return 0;
}

