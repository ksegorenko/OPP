#include <iostream>
#include <mpi.h>
#include <pthread.h>
#include <cmath>
#include <deque>
#include <cstdlib>

using namespace std;

#define TASKS_PER_PROC 500
deque<int> tasks;         // shared data

bool tasksFinished;       // shared data
pthread_mutex_t mutex;

int iterCounter = 0;
double globalRes = 0;
int executedTasks = 0;

int countWeight(int size, int rank, int idx, int L) {
    int res = abs(50 - (idx % TASKS_PER_PROC)) * abs(rank - (iterCounter % size)) * L; // по условию задания вычисляем нагрузку задач
    return res;
}

void* workerFunc(void* attrs) {
    int task_weight;
    bool running = true;
    while (running) {
        // Receive task
        pthread_mutex_lock(&mutex);
        if (tasks.empty()) { // задач в очереди больше нет
            task_weight = -1;
        }
        else {
            task_weight = tasks.front(); // берем первую задачу из очереди 
            tasks.pop_front(); // удаляем ее из очереди
        }
        pthread_mutex_unlock(&mutex);

        // если еще есть задачи вычисляем их
        if (task_weight != -1) {
            double tempRes = 0;
            for (int i = 0; i < task_weight; ++i) {
                tempRes += sin(i);
            }
            globalRes += tempRes;
            executedTasks++;
        }
        // проверка на то что все задачи в очереди выполнены
        pthread_mutex_lock(&mutex);
        if (tasksFinished) {
            running = false;
        }
        pthread_mutex_unlock(&mutex);
    }
}

int DecideProcessRoot(int rank, bool processDidAllTasks) {
    int root;
    int challenger = -1;   // если не хочет быть root'ом то значение -1
    if (processDidAllTasks) {
        challenger = rank; // если хочет быть root'ом то значение его ранг
    }
    MPI_Allreduce(&challenger, &root, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD); // выбираем максимальный ранг и делаем его рутом
    return root;
}

int DecideRootPretendCount(bool processDidAllTasks) {
    int result;
    int wantBeRootNum = (int)processDidAllTasks;
    MPI_Allreduce(&wantBeRootNum, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); // складываем в result количество процессов которые выполнили свои задачи и хотят получить больше задач
    return result;
}

void delegateTask(int size, int rank, int workerRank, int extraTaskWeight) {
    int extraTasks[size];
    MPI_Gather(&extraTaskWeight, 1, MPI_INT, extraTasks, 1, MPI_INT, workerRank, MPI_COMM_WORLD); // собираем от всех процессов их extra tasks

    if (rank == workerRank) {
        pthread_mutex_lock(&mutex);
        for (int i = 0; i < size; ++i) {
            if (extraTasks[i] != -1) {
                tasks.push_back(extraTasks[i]);
            }
        }
        pthread_mutex_unlock(&mutex);
    }
}

void* ManagerTask(void* attrs) {
    int size = ((int*)attrs)[0];
    int rank = ((int*)attrs)[1];
    int currentCountOfTasksInDeque;
    int rootProcessesCount = 0;

    // выполняем цикл пока все процессы не выполнят все задачи
    while (rootProcessesCount != size) {
        pthread_mutex_lock(&mutex);
        currentCountOfTasksInDeque = (int)tasks.size();
        pthread_mutex_unlock(&mutex);

        bool processDidAllTasks = (bool)(currentCountOfTasksInDeque == 0);
        int actualRootProcess = DecideProcessRoot(rank, processDidAllTasks);

        if (actualRootProcess != -1) { // = если какой-то процесс хочет получить задачи
            rootProcessesCount = DecideRootPretendCount(processDidAllTasks);

            int extraTaskWeight = -1;
            pthread_mutex_lock(&mutex);
            if (!processDidAllTasks && !tasks.empty()) {
                extraTaskWeight = tasks.back(); // берем последний элемент из очереди процесса который еще не закончил выполнение своих задач
                tasks.pop_back(); // удаляем последний элемент из очереди процесса который еще не закончил выполнение своих задач
            }
            pthread_mutex_unlock(&mutex);
            delegateTask(size, rank, actualRootProcess, extraTaskWeight);
        }
    }

    // сообщаем потоку worker что выполнили все задачи
    pthread_mutex_lock(&mutex);
    tasksFinished = true;
    pthread_mutex_unlock(&mutex);
}


int main(int argc, char** argv) {
    int rank;
    int size;
    int provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    // обработка несоответствия ожидаемого и фактического уровней поддержки потоков
    if (provided != MPI_THREAD_MULTIPLE) {
        if (rank == 0) {
            printf("The requested level of thread support does not correspond to the actual level of thread support.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // вычисляем ранг и количество процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // читаем входные параметры и обрабатываем ошибки ввода
    if (argc < 3) {
        if (rank == 0) {
            printf("Usage: mpirun -n countOfProcesses ./fileName L maxIterationsCount\n");
        }
        MPI_Finalize();
        return 1;
    }
    int L = std::atoi(argv[1]);
    if (L <= 0) {
        if (rank == 0) {
            printf("Usage: L is a positive parameter. Please, enter a positive number.\n");
        }
        MPI_Finalize();
        return 1;
    }
    int itersMax = std::atoi(argv[2]);
    if (itersMax <= 0) {
        if (rank == 0) {
            printf("Usage: maxIterationsCount is a positive parameter. Please, enter a positive number.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // инициализация мьютекса
    pthread_mutex_init(&mutex, NULL);

    int startTasksRange = TASKS_PER_PROC * rank;
    int endTasksRange = startTasksRange + TASKS_PER_PROC;

    // подготовка атрибутов для функции менеджера
    int size_rank[2] = { size, rank };

    pthread_attr_t attrs;

    // инициализация атрибутов потоков
    pthread_attr_init(&attrs);
    pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE);

    double maxImbalanceProportion = 0; // перменная которая будет хранить максимальную долю дисбаланса
    double startTime = MPI_Wtime();

    for (int i = 0; i < itersMax; ++i) { // идем в цикле по заданному числу итераций
        // загрузка задач в очередь
        for (int j = startTasksRange; j < endTasksRange; ++j) {
            tasks.push_back(countWeight(size, rank, j, L));
        }
        tasksFinished = false;

        pthread_t worker;
        pthread_t manager;

        // создание потоков с ранее созданными атрибутами
        pthread_create(&worker, &attrs, workerFunc, NULL);
        pthread_create(&manager, &attrs, ManagerTask, (void*)size_rank);

        double iterationStartTime = MPI_Wtime();

        // завершение потоков
        pthread_join(worker, NULL);
        pthread_join(manager, NULL);

        double iterationEndTime = MPI_Wtime();

        double iterationTime = iterationEndTime - iterationStartTime; // время затраченное на итерацию
        double maxIterationTime;
        double minIterationTime;

        MPI_Reduce(&iterationTime, &maxIterationTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // определяем максимальное время на итерации среди всех процессов
        MPI_Reduce(&iterationTime, &minIterationTime, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD); // определяем минимальное время на итерации среди всех процессов

        // вывод результатов итерации
        for (int j = 0; j < size; ++j) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (j == rank) {
                printf("| iter %2d | rank %2d | Tasks computed: %10d      |\n", iterCounter, rank, executedTasks);
                printf("| iter %2d | rank %2d | Global result: %11lf      |\n", iterCounter, rank, globalRes);
                printf("| iter %2d | rank %2d | Time for iteration: %11lf |\n", iterCounter, rank, iterationTime);
                printf("+----------+----------+---------------------------+\n");
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            double imbalanceTime = maxIterationTime - minIterationTime; // считаем время дисбаланса
            double imbalanceProportion = imbalanceTime / maxIterationTime * 100.0; // считаем долю дисбаланса
            if (imbalanceProportion > maxImbalanceProportion) {
                maxImbalanceProportion = imbalanceProportion; // определяем максимальную долю дисбаланса   
            }
            printf("| iter %2d | Imbalance time: %14lf           |\n", iterCounter, imbalanceTime);
            printf("| iter %2d | Imbalance proportion: %12lf %   |\n", iterCounter, imbalanceProportion);
            printf("+----------+---------------------------------+\n");
        }
        executedTasks = 0;
        globalRes = 0;

        iterCounter++;
    }
    double endTime = MPI_Wtime();

    // освобождение ресурсов занятых мьютексом и атрибутами потоков
    pthread_attr_destroy(&attrs);
    pthread_mutex_destroy(&mutex);

    // нулевой процесс выводит итоговое время работы программы и максимальную долю дисбаланса
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Total time spent: %lf\n", endTime - startTime);
        printf("Max imbalance proportion: %lf\n", maxImbalanceProportion);
    }

    MPI_Finalize();
    return 0;
}
