/*
 * mpi_dynamic.cpp - MPI version with dynamic row distribution
 *
 * Compilation:
 *   mpic++ -o mpi_dynamic mpi_dynamic.cpp
 *
 * Usage:
 *   mpirun -np <P> ./mpi_dynamic <M> <R> [seed]
 *
 *   <M>    : number of rows    (positive integer)
 *   <R>    : number of columns (positive integer)
 *   [seed] : optional random seed shared across versions
 *
 * Example:
 *   mpirun -np 4 ./mpi_dynamic 1000 1000 12345
 */

#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

static const int MAX_VAL = 1000000000;
static const int MIN_VAL = 5000000;
static const int DYNAMIC_CHUNK_ROWS = 4; // Filas por tarea
static const int TAG_TASK = 1;
static const int TAG_RESULT = 2;
static const int TAG_DATA = 3; // etiqueta para enviar bloques de datos

// Puntero al bloque local de la matriz por proceso.
static int* matrix = nullptr;

// Construye el bloque local: filas [startRow, startRow+rowCount).
inline int generateValue(unsigned int seed, long long index)
{
    // Hash determinista rapido para un indice
    unsigned int x = seed ^ (index >> 32);
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x ^= index & 0xFFFFFFFF;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return MIN_VAL + (x % (MAX_VAL - MIN_VAL + 1));
}


void buildMatrixChunk(int startRow, int rowCount, int R, unsigned int seed) {
    if (rowCount <= 0) return;
    long long baseIndex = (long long)startRow * R;
    long long total = (long long)rowCount * R;
    for (long long k = 0; k < total; ++k) {
        // Genera el valor determinista para el indice global
        matrix[k] = generateValue(seed, baseIndex + k);
    }
}


bool isPrime(int n)
{
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;

    int raiz = static_cast<int>(std::sqrt(static_cast<double>(n)));
    for (int i = 3; i <= raiz; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

long long countPrimesInRowRange(int startRow, int rowCount, int R, unsigned int seed)
{
    long long count = 0;

    // Genera y cuenta primos fila a fila sin guardar matriz
    long long baseIndex = (long long)startRow * R;
    for (int i = 0; i < rowCount; ++i) {
        for (int j = 0; j < R; ++j) {
            int value = generateValue(seed, baseIndex + j);
            if (isPrime(value)) {
                ++count;
            }
        }
        baseIndex += R;
    }

    return count;
}

void sendTask(int destination, int startRow, int rowCount)
{
    int buf[2]; buf[0] = startRow; buf[1] = rowCount;
    MPI_Send(buf, 2, MPI_INT, destination, TAG_TASK, MPI_COMM_WORLD);
}


int main(int argc, char* argv[])
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3 && argc != 4) {
        if (rank == 0) {
            std::cerr << "\n[ERROR] Incorrect number of arguments.\n"
                      << "  Correct usage : mpirun -np <P> " << argv[0] << " <M> <R> [seed]\n"
                      << "  Example       : mpirun -np 4 " << argv[0] << " 1000 1000 12345\n\n";
        }
        MPI_Finalize();
        return 1;
    }

    int M = std::atoi(argv[1]);
    int R = std::atoi(argv[2]);
    unsigned int seed = (argc == 4) ? static_cast<unsigned int>(std::strtoul(argv[3], nullptr, 10)) : 12345u;

    if (M <= 0 || R <= 0) {
        if (rank == 0) {
            std::cerr << "\n[ERROR] M and R must be positive integers.\n"
                      << "  Received: M=" << argv[1] << "  R=" << argv[2] << "\n\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Configuration display removed

    // Comparte la semilla con todos
    MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    // Rango local de filas (igual que en mpi_static)
    int baseRows = M / size;
    int extraRows = M % size;
    int localRows = baseRows + (rank < extraRows ? 1 : 0);
    int localStartRow = rank * baseRows + std::min(rank, extraRows);

    // Cada proceso genera su bloque local
    double localGenMs = 0.0;
    if (rank == 0) {
        std::cout << "[INFO] Each process pre-generates its chunk, then master gathers..." << std::flush;
    }

    double gen0 = MPI_Wtime();
    int sendCount = localRows * R; // number of ints to send from this rank
    if (localRows > 0) {
        matrix = new int[(size_t)sendCount];
        buildMatrixChunk(localStartRow, localRows, R, seed);
    }
    double gen1 = MPI_Wtime();
    localGenMs += (gen1 - gen0) * 1000.0;

    // Prepara Gatherv en el root
    int* recvcounts = nullptr;
    int* displs = nullptr;
    int* allRows = nullptr;
    if (rank == 0) {
        recvcounts = new int[size];
        displs = new int[size];
        allRows = new int[size];
        int offset = 0;
        for (int i = 0; i < size; ++i) {
            int rows = baseRows + (i < extraRows ? 1 : 0);
            allRows[i] = rows;
            recvcounts[i] = rows * R;
            displs[i] = offset;
            offset += recvcounts[i];
        }
    }

    // Reune todos los bloques en root
    int totalElems = M * R;
    int* fullMatrix = nullptr;
    if (rank == 0) fullMatrix = new int[(size_t)totalElems];

    MPI_Gatherv(matrix, sendCount, MPI_INT,
                fullMatrix, recvcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    // Libera el bloque local
    if (matrix) { delete[] matrix; matrix = nullptr; }

    // Sincroniza antes de la planificacion dinamica
    MPI_Barrier(MPI_COMM_WORLD);

    double processStart = MPI_Wtime();
    double localTimeMs = 0.0;
    long long totalCount = 0;

    if (size == 1) {
        // Un solo proceso: genera y cuenta todo
        double t0 = MPI_Wtime();
        totalCount = countPrimesInRowRange(0, M, R, seed);
        double t1 = MPI_Wtime();
        localTimeMs = (t1 - t0) * 1000.0;
    } else if (rank == 0) {
        // Master: distribuye tareas y recoge resultados
        int nextRow = 0;
        int activeWorkers = 0;

        for (int worker = 1; worker < size; ++worker) {
            if (nextRow < M) {
                int rowsToSend = std::min(DYNAMIC_CHUNK_ROWS, M - nextRow);
                int start = nextRow;
                // envia metadatos
                sendTask(worker, start, rowsToSend);
                // envia bloque desde fullMatrix
                MPI_Send(fullMatrix + (start * R), rowsToSend * R, MPI_INT, worker, TAG_DATA, MPI_COMM_WORLD);
                nextRow += rowsToSend;
                ++activeWorkers;
            } else {
                sendTask(worker, 0, 0);
            }
        }

        while (activeWorkers > 0) {
            long long partialCount = 0;
            MPI_Status status{};
            MPI_Recv(&partialCount, 1, MPI_LONG_LONG, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);
            totalCount += partialCount;

            int worker = status.MPI_SOURCE;
            if (nextRow < M) {
                int rowsToSend = std::min(DYNAMIC_CHUNK_ROWS, M - nextRow);
                int start = nextRow;
                sendTask(worker, start, rowsToSend);
                MPI_Send(fullMatrix + (start * R), rowsToSend * R, MPI_INT, worker, TAG_DATA, MPI_COMM_WORLD);
                nextRow += rowsToSend;
            } else {
                sendTask(worker, 0, 0);
                --activeWorkers;
            }
        }
    } else {
        // Worker: recibe tareas y procesa
        while (true) {
            int buf[2] = {0,0};
            MPI_Recv(buf, 2, MPI_INT, 0, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int startRow = buf[0];
            int rowCount = buf[1];

            if (rowCount <= 0) {
                break;
            }

            long long localCount = 0;

            // Recibe bloque del master (rowCount * R ints)
            int total = rowCount * R;
            int* dataBuf = new int[(size_t)total];
            MPI_Recv(dataBuf, total, MPI_INT, 0, TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            double c0 = MPI_Wtime();
            for (int k = 0; k < total; ++k) if (isPrime(dataBuf[k])) ++localCount;
            double c1 = MPI_Wtime();
            localTimeMs += (c1 - c0) * 1000.0;

            delete[] dataBuf;

            MPI_Send(&localCount, 1, MPI_LONG_LONG, 0, TAG_RESULT, MPI_COMM_WORLD);
        }
    }

    // Reduce timing statistics - only max values needed
    double maxGenMs = 0.0, maxTimeMs = 0.0;
    MPI_Reduce(&localGenMs, &maxGenMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localTimeMs, &maxTimeMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double totalTimeMs = maxGenMs + maxTimeMs;
        std::cout << "mpi_dynamic," << M << "," << R << "," << size << ","
                  << maxGenMs << "," << maxTimeMs << "," << totalTimeMs << "\n";
    }

    // Limpieza de arrays y fullMatrix en root
    if (rank == 0) {
        delete[] recvcounts;
        delete[] displs;
        delete[] allRows;
        delete[] fullMatrix;
    }

    // No hay matriz global que liberar
    MPI_Finalize();
    return 0;
}