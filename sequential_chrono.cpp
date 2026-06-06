/*
 * sequential_chrono.cpp — versión secuencial sin MPI (usa std::chrono)
 *
 * Compilación:
 *   g++ -o sequential_chrono sequential_chrono.cpp
 *
 * Uso:
 *   ./sequential_chrono <M> <R> [seed]
 *
 *   <M>  : número de filas  (entero positivo)
 *   <R>  : número de columnas (entero positivo)
 *
 * Ejemplo:
 *   ./sequential_chrono 1000 1000 12345
 */

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#endif

// Valor máximo que puede tomar un elemento de la matriz
static const int MAX_VAL = 1000000000;
static const int MIN_VAL = 5000000;

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

int* buildMatrix(int M, int R, unsigned int seed) {
    int total = M * R;
    int* mat = new int[total];

    for (long long k = 0; k < total; ++k) {
        mat[k] = generateValue(seed, k);
    }

    return mat;
}

int main(int argc, char* argv[]){

#ifdef _WIN32
    // Forzar la consola de Windows a UTF-8 para evitar caracteres corruptos
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    if (argc != 3 && argc != 4) {
        std::cerr << "\n[ERROR] Número incorrecto de argumentos.\n"
                  << "  Uso correcto : " << argv[0] << " <M> <R> [seed]\n"
                  << "  Ejemplo      : " << argv[0] << " 1000 1000 12345\n\n";
        return 1;
    }

    int M = std::atoi(argv[1]);
    int R = std::atoi(argv[2]);
    unsigned int seed = (argc == 4) ? static_cast<unsigned int>(std::strtoul(argv[3], nullptr, 10)) : 12345u;

    if (M <= 0 || R <= 0) {
        std::cerr << "\n[ERROR] Los tamaños M y R deben ser enteros positivos.\n"
                  << "  Recibido: M=" << argv[1] << "  R=" << argv[2] << "\n\n";
        return 1;
    }

    // Configuration display removed

    auto t_build_inicio = std::chrono::high_resolution_clock::now();

    int* A = buildMatrix(M, R, seed);

    auto t_build_fin = std::chrono::high_resolution_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(t_build_fin - t_build_inicio).count();
    auto t_inicio = std::chrono::high_resolution_clock::now();

    long long conteo = 0;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < R; j++) {
            int valor = A[i * R + j];
            if (isPrime(valor)) {
                conteo++;
            }
        }
    }

    auto t_fin = std::chrono::high_resolution_clock::now();

    double tiempo_ms = std::chrono::duration<double, std::milli>(t_fin - t_inicio).count();
    double totalMs = build_ms + tiempo_ms;

    std::cout << "sequential," << M << "," << R << ","
              << build_ms << "," << tiempo_ms << "," << totalMs << "\n";

    delete[] A;
    return 0;
}
