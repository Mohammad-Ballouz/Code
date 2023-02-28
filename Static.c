#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#define master 0

void mandelbro_t(int width, int height, int s, int e, double left, double right, double bottom, double high, int* result) {
    int maxi = 1000;
    double astro, astros, dg, dt, astro2, astros2;
    int count;
    for (int i = s; i < e; i++) {
        for (int j = 0; j < width; j++) {
            astro = left + (right - left) * j / width;
            astros = bottom + (high - bottom) * i / height;
            dg = 0.0;
            dt = 0.0;
            astro2 = 0.0;
            astros2 = 0.0;
            count = 0;
            
            while (astro2 + astros2 < 4.0 && count < maxi) {
                dt = 2 * dg * astros + astros;
                dg = astro2 - astros2 + astro;
                astro2 = dg*dg;
                astros2 = dt * dt;
                count++;
            }
            result[i * width + j] = count;
        }
    }
}

int main(int argc, char** argv) {
    double start = clock();
    int width = 800;
    int height = 800;
    double left = -2.0;
    double right = 1.0;
    double bottom = -1.5;
    double high = 1.5;
    int maxi = 1000;
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int rows = height / size;
    int first_row = rank * rows;
    int last_row = (rank + 1) * rows;
    if (rank == size - 1) {
        last_row = height;
    }
    int count = last_row - first_row;
    
    int* astro = (int*) malloc(count * width * sizeof(int));
    mandelbro_t(width, height, first_row, last_row, left, right, bottom, high, astro);
    
    int* rip = NULL;
    if (rank == master) {
        rip = (int*) malloc(width * height * sizeof(int));
    }
    
    MPI_Gather(astro, count * width, MPI_INT, rip, count * width, MPI_INT, master, MPI_COMM_WORLD);
    
    if (rank == master) {
        FILE* file = fopen("mandelbrot.pgm", "wb");
        fprintf(file, "P2\n%d %d\n%d\n", width, height, maxi - 1);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                fprintf(file, "%d ", rip[i * width + j]);
            }
            fprintf(file, "\n");
        }
        fclose(file);
        free(rip);
    }
    
    free(astro);
    MPI_Finalize();
    
    double end = clock();
    double cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("CPU time used: %f seconds\n", cpu_time);
    return 0;
}
