#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#define width 800
#define height 600
#define maxi 1000

int main(int argc, char** argv) {
    clock_t start, end;
    double cputime;
    start = clock();
    
    int rank;
    int size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double t_start = MPI_Wtime();

    int tasks = width * height;
    int proc_tasks = ceil(tasks / (double)size);

    int s_task = rank * proc_tasks;
    int e_task = fmin(s_task + proc_tasks, tasks);

    int mandelbrot[height][width] = {0};
    for (int i = s_task; i < e_task; i++) {
        int x = i % width;
        int y = i / width;

        double a = (x - width/2.0)*4.0/width;
        double b = (y - height/2.0)*4.0/width;
        double c = 0, d = 0;

        int iteration;
        for (iteration = 0; iteration < maxi; iteration++) {
            double q = c*c - d*d + a;
            double p = 2*c*d + b;
            c = q;
            d = p;

            if (c*c + d*d > 4) {
                break;
            }
        }
        mandelbrot[y][x] = iteration;
    }

    // Gather computed results to root process
    int* count = malloc(size * sizeof(int));
    int* astro = malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        int astros = i == size - 1 ? tasks - i * proc_tasks : proc_tasks;
        count[i] = astros;
        astro[i] = i * proc_tasks;
    }

    int* mb = NULL;
    if (rank == 0) {
        mb = malloc(tasks * sizeof(int));
    }
    MPI_Gatherv(&(mandelbrot[0][s_task]), e_task - s_task, MPI_INT,
                mb, count, astro, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE* file = fopen("mandelbrot.ppm", "wb");
        fprintf(file, "P6 %d %d 255\n", width, height);
        for (int i = 0; i < tasks; i++) {
            int dg = mb[i] * 255 / maxi;
            fputc(dg, file);
            fputc(dg, file);
            fputc(dg, file);
        }
        fclose(file);
        free(mb);

        double t_end = MPI_Wtime();
        printf("Execution time: %.2f seconds\n", t_end – t_start);
    }

    MPI_Finalize();
    
    end = clock();
    cputime = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("CPU time used: %f seconds\n", cputime);
    
    return 0;
}
