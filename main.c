#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "myProto.h"

#define ROOT 0

int* read_data(int *N);
void openMP_task(int *inputs, int *histogram, int size);
void join_histogram(int *histogram, int *to_join);
void print_histogram(int * histogram);

int main(int argc, char *argv[])
{
	int rank, num_of_procs, N, job_size, histogram[RANGE + 1] = {0}, temp_hist[RANGE + 1];
	int* inputs;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_of_procs);

	if (num_of_procs != 2)
	{
		printf("This program requires 2 processes only\n");
		MPI_Abort(MPI_COMM_WORLD, __LINE__);
	}

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == ROOT)
	{
		inputs = read_data(&N);

		job_size = N / 4;

		// Send proc 1 number of input
		MPI_Send(&N, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

		// Send proc 1 half of the input
		MPI_Send(inputs + 2*job_size, 2*job_size, MPI_INT, 1, 0, MPI_COMM_WORLD);
	} else {
		MPI_Recv(&N, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		
		job_size = N / 4;
		// Allocate memory for input
		if ((inputs = (int *) calloc(2*job_size, sizeof(int))) == NULL)
			MPI_Abort(MPI_COMM_WORLD, __LINE__);

		MPI_Recv(inputs, 2*job_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
	}

	// Calculate histogram for 1 / 4 of the input size with openMP
	openMP_task(inputs, histogram, job_size);

	// Calculate histogram for 1 / 4 of the input size with cuda
	cuda_task(histogram, inputs + job_size, job_size);


	MPI_Barrier(MPI_COMM_WORLD);

	if (rank != ROOT)
	{
		// Send proc 0 the resulted histogram for proc 1 part
		MPI_Send(histogram, RANGE + 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Recv(temp_hist, RANGE + 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
		
		join_histogram(histogram, temp_hist);

		print_histogram(histogram);
	}

	free(inputs);

	MPI_Finalize();

}

int* read_data(int *N)
{
	int i, n = 0;
	int* inputs;

	fscanf(stdin, "%d", N);

	inputs = (int*)calloc(*N, sizeof(int));

	n = *N;
	for (i = 0; i < n; ++i) {
		fscanf(stdin, "%d", &inputs[i]);
	}

	return inputs;
}

void openMP_task(int *inputs, int *histogram, int size) {
#pragma omp parallel
{
    int i, histogram_private[RANGE + 1] = {0};
    #pragma omp for
    for(i=0; i<size; i++) {
           histogram_private[inputs[i]]++;
    }
    #pragma omp critical
    {
        for(i = 0; i < RANGE + 1; i++) histogram[i] += histogram_private[i];
    }
}
}

void join_histogram(int *histogram, int *to_join) {
#pragma omp for
	for (int i = 0; i < RANGE + 1; ++i)
		histogram[i] += to_join[i];
}

void print_histogram(int * histogram) {
	for (int i = 0; i < RANGE + 1; ++i) {
		if (histogram[i] != 0) {
			printf("%d: %d\n", i, histogram[i]);
		}
	}
}
