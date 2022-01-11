#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0 // taskid von erstem task
#define FROM_MASTER 1 // Nachrichtentypen
#define FROM_WORKER 2
#define D1 1000 // Zeilenanzahl von A und C
#define D2 1000 // Spaltenanzahl von A und Zeilenanzahl von B
#define D3 1000 // Spaltenanzahl von B und C

// ---------------------------------------------------------------------------
// allocate space for empty matrix A[row][col]
// access to matrix elements possible with:
// - A[row][col]
// - A[0][row*col]

float** alloc_mat(int row, int col)
{
	float** A1, * A2;

	A1 = (float**)calloc(row, sizeof(float*));	 // pointer on rows
	A2 = (float*)calloc(row * col, sizeof(float));    // all matrix elements
	for (int i = 0; i < row; i++)
		A1[i] = A2 + i * col;

	return A1;
}

// ---------------------------------------------------------------------------
// random initialisation of matrix with values [0..9]

void init_mat(float** A, int row, int col)
{
	for (int i = 0; i < row * col; i++)
		A[0][i] = (float)(rand() % 10);
}

// ---------------------------------------------------------------------------
// DEBUG FUNCTION: printout of all matrix elements

void print_mat(float** A, int row, int col, char const* tag)
{
	int i, j;

	printf("Matrix %s:\n", tag);
	for (i = 0; i < row; i++)
	{
		for (j = 0; j < col; j++)
			printf("%6.1f   ", A[i][j]);
		printf("\n");
	}
}

// ---------------------------------------------------------------------------
// free dynamically allocated memory, which was used to store a 2D matrix
void free_mat(float** A) {
	free(A[0]); // free contiguous block of float elements (row*col floats)
	free(A);    // free memory for pointers pointing to the beginning of each row
}

float** serial_matrix(int d1, int d2, int d3, float** A, float** B) {
	float** C;	                    // matrix
	int i, j, k;			        // loop variables
	C = alloc_mat(d1, d3);	        // no initialisation of C, because it gets filled by matmult

	/* serial version of matmult */
	printf("Perform matrix multiplication...\n");
	for (i = 0; i < d1; i++)
		for (j = 0; j < d3; j++)
			for (k = 0; k < d2; k++)
				C[i][j] += A[i][k] * B[k][j];
	return C;
}

bool compare_function(float** A, int d1, int d2, int d3, float** A_, float** B_) {
	float** B = serial_matrix(d1, d2, d3, A_, B_);
	for (int i = 0; i < d1; i++) {
		for (int j = 0; j < d3; j++) {
			if (A[i][j] != B[i][j]) {
				return false;
			}
		}
	}
	return true;
}

// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	int numtasks, // Anzahl an Tasks
		taskid, // Task ID
		numworkers, i, // Anzahl an Arbeitern
		bsize, bpos, // Zeilenabschnitt von Matrix A
		averow, extra, // Berechnung von Zeilenabschnitten
		k, j; // Zählvariablen
	float** A, ** B, ** C, ** D; // Matrizen

	double start_verteilung,
		   end_verteilung,
		   start_serial,
		   end_serial;

	MPI_Status status[7]; // Statusvariable
	MPI_Request send_request[7], recv_request[7]; 
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	if (numtasks < 2) {
		printf("Need at least two tasks!\n");
		MPI_Abort(MPI_COMM_WORLD, 0); exit(1);
	}
	//****************************** Master Task ************************************
	if (taskid == MASTER) {
		printf("MatMult started with %d tasks.\n", numtasks);
		A = alloc_mat(D1, D2); init_mat(A, D1, D2); // Speicher für Matrizen holen
		B = alloc_mat(D2, D3); init_mat(B, D2, D3); // und initialisieren
		C = alloc_mat(D1, D3);
		start_verteilung = MPI_Wtime();
		numworkers = numtasks - 1; // Anzahl der Arbeiter
		averow = D1 / numworkers; // Mittlere Blockgröße
		extra = D1 % numworkers; // Restzeilen
		for (i = 1, bpos = 0; i <= numworkers; i++, bpos += bsize) {
			if (i > extra) { // Restzeilen aufteilen
				bsize = averow;
			}
			else {
				bsize = averow + 1;
			} // Senden der Matrixblöcke an die Arbeiter
			printf("Sending %d rows to task %d\n", bsize, i);
			MPI_Isend(&bpos, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &send_request[0]);
			MPI_Wait(&send_request[0], &status[0]);
			MPI_Isend(&bsize, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &send_request[1]);
			MPI_Wait(&send_request[1], &status[1]);
			MPI_Isend(A[bpos], bsize * D2, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &send_request[2]);
			MPI_Wait(&send_request[2], &status[2]);
			MPI_Isend(B[0], D2 * D3, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &send_request[3]);
			MPI_Wait(&send_request[3], &status[3]);

			end_verteilung = MPI_Wtime();
		}
		
		for (i = 1; i <= numworkers; i++) { // Empfangen der Ergebnisse von den Arbeitern
			MPI_Irecv(&bpos, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &recv_request[0]);
			MPI_Wait(&recv_request[0], &status[0]);
			MPI_Irecv(&bsize, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &recv_request[1]);
			MPI_Wait(&recv_request[1], &status[0]);
			MPI_Irecv(C[bpos], bsize * D3, MPI_FLOAT, i, 2, MPI_COMM_WORLD, &recv_request[2]);
			MPI_Wait(&recv_request[2], &status[0]);
			printf("Received results from task %d\n", i);
		}

		// Ergebnis überprüfes
		start_serial = MPI_Wtime();
		if (compare_function(C, D1, D2, D3, A, B)) {
			printf("Funktioniert optimal!\n");
		}
		else {
			printf("Funktioniert nicht so optimal!\n");
		}
		end_serial = MPI_Wtime();

		printf("Verteilung: %f\n", end_verteilung - start_verteilung);
		printf("Seriell: %f\n\n", end_serial - start_serial);
	}
	//****************************** Worker Task ************************************
	if (taskid > MASTER) { // Worker

		double start_worker_verteilung = MPI_Wtime();

		MPI_Irecv(&bpos, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &recv_request[3]);
		MPI_Wait(&recv_request[3], &status[0]);
		MPI_Irecv(&bsize, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &recv_request[4]);
		MPI_Wait(&recv_request[4], &status[0]);

		A = alloc_mat(bsize, D2); // Speicher für die Matrixblöcke holen
		B = alloc_mat(D2, D3);
		C = alloc_mat(bsize, D3);

		MPI_Irecv(A[0], bsize * D2, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &recv_request[5]);
		MPI_Wait(&recv_request[5], &status[0]);
		MPI_Irecv(B[0], D2 * D3, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &recv_request[6]);
		MPI_Wait(&recv_request[6], &status[0]);

		printf("Worker verteilung: %f\n", MPI_Wtime() - start_worker_verteilung);
		double start_worker_berechnung = MPI_Wtime();

		for (i = 0; i < bsize; i++)
			for (j = 0; j < D3; j++)
				for (k = 0; k < D2; k++)
					C[i][j] += A[i][k] * B[k][j];

		printf("Worker berechnung: %f\n", MPI_Wtime() - start_worker_berechnung);
		double start_worker_einsammeln = MPI_Wtime();

		MPI_Isend(&bpos, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &send_request[4]);
		MPI_Wait(&send_request[4], &status[0]);
		MPI_Isend(&bsize, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &send_request[5]);
		MPI_Wait(&send_request[5], &status[0]);
		MPI_Isend(C[0], bsize * D3, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &send_request[6]);
		MPI_Wait(&send_request[6], &status[0]);
		//MPI_Waitall(7, request, status);
		
		printf("Worker einsammeln: %f\n\n", MPI_Wtime() - start_worker_einsammeln);
	}
	MPI_Finalize();
}