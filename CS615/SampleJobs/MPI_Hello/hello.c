#include <mpi.h>
#include <stdio.h>

int ID;

void hello ()
{
	printf ("My ID is %i\n", ID);
	fflush (stdout);
}

int main (int argc, char **argv)
{
	int size;

	MPI_Init (&argc, &argv);

	MPI_Comm_size (MPI_COMM_WORLD, &size);
	MPI_Comm_rank (MPI_COMM_WORLD, &ID);

	hello ();

	MPI_Finalize ();

	return 0;
}
