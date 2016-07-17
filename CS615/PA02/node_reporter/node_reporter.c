#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main( int argc, char** argv)
{
  // variables
  const char* hostname = getenv( "HOSTNAME" );
  int processor_id = 0;
  int num_processors = 0;
  int processor_rank = 0;

  // intialize MPI
  MPI_Init( &argc, &argv );

  // find the size of the MPI world
  MPI_Comm_size( MPI_COMM_WORLD, &num_processors );

  // find this processor's rank
  MPI_Comm_rank( MPI_COMM_WORLD, &processor_rank );


  // report all processor names
  for( processor_id = 0; processor_id < num_processors; processor_id++ )
  {
    MPI_Barrier( MPI_COMM_WORLD );
    if( processor_rank == processor_id )
    {
      printf( "%d: %s\r\n", processor_id, hostname );
      fflush( stdout );
    }
    MPI_Barrier( MPI_COMM_WORLD );
  }

  // finalize MPI
  MPI_Finalize();

  return 0;
}

