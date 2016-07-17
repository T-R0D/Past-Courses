/**
    @file PA03_SUMMA_variation.c

    @author Terence Henriod

    Parallel Matrix Multiplication: Modified SUMMA

    @brief This program demonstrates the use of a modified SUMMA, an easily
           implemented matrix multiplication algoritm. 

    @version Original Code 1.00 (4/2/2014) - T. Henriod

    UNOFFICIALLY:
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

/*==============================================================================
=======     HEADER FILES     ===================================================
==============================================================================*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include "my_matrix.h"
#include "my_stopwatch.h"

/*==============================================================================
=======     CONSTANTS / MACROS     =============================================
==============================================================================*/
#define true 1
#define false 0
#define SUCCESS 0
#define ERROR 1
#define MASTER 0
#define VERIFY 0
#define FILE_NAME_SIZE 40
#define MAX_NODES 100


/*==============================================================================
=======     USER DEFINED TYPES     =============================================
==============================================================================*/
typedef struct
{
  int rank;
  int total_nodes;
  int num_slaves;
} NodeInfo;


typedef struct
{
  int use_explicit_results;
  int square_matrix_size;
  char matrix_A_filename[FILE_NAME_SIZE];
  char matrix_B_filename[FILE_NAME_SIZE];
  char output_matrix_filename[FILE_NAME_SIZE];
} ArgData;


typedef struct
{
  Matrix* matrix_A;
  Matrix* matrix_B;
  Matrix* matrix_C;
} MatrixTriplet;


/*==============================================================================
=======     GLOBAL VARIABLES     ===============================================
==============================================================================*/

  // none


/*==============================================================================
=======     FUNCTION PROTOTYPES     ============================================
==============================================================================*/

/**
processCommandLineArguments

Checks the command line arguments for the correct count and validity.
The argument values are then copied to variables local to main.

@param argc               The number of arguments passed on the command line.
@param argv               An array of C-strings containing all of the arguments
                          passed on the command line.

@return arguments_are_valid   Used to indicate to the main program if
                              the agruments are acceptable.

@pre
-# None

@post
-# In the case that there was a problem with the command line arguments are
   bad, an error code is returned.
-# If the arguments are acceptable, then the information is stored in the
   appropriate variables in main.

@detail @bAlgorithm
-# The count of the arguments is checked for the correct number
-# If all of the above conditions are met, then the command line argument
   values are stored to the program. Otherwise, an error code is returned.

@code
@endcode
*/
int processCommandLineArguments( int argc, char** argv, ArgData* arguments );

void master( NodeInfo* job_info, ArgData* arguments, MatrixTriplet* triple );

void slave( NodeInfo* job_info, ArgData* arguments, MatrixTriplet* triple );

void gatherMatrixPortions( Matrix* result_matrix, NodeInfo* job_info,
                           int master_start, int masters_share,
                           int slave_share );

void sendMatrixPortion( Matrix* result_matrix, int start_row, int slave_share );


/*==============================================================================
=======     MAIN FUNCTION     ==================================================
==============================================================================*/

/**
main

The main driver for the demonstration of a sequential matrix multiplication
algorithm.

@param argc   The count of arguments the program was passed with from the
              command line.
@param argv   The list of arguments passed via the command line as c-strings.

@return program_success   A signal to the OS to indicate if the program
                          executed properly.

@pre
-# The program must be run in an MPI supportive environment.

@post
-# The program will accomplish its task of testing the message passing speed
   and report pertinent details.

@code
@endcode
*/
int main( int argc, char** argv )
{
  // variables
  unsigned int program_status = SUCCESS;
  NodeInfo job_info;
  ArgData arguments;
  MatrixTriplet triple = (MatrixTriplet) { NULL, NULL, NULL };

  // initialize MPI
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &(job_info.total_nodes) );
  MPI_Comm_rank( MPI_COMM_WORLD, &(job_info.rank) );
  job_info.num_slaves = job_info.total_nodes - 1;

  // process the command line arguments
  processCommandLineArguments( argc, argv, &arguments );

/*printf( "1.%d\r\n2.%d\r\n3.%s\r\n4.%s\r\n5.%s\r\n", arguments.use_explicit_results,
  arguments.square_matrix_size, arguments.matrix_A_filename,
  arguments.matrix_B_filename, arguments.output_matrix_filename ); */

  // case: we are care about output results for verification
  if( arguments.use_explicit_results )
  {
    // read in the matrices to be multiplied
    readMatrixFromFile( &(triple.matrix_A), arguments.matrix_A_filename );
    readMatrixFromFile( &(triple.matrix_B), arguments.matrix_B_filename );

    // a dummy allocation to prevent segmentation-fault
    triple.matrix_C = createMatrix( 1, 1 );
  }
  // case: we do not care about verifying the ouptut
  else
  {
     // allocate the the matrices with garbage so we have something to
     // compute on
     triple.matrix_A = createMatrix( arguments.square_matrix_size,
                                     arguments.square_matrix_size );
     triple.matrix_B = createMatrix( arguments.square_matrix_size,
                                     arguments.square_matrix_size );
     triple.matrix_C = createMatrix( arguments.square_matrix_size,
                                     arguments.square_matrix_size );  
  }

  // case: this node is the master node
  if( job_info.rank == MASTER )
  {
    // execute the master's code for the procedure/experiment
    master( &job_info, &arguments, &triple );
  }
  // case: this node is a slave
  else
  {
    // execute the slave's code for the procedure/experiment
    slave( &job_info, &arguments, &triple );
  }

  // destroy the matrices
  destroyMatrix( &(triple.matrix_A) );
  destroyMatrix( &(triple.matrix_B) );
  destroyMatrix( &(triple.matrix_C) );

  // end the program and return the program status code
  MPI_Finalize();
  return program_status;
}


/*==============================================================================
=======     FUNCTION IMPLEMENTATIONS     =======================================
==============================================================================*/

int processCommandLineArguments( int argc, char** argv, ArgData* arguments )
{
  // case: there was the correct number of arguments
  if( argc > 1 )
  {
    // case: the command line arguments indicate not to print the matrix results
    //       explicitly for verification purposes
    if( strcmp( argv[1], "Y" ) == 0 )
    {
      // set the flag appropriately
      arguments->use_explicit_results = true;

      // get each of the file name arguments
      strcpy( arguments->matrix_A_filename, argv[2] );
      strcpy( arguments->matrix_B_filename, argv[3] );
      strcpy( arguments->output_matrix_filename, argv[4] );
    }
    // otherwise, do use the explicit results
    else
    {
      // set the flag appropriately
      arguments->use_explicit_results = false;

      // use the first argument to instead define the size of square matrices
      // to use
      arguments->square_matrix_size = atoi( argv[1] );
    }
  }

  // return the arguments converted to appropriate variables by reference
  // return an indication of the validity of the arguments
  return true; // TODO: actual error checking
}


void master( NodeInfo* job_info, ArgData* arguments, MatrixTriplet* triple )
{
  // variables
  int start_row = 0;
  int last_row = 0;
  int masters_share = 0;
  int slave_share = 0;
  float computation_time = 0;
  float total_time = 0;

  // compute the range of rows this process is responsible for
  masters_share = triple->matrix_A->i_size % job_info->num_slaves;
  last_row = triple->matrix_A->i_size - 1;
  start_row = triple->matrix_A->i_size - masters_share;

  // compute the number of rows each slave will be responsible for
  slave_share = triple->matrix_A->i_size / job_info->num_slaves;

  // wait for all processes to get ready
  MPI_Barrier( MPI_COMM_WORLD );

  // start stopwatches for the multiplication timing
  stopwatch( 's', 0 );
  stopwatch( 's', 1 );

  // do the master's part of the multiplication
  partialMatrixMultiply( triple->matrix_C, triple->matrix_A, triple->matrix_B,
                         start_row, last_row );

  // wait for all processors to finish the computation
  MPI_Barrier( MPI_COMM_WORLD );

  // stop the computation time stopwatch
  computation_time = stopwatch( 'x', 0 );

  // case: multiple nodes were required for computations
  if( start_row > 0 )
  {
    // gather the results from the other nodes
    gatherMatrixPortions( triple->matrix_C, job_info, start_row,
                          masters_share, slave_share );
  }

  // stop the total time stopwatch
  total_time = stopwatch( 'x', 1 );

  // output the results in the format:
  // num_nodes, matrix I, matrix J, total (sec), computation (sec),
  // collection (sec) 
  printf( "%d, %d, %d, %F, %F, %F\r\n", job_info->total_nodes,
          triple->matrix_C->i_size, triple->matrix_C->j_size, total_time,
          computation_time, ( total_time - computation_time ) );

  // case: we are using the explicit verification results
  if( arguments->use_explicit_results )
  {
    // write the 3 matrices to the results file
    writeMatrixToFile( triple->matrix_A, arguments->output_matrix_filename );
    writeMatrixToFile( triple->matrix_B, arguments->output_matrix_filename );
    writeMatrixToFile( triple->matrix_C, arguments->output_matrix_filename );
  }

  // no return - void
}


void slave( NodeInfo* job_info, ArgData* arguments, MatrixTriplet* triple )
{
  // variables
  int start_row = 0;
  int last_row = 0;
  int slave_share = 0;

  // compute this node's rows to compute
  slave_share = triple->matrix_A->i_size / job_info->num_slaves;
  start_row = ( job_info->rank - 1 ) * slave_share;
  last_row = ( job_info->rank * slave_share ) - 1;

  // wait for all processes to get ready to start
  MPI_Barrier( MPI_COMM_WORLD );

  // master will start the stopwatches at this point

  // multiply this processor's portion of the matrix
  partialMatrixMultiply( triple->matrix_C, triple->matrix_A, triple->matrix_B,
                         start_row, last_row );

  // wait for all processors to finish their computations
  MPI_Barrier( MPI_COMM_WORLD );

  // master will stop the computation time stopwatch at this point

  // case: multiple nodes were required for computations
  if( last_row > -1 )
  {
    // send the results to the master to be gathered
    sendMatrixPortion( triple->matrix_C, start_row, slave_share );
  }

  // master will stop the total time stopwatch at this point

  // no return - void 
}


void gatherMatrixPortions( Matrix* result_matrix, NodeInfo* job_info,
                           int master_start, int masters_share,
                           int slave_share )
{
  // variables
  int i = 0;
  int number_of_master_elements = masters_share * result_matrix->j_size;
  int number_of_slave_elements = slave_share* result_matrix->j_size;;
  int* receive_counts = (int*) calloc( job_info->total_nodes, sizeof( int ) );
  int* receive_offsets = (int*) calloc( job_info->total_nodes, sizeof( int ) );
  float* master_result_start;

  // load the receive counts and displacement arrays
  receive_counts[0] = number_of_master_elements;
  receive_offsets[0] = job_info->num_slaves * number_of_slave_elements;
  for( i = 1; i < job_info->total_nodes; i++ )
  {
    // load the count
    receive_counts[i] = number_of_slave_elements;

    // load the offset
    receive_offsets[i] = ( i - 1 ) * number_of_slave_elements;
  }

  // use some hacky pointer arithmetic to get a pointer the start of the
  // master computed data
  master_result_start = result_matrix->data + receive_offsets[0];

  // RECEIVE the rows using the MPI_Gatherv function
  MPI_Gatherv( master_result_start,  // send data buffer
               masters_share,        // send count
               MPI_FLOAT,            // send type
               result_matrix->data,  // receive buffer
               receive_counts,       // expected number of elements
                                     //   from each node
               receive_offsets,      // displacements (start indices) for
                                     //   each data set
               MPI_FLOAT,            // receive type
               MASTER,               // root (the receiving node)
               MPI_COMM_WORLD );     // communicator

  // return the dynamic memory used
  free( receive_counts );
  free( receive_offsets );
  receive_counts = NULL;
  receive_offsets = NULL;

  // return matrix C by reference
}


void sendMatrixPortion( Matrix* result_matrix, int start_row, int slave_share )
{
  // variables
  int num_slave_elements = 0;
  float* slave_data_start = NULL;

  // compute the number of individual matrix elements that will be sent
  num_slave_elements = slave_share * result_matrix->j_size;

  // use hacky pointer arithmetic to find a pointer to the start
  // of the slaves computed data
  slave_data_start = result_matrix->data +
                     ( start_row * result_matrix->j_size );

  // SEND the rows using the MPI_Gather function
  MPI_Gatherv( slave_data_start,   // send data buffer
               num_slave_elements, // send count
               MPI_FLOAT,          // send type
               NULL,               // receive buffer
               NULL,               // expected number of elements from
                                   //   each node
               NULL,               // displacements (start indices) for
                                   //   each data set
               MPI_FLOAT,          // receive type
               MASTER,             // root (the receiving node)
               MPI_COMM_WORLD );   // communicator

  // no return - void
}



