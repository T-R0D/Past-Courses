/**
    @file PA01_pingpong.cpp

    @author Terence Henriod

    Ping Pong: Passing Messages Between Cores of the Same Machine

    @brief This program simply finds the average time for two processor
           cores to pass messages back and forth repeatedly using MPI
           on the UNR grid (although the program should work for any
           network).

    @version Original Code 1.00 (2/12/2014) - T. Henriod
*/

/*==============================================================================
=======     HEADER FILES     ===================================================
==============================================================================*/
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <stdio.h>


/*==============================================================================
=======     USER DEFINED TYPES     =============================================
==============================================================================*/

/**
@struct

Description

@var
*/


/*==============================================================================
=======     GLOBAL CONSTANTS     ===============================================
==============================================================================*/
#define kSuccess 0
#define kError 1
#define kBufferSize 1
#define kTag 0
#define kNumTrials 1000
#define kLeader 0
#define kSubordinate 1
#define NAME_LEN 40


/*==============================================================================
=======     GLOBAL VARIABLES     ===============================================
==============================================================================*/


/*==============================================================================
=======     FUNCTION PROTOTYPES     ============================================
==============================================================================*/
/**
playPingPong

Conducts an experiment to determine the average message passing time
between two nodes using MPI.

@param processor_id   
@param status_ptr

@return average_message_time   The average time it takes to "ping-pong" a
                               message in seconds.

@pre
-# MPI has been properly initialized and the rank/id of each processor has
   been determined.

@post
-# Many messages will have been "ping-ponged" using synchronous communication
   and the average time for the messages will be returned.

@detail @bAlgorithm
-# A "wall clock" time is collected.
-# The lead processor loads a trivial message into the buffer.
-# The lead processor then sends the message to the subordinate processor.
-# Meanwhile, the subordinate processor waits to receive the message.
-# Once the message is received, the subordinate processor sends the same
   message right back.
-# The lead processor recieves the message.
-# This is repeated many times.
-# Once all trials have been completed, a second "wall clock" time is collected.
-# The difference of the wall clock times is then divided by the number
   of trials to find the average ping-pong time, and this is returned.

@code
@endcode
*/
double playPingPong( int processor_id, MPI_Status* status_ptr );


/**
getTimeDifference

Computes the time difference of two "wall clock" times. Converts the result to
seconds.

@param time_one   The first time. Ideally, this will be a "start" time so as
                  to produce a positive time difference, thus indicating
                  elapsed time.
@param time_two   The second time. Ideally this will be a "stop" time.

@return elapsed_seconds   The time difference in seconds.

@pre
-# The parameters represent valid times.

@post
-# The number of seconds difference between the two times will be returned, to
   the precision that the type double offers.

@detail @bAlgorithm
-# Simply takes the difference between the two timeval structs, making
   appropriate order of magnitude conversions between struct members for
   mathematical capability.

@code
@endcode
*/
double getTimeDifference( struct timeval* time_one, struct timeval* time_two  );


/**
reportHostName

Simply a wrapper function for reporting the host machine's name. Handles the
case where hostname is not in the environment variables and available.

@pre
-# None, although this function is more helpful if the hostname is in the
   environment variables list.

@post
-# The host machine's name will either be displayed or the error will be
   reported .

@detail @bAlgorithm
-# A call to getenv() is made.
-# If any name is collected, it is reported.
-# If getenv() returns NULL, then the funciton reports that the program
   could not figure out it's environment name.

@code
@endcode
*/
void reportHostName();

/*==============================================================================
=======     MAIN FUNCTION     ==================================================
==============================================================================*/

/**
main

The main driver for the ping pong timing program.

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
  int program_status = kSuccess;
  int processor_id = 0;
  int num_processors = 0;
  double experiment_result = 0;
  //char processor_name[NAME_LEN];
  MPI_Status status;

  // initialize MPI
  MPI_Init( &argc, &argv );

  // find the size of the Single Program Multiple Data (SPMD) "world"
  MPI_Comm_size( MPI_COMM_WORLD, &num_processors );

  // get this processor's rank
  MPI_Comm_rank( MPI_COMM_WORLD, &processor_id );

  // conduct the experiment
  experiment_result = playPingPong( processor_id, &status );

  // case: this is the lead processor
  if( processor_id == kLeader )
  {
    // report the result of the experiment
    printf( "Average time to ping-pong a message was: %E seconds\n",
             experiment_result );
  }

  // report the host name for to verify which machines this program was run on
  printf( "Processor %d says: ", processor_id );
  reportHostName();

  // try it using MPI_Get_processor_name
  //int length = NAME_LEN;
  //MPI_Get_processor_name( processor_name, &length );
  //printf( "(using MPI_Get_processor_name) I am on %s\n", processor_name);

  // finalize the program
  // end MPI
  MPI_Finalize();

  // return the program status code
  return program_status;
}


/*==============================================================================
=======     FUNCTION IMPLEMENTATIONS     =======================================
==============================================================================*/
double playPingPong( int processor_id, MPI_Status* status_ptr )
{
  // variables
  double average_message_time = 0;
  int trial_num = 0;
  int message_buffer[kBufferSize];
  struct timeval start_time;
  struct timeval stop_time;

  // ensure that the processors are in sync before starting the experiment
  MPI_Barrier( MPI_COMM_WORLD );

  // conduct the message passing experiment many times
  // case: this is the lead processor
  if( processor_id == kLeader )
  {
    // start the clock
    gettimeofday( &start_time, NULL );

    // perform the "ping-pong"
    for( trial_num = 0; trial_num < kNumTrials; trial_num++ )
    {
      // set the message to send
      *message_buffer = trial_num;

      // send the "ping" message
      MPI_Send( message_buffer, kBufferSize, MPI_INT, kSubordinate, kTag,
              MPI_COMM_WORLD );

      // receive the "pong" message
      MPI_Recv( message_buffer, kBufferSize, MPI_INT, kSubordinate, kTag,
              MPI_COMM_WORLD, status_ptr );
    }

    // stop the clock
    gettimeofday( &stop_time, NULL );

    // compute the average message exchange time
    average_message_time = ( getTimeDifference( &start_time, &stop_time ) /
                             kNumTrials );
  }
  // case: this is the subordinate processor
  else
  {
    // conduct the "pong-ping"
    for( trial_num = 0; trial_num < kNumTrials; trial_num++ )
    {
      // receive the "ping" message
      MPI_Recv( message_buffer, kBufferSize, MPI_INT, kLeader, kTag,
                MPI_COMM_WORLD, status_ptr );

      // send the "pong" message
      MPI_Send( message_buffer, kBufferSize, MPI_INT, kLeader, kTag,
                MPI_COMM_WORLD );
    }
  }

  // return the resulting average message passing time
  return average_message_time;
}


double getTimeDifference( struct timeval* time_one, struct timeval* time_two  )
{
  // variables
  double elapsed_seconds = 0;
  long long total_useconds = 0;

  // add the seconds portions of the structs (converted to microseconds)
  total_useconds = ( ( time_two->tv_sec * 1000000 ) -
                      ( time_one->tv_sec * 1000000 ) );

  // add the microsecond portions of the structs
  total_useconds += ( time_two->tv_usec - time_one->tv_usec );

  // convert the number of useconds to seconds in a double
  elapsed_seconds = (double)total_useconds / 1000000.0;

  // return the time difference
  return elapsed_seconds;
}

void reportHostName()
{
  // variables / get the host name
  const char* hostname = getenv( "HOSTNAME" );

  // case: the host name was accessed
  if( hostname != NULL )
  {
    // report the hostname
    printf( "I am running on host: %s\n", hostname );
  }
  // case: the host name could not be found
  else
  {
    // report the lack of a host name
    printf( "I don't know where I am! Help!\n" );
  }

  // no return - void
}

