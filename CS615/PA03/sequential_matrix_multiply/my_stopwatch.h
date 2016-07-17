/**
    @file my_stopwatch.h

    @author Terence Henriod

    @brief Provides a functions for a "wall clock" stopwatch. Supports five
           stopwatches simultaneously.    

    @version Original Code 1.00 (2/28/2014) - T. Henriod

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

#ifndef __MY_STOPWATCH_H__
#define __MY_STOPWATCH_H__

/*==============================================================================
=======     HEADER FILES     ===================================================
==============================================================================*/
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <assert.h>


/*==============================================================================
=======     CONSTANTS / MACROS     =============================================
==============================================================================*/
#define false 0
#define true 1
#define NUM_STOPWATCHES 10
#define WATCH_ONE 0
#define WATCH_TWO 1
#define WATCH_THREE 2
#define WATCH_FOUR 3
#define WATCH_FIVE 4
#define WATCH_SIX 5
#define WATCH_SEVEN 6
#define WATCH_EIGHT 7
#define WATCH_NINE 8
#define WATCH_TEN 9


/*==============================================================================
=======     FUNCTION PROTOTYPES     ============================================
==============================================================================*/

/**
stopwatch

Can be used for timing program operation via the wall clock. When stopped,
returns the elapsed time from when the stopwatch is started.

@param action            Used to indicate which action the stopwatch should
                         take, starting or stopping. Passing argument 's'
                         starts the time, any other argument will return the
                         time since the stopwatch was started.
@param which_stopwatch   Indicates which of the stopwatches is being referenced.

@return elapsed_time   The elapsed time seen by the stopwatch in seconds. Only
                       returns a valid value if the stopwatch was started,
                       then stopped.

@pre
-# The stopwatch should have been started previously to "stopping" it to give
   a valid elapsed time.

@post
-# If the stopwatch was started, the current time of day is collected and
   stored statically. -1 elapsed time is returned.
-# If the stopwatch is checked and it was started previously, a valid elapsed
   time will be returned.
-# If the stopwatch is stopped without having been started, the return value is
   undefined.

@detail @bAlgorithm
-# The start and stop times are stored statically so they are available for
   subsequent stopwatch calls.
-# If the stopwatch is "started" with argument 's', then the current time of
   day is stored.
-# If the stopwatch is "stopped", then the current time of day is stored.
   The time difference is then computed and returned.

@code
  double time_passed;

  stopwatch( 's', WATCH_ONE ); // starts the first stopwatch

  time_passed = stopwatch( 'x', WATCH_ONE ); // checks the time elapsed from the
                                             // first stopwatch call

  time_passed = stopwatch( 'c', WATCH_ONE ); // also checks the time elapsed
                                             // from the first stopwatch call

  stopwatch( 's', WATCH_ONE ); // re-starts the first stopwatch

  time_passed = stopwatch( 'z', WATCH_ONE ); // checks the time elapsed from
                                             // the re-started stopwatch

  // Note that any argument works for getting the elapsed time

@endcode
*/
double stopwatch( char action, int which_stopwatch );

unsigned int getElapsedMilliseconds( int which_stopwatch );

unsigned int getElapsedMicroseconds( int which_stopwatch );

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
   mathematical compatability.

@code
@endcode
*/
double getTimeDifference( struct timeval* time_one, struct timeval* time_two  );


/*==============================================================================
=======     FUNCTION IMPLEMENTATIONS     =======================================
==============================================================================*/

double stopwatch( char action, int which_stopwatch )
{
  // variables
  double elapsed_time = -1;
  static struct timeval start_times[NUM_STOPWATCHES];
  static struct timeval stop_times[NUM_STOPWATCHES];
  static int stopwatch_started_flags[NUM_STOPWATCHES] = {false, false, false,
                                                         false, false, false,
                                                         false, false, false,
                                                         false };
  // TODO: make the stopwatch started flags relevant

  // case: we are starting the time
  if( action == 's' )
  {
    // track the start time
    gettimeofday( &(start_times[which_stopwatch]), NULL );
  }
  // case: we are stopping the time
  else
  {
    // take the stop time
    gettimeofday( &(stop_times[which_stopwatch]), NULL );

    // find out how long the stopwatch ran for
    elapsed_time = getTimeDifference( &(start_times[which_stopwatch]),
                                      &(stop_times[which_stopwatch]) );
  }

  // return the time value in seconds
  return elapsed_time;
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

#endif
