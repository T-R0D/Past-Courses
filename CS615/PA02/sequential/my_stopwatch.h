/**
    @file my_stopwatch.h

    @author Terence Henriod

    @brief Provides a functions for a "wall clock" stopwatch.    

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
#include <assert.h>


/*==============================================================================
=======     CONSTANTS / MACROS     =============================================
==============================================================================*/
#define false 0
#define true 1


/*==============================================================================
=======     FUNCTION PROTOTYPES     ============================================
==============================================================================*/

/**
stopwatch

Can be used for timing program operation via the wall clock. When stopped,
returns the elapsed time from when the stopwatch is started.

@param action   Used to indicate which action the stopwatch should take,
                starting or stopping. Passing argument 's' starts the time,
                any other argument will stop the watch.

@return elapsed_time   The elapsed time seen by the stopwatch in seconds. Only
                       returns a valid value if the stopwatch was started,
                       then stopped.

@pre
-# The stopwatch should have been started previously for stopping it to give
   a valid elapsed time.

@post
-# If the stopwatch was started, the current time of day is collected and
   stored statically. 0 elapsed time is returned.
-# If the stopwatch is stopped and it was started previously, a valid elapsed
   time will be returned.
-# If the stopwatch is stopped without having been started, the return value is
   undefined.

@detail @bAlgorithm
-# The start and stop times are stored statically so they are available for
   subsequent stopwatch calls.
-# If the stopwatch is started with argument 's', then the current time of
   day is stored.
-# If the stopwatch is stopped, then the current time of day is stored. If the
   stopwatch was started previously, than a valid time difference is
   computed and returned.

@code
@endcode
*/
double stopwatch( char action );


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

double stopwatch( char action )
{
  // variables
  double elapsed_time = 0;
  static struct timeval start_time;
  static struct timeval stop_time;

  // case: we are starting the time
  if( action == 's' )
  {
    // track the start time
    gettimeofday( &start_time, NULL );
  }
  // case: we are stopping the time
  else
  {
    // take the stop time
    gettimeofday( &stop_time, NULL );

    // find out how long the stopwatch ran for
    elapsed_time = getTimeDifference( &start_time, &stop_time );
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
