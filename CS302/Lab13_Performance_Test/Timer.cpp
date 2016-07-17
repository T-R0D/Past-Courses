// Timer.cpp

#include "Timer.h"
#include "sys/time.h"

/**
The default constructor for the timer class.
This constructor initializes all data members to 0 or
equivalent values.

@code
@endcode
*/
Timer::Timer()
{
  // initialize data members
  beginTime.tv_sec = 0;      // timeval is a struct with second and
  beginTime.tv_usec = 0;     // microsecond members
  duration.tv_sec = 0;
  duration.tv_usec = 0;
  timerWasStarted = false;
  timerWasStopped = false;

  // no return - constructor
}

/**
Starts the timer for the Timer class.

Uses "wall clock" functionality to get a beginning time. This time is then stored
in the beginTime data member. Will throw a runtime error if the gettimeofday
function fails.

@code
@endcode
*/
void Timer::start() throw (runtime_error)
{
  // variables
  int errorFlag = 0;

  // get the time of day
  errorFlag = gettimeofday( &beginTime, NULL );

  // check for time getting error
  if( errorFlag == -1 )
  {
    // throw exception
    throw( "Failed to get time of day for timer start" );
  }

  // set timer was started flag
  timerWasStarted = true;

  // no return - void
}

/**
Stops the timer in the Timer class.

Uses the "wall clock" functionality to record a stopping time in the duration data member.
This time will be compared to the timer's start time to compute and elapsed time. Will throw
a logic error if the the timer has not been started. Throws a runtime error if the
gettimeofday function fails.

@code
@endcode
*/
void Timer::stop() throw (logic_error, runtime_error)
{
  // variables
  int errorFlag = 0;

  // case: timer has been started
  if( timerWasStarted )
  {
    // stop the timer by getting an end time
    errorFlag = gettimeofday( &duration, NULL );

    // check for time getting error
    if( errorFlag == -1 )
    {
      // throw exception
      throw( "Failed to get time of day for timer stop" );
    }

    // reset timer was started flag, set timer was stopped flag
    timerWasStarted = false;
    timerWasStarted = true;
  }
  // case: timer has not been started
  else
  {
    throw logic_error( "The timer was not started, so it can't be stopped." );
  }

  // no return - void
}

/**
Computes the time measured by the Timer class after a complete start/stop cycle.

Utilizes stored beginTime and duration values to return an elapsed time in seconds.
This function will throw a logic error if an elapsed time is solicited, but the
timer is still being run (that is, the timer has not been stopped).

@code
@endcode
*/
double Timer::getElapsedTime() const throw (logic_error)
{
  // variables
  double elapsedTime = 0;
  double startDouble = 0;
  double stopDouble = 0; 

  // check to make sure the timer was stopped and that the timer is not running
  if( timerWasStopped )
  {
    // throw exception
    throw logic_error( "The timer was not stopped for a measurement - cannot calculate elapsed time." );
  }

  // convert the start and stop times to doubles (microseconds)
    // start time
    startDouble = double( (beginTime.tv_sec * 1000000) + beginTime.tv_usec );

    // stop time
    stopDouble = double( (duration.tv_sec * 1000000) + duration.tv_usec );

  // perform calculation
  elapsedTime = stopDouble - startDouble;

  // convert to full seconds
  elapsedTime /= 1000000;

  // return elapsed time
  return elapsedTime;
}


