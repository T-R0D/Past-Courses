/**

@file Timer.h

@author Terence Henriod

*/

#ifndef TIMER_H
#define TIMER_H

#include <ctime>
#include <stdexcept>
#include <iostream>
#include <sys/time.h>

using namespace std;


class Timer
{
  public:
    Timer();
    void start() throw (runtime_error);
    void stop() throw (logic_error, runtime_error);
    double getElapsedTime() const throw (logic_error);

  private:
    timeval beginTime;
    timeval duration;
    bool timerWasStarted;
    bool timerWasStopped;
};

#endif	// ifndef TIMER_H
