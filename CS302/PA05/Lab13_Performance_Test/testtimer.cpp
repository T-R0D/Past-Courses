#include "Timer.h"
#include  <iostream> 
#include  <stddef.h> 
#include  <sys/time.h> 
#include  <cstdio>

double getElapsed(timeval & t1)
{
    double ret;
    timeval t2;
    gettimeofday(&t2, NULL);
    ret = t2.tv_usec + t2.tv_sec * 1000000;
    ret -= t1.tv_usec + t1.tv_sec * 1000000;
    ret /= 1000000.0;

    return ret;
}

int main(int argc, char ** argv)
{
    double d, d1;
    Timer timer;
    timeval start;
    std::cin  >> d;
    timer.start();
    gettimeofday(&start, NULL);
    while ((d1 = getElapsed(start))  <  d)
       {
       }
    timer.stop();

    printf("Slept for %.2lf seconds. Measured %.2lf seconds.\n", d1, timer.getElapsedTime());

    return 0;
}
