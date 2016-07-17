#include <iostream>
#include "Timer.h"

const int ONE_MIL = 1000000;
const int ONE_BIL = 1000000000;
const int FIVE_MIL = 5000000;

double res_test( const int numTrials );

int main ()
{
  // try all tests, output results
  cout << "With one million trials: " << res_test( ONE_MIL ) << endl;
  cout << "With five million trials: " << res_test( FIVE_MIL ) << endl;
  cout << "With one billion trials: " << res_test( ONE_BIL ) << endl;

  // return 0
  return 0;
}


/**
The default constructor for the timer class.
This constructor initializes all data members to 0 or
equivalent values.

@code
@endcode
*/
double res_test( const int numTrials )
{
  // vars
  int ndx = 0;
  double tests[numTrials];
  Timer timer;
  double total = 0;
  double average = 0;

  // get times
  for( ndx = 0; ndx < numTrials; ndx ++ )
  {
  timer.start();
  timer.stop();
  tests[ndx] = timer.getElapsedTime();
  }

  // find average
  for( ndx = 0; ndx < numTrials; ndx ++ )
  {
    // add em up
    total += tests[ndx];
  }

  average = (total / numTrials);

  // return average
  return average;
}
