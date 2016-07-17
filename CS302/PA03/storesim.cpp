////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      
//  Created By:
//  Reviewed By:
//  Course:     
//
//  Summary:    
// 
//  Last Modified: 
//
////////////////////////////////////////////////////////////////////////////////

//============================================================================//
//= Header Files =============================================================//
//============================================================================//

// common headers/namespaces
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <ctime>
using namespace std;

#include "config.h"

// Use which ever implementation is currently configured.
#if LAB7_TEST1
  #include "QueueLinked.cpp"
#else
  #include "QueueArray.cpp"
#endif


//============================================================================//
//= Global Constants =========================================================//
//============================================================================//


//============================================================================//
//= Function Prototypes ======================================================//
//============================================================================//



//============================================================================//
//= Main Program Definition ==================================================//
//============================================================================//
int main() {
  // variables
    // the line (queue) of customers containing the "minute" number that
    // the customer entered the line
    #if LAB7_TEST1
      QueueLinked<int> the_line;
    #else
      QueueArray<int> the_line;
    #endif
    // time vars
    int simulation_run_time = 0;
    int time = 0;
    int entry_time = -8;  // garbage clearing/debugging
    int wait_time = 0;

    // summary statistics
    int total_served = 0;
    int total_wait = 0;
    int max_wait = 0;

    // utitlity vars
    int random_decision = 0;

  // seed RNG
  srand( 7 );   // 7 was agreed on by class to standardize results

  // collect the desired simulation run time
  cout << endl << "Enter the length of time to run the simulator : ";
  cin >> simulation_run_time;

  // implement simulation
  while( time < simulation_run_time ) {
    // increment time indicator
    ++ time;

    // if there are any customers in line, service one of them
    if( !the_line.isEmpty() ) {
      // "service" customer, get them out of line
      entry_time = the_line.dequeue();

      // compute that customer's wait time
      wait_time = (time - entry_time);

      // update statistics
        // update total customers served
        ++ total_served;

        // update combined waiting time
        total_wait += wait_time;

        // update maximum time waited if necessary
        if( wait_time > max_wait ) {
          // there is a new maximum waiting time
          max_wait = wait_time;
        }

        // DO NOT compute average wait time, this will be done at conclusion
     }

    // use RNG to decide if more customers come
      // get a random number
      random_decision = (rand() % 4);

      // make decision based on RNG result
      switch( random_decision ) {
        // case: add one customer
        case 1:
          // add the customer
          the_line.enqueue(time);
          break;

        // case: add two customers
        case 2:
          // add both customers
          the_line.enqueue(time);
          the_line.enqueue(time);
          break;

        // cases where no customers are added or we have unexpected result
        case 0:
        case 3:
        default:
          // no customers came, do nothing
          break; 
      }
  }

  // Print out simulation results
  cout << endl;
  cout << "Customers served : " << total_served << endl;
  cout << "Average wait     : " << setprecision(2)
       << double(total_wait) / total_served << endl;
  cout << "Longest wait     : " << max_wait << endl;

  // return 0
  return 0;
}


//============================================================================//
//= Function Implementation ==================================================//
//============================================================================//


