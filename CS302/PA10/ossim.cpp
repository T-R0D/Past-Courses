/**
    @file ossim.cpp

    @author Terence Henriod

    Lab 10: Operating System Scheduling Simulator

    @brief A shell program that utilizes the PriorityQueue ADT to simulatean
           operating system's use of a priority queue to regulate access to a
           system resource (printer, disk, etc.).

    @version Original Code 1.00 (11/8/2013) - T. Henriod
*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include "PriorityQueue.cpp"
using namespace std;


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   GLOBAL CONSTANTS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

// none


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   STRUCT / CLASS DECLARATIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
@struct TaskData

A struct used to simulate a task an operating system might have to schedule.
Contains members that contain data pertaining to priority and arrival time.
Supports a getPriority method to maintain compatibility with the Heap ADT.
*/
struct TaskData
{
  int getPriority() const { return priority; }     // Needed by the heap.
  int getArrival() const { return arrived; }       // Needed by PQ Comparator
  int priority;                // Task's priority
  int arrived;                 // Time when task was enqueued
};

/**
@class PriorityArrivalCompare

A class that acts as a function to compare both priority and arrival
time to create a "fair queue."

CURRENTLY NOT IN USE.

*/
template < typename TaskType = TaskData >
class PriorityArrivalCompare
{
 public:
  /*---   Overloded Function Operator   ---*/
  bool operator()( const TaskType& first, const TaskData& second )
  {
    // variables
    bool firstIsLess = false;

    // case: first has a lower priority
    if( ( first.getPriority() < second.getPriority() ) &&
         ( first.getArrival() >= second.getArrival() ) )
    {
      // the first one compares to be less
      firstIsLess = true;
    }
    // case: the priorities are the same, but the first one arrived later
    else if( ( first.getPriority() == second.getPriority() ) &&
             ( first.getArrival() >= second.getArrival() ) )
    {
      // the first one compares to be less
      firstIsLess = true;
    }

    // return the truth of the first item being less
    return firstIsLess;
  };
};

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   GLOBAL FUNCTION PROTOTYPES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

TaskData addTask( int arrivalTime, int numPriorities );


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   MAIN PROGRAM IMPLEMENTATION
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
main

The driving function of the program. A number of differing priority levels and a
length of time to run the simulation are prompted for, and then the simulation
is run, randomly adding 0, 1, or 2 tasks of random priority each time. Each time
a task is "processed," its summary statistics are reported.

@return 0   This return value indicates error free execution.

@pre
-# None.

@post
-# A operating system task scheduling simulation will have been run.

@code
@endcode
*/
int main()
{
  // variables
  PriorityQueue< TaskData, int > taskPQ;  
                               // Priority queue of tasks
  TaskData task;               // Task
  int simLength;               // Length of simulation (minutes)
  int minute;                  // Current minute
  int numPtyLevels;            // Number of priority levels
  int numArrivals;             // Number of new tasks arriving
  int j;                       // Loop counter

  // Seed the random number generator
  srand( 7 );   // 7 was agreed upon in class

  cout << endl << "Enter the number of priority levels : ";
  cin >> numPtyLevels;

  cout << "Enter the length of time to run the simulator : ";
  cin >> simLength;

  // run the simulation for the given amount of time
  for( minute = 0 ; minute < simLength ; minute++ )
  {
    // Dequeue the first task in the queue (if any)
    if( !taskPQ.isEmpty() )
    {
      // get a task
      task = taskPQ.dequeue();

      // perform the task (output its information)
      cout << "At " << minute << " dequeued : " << task.getPriority() << ' '
           << task.getArrival() << ' ' << ( minute - task.getArrival() )
           << endl;
    }

    // Determine the number of new tasks and add them to the queue
    numArrivals = ( rand() % 4 );

    cout << "numArrivals: " << numArrivals << endl;

    // if an appropriate random number was generated, add tasks to the queue
    if( ( 1 <= numArrivals ) && ( numArrivals <= 2 ) )
    {
      // add the specified number of tasks to the queue
      for( j = 0; j < numArrivals; j++ )
      {
        // add a task
        taskPQ.enqueue( addTask( minute, numPtyLevels ) );
      }
    }
  }

  // end simulation program
  return 0;
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   GLOBAL FUNCTION IMPEMENTATIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
addTask

Generates a new task for the simulation using the arrival time of the task and a
random priority level.

@param arrivalTime     The time the task is simulated to arrive into the
                       PriorityQueue.
@param numPriorities   The number of different priority levels being used in the
                       simulation. 

@return newTask   A new task of type TaskData. Contains information relevant to
                  the simulation.

@pre
-# Ideally, int arrivalTime and numPriorities should be passed logical values.

@post
-# A new task will be generated with a simulation arrival time and a random
   priority.

@code
@endcode
*/
TaskData addTask( int arrivalTime, int numPriorities )
{
  // variables
  TaskData newTask;

  // give the task an arrival time
  newTask.arrived = arrivalTime;

  // give the task a priority
  newTask.priority = ( rand() % numPriorities );

  // return the new task
  return newTask;
}

