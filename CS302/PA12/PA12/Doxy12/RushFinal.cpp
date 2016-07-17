/**
    @file RushFinal.cpp

    @author Terence Henriod

    Rush Hour: Breadth First Search

    @brief The driver program for a fast Rush Hour Solver

    @version Original Code 1.00 (10/29/2013) - T. Henriod
*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>
#include <queue>
#include <map>
#include "GameState.h"
using namespace std;


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   GLOBAL CONSTANTS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

const int kUnsolvable = -8;


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   USER-DEFINED TYPES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   FUNCTION PROTOTYPES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
solveRushHour

Finds the lowest number of moves required to find the solution to a Rush Hour
puzzle. If a solution cannot be found, an error code is returned.

@param numCars            The number of cars to be used in a given puzzle.
@param initial            The initial board for a given problem.
@param statesToEvaluate   A queue that will be used in a breadth first search to
                          contain every game state that will be considered.
@param observedStates     A map used to contain an identifying key string that
                          contains vehicle list data used to check to see if a
                          state has already been considered.

@return numMoves   The minimum number of moves required to solve the given
                   puzzle.

@pre
-# A valid number of cars that corresponds precisely to the data waiting to be
   read in is required.
-# statesToEvaluate must be an empty queue.
-# observedStates must be empty as well.
-# The initial state will be representative of the starting condition for the
   puzzle to be solved.

@post
-# A number of moves corresponding to either the number of moves needed to solve
   a given puzzle or an error code for an unsolvable puzzle is returned.

@b@Algorithm
-# A breadth first search is implemented.
-# GameStates are added to the queue and their representative strings are added
   to the queue if they are newly encountered.
-# States are iteratively popped off the queue and checked to see if they are a
   winning state.
-# If they are not, all possible GameStates based on legal moves are generated
   and stored in the queue. Their representative strings and the number of moves
   used up to that point are stored for future reference.
-# Should the queue ever become empty, this indicates that all possible moves
   have been attempted and the puzzle is unsolvable.

@code
@endcode
*/
int solveRushHour( int numCars, GameState initial,
                     queue< GameState > statesToEvaluate,
                     map< string, int > observedStates );

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
================================================================================
                   MAIN PROGRAM IMPLEMENTATION
================================================================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/**
main

The driver of a Rush Hour solver. Reports the number of moves required to solve
given rush hour puzzles.

@return programSuccess   The success/error code of the program's execution.

@pre
-# There is valid data waiting to be provided to the program.

@post
-# A number of Rush Hour puzzles will have been solved

@code
@endcode
*/
int main()
{
  // variables
  bool keepSolving = true;
  int scenarioNumber = 1;
  int numCars = 0;
  int numMoves = 0;
  GameState initialState;
  queue< GameState > gameStates;
  map< string, int > observedStates;

  // continue processing input until the signal to stop is given
  while( keepSolving )
  {
    // read in the problem size
    cin >> numCars;

    // case: there is a problem to solve
    if( numCars > 0 )
    {
      // read in the problem setup
      initialState.clear();
      initialState.setNumCars( numCars );
      initialState.readIn();

      // attempt to find a solution
      numMoves = solveRushHour( numCars, initialState, gameStates,
                                observedStates );

      // output the solution result
      // case: the problem has a solution
      if( numMoves >= 0 )
      {
        // report this
        cout << "Scenario " << scenarioNumber << " requires "
             << numMoves << " moves" << endl;
      }
      // case: the problem was not solvable
      else if( numMoves == kUnsolvable )
      {
        // report this
        cout << "Scenario: " << scenarioNumber
             << " is not solvable" << endl; 
      }

      // setup for a new problem
      scenarioNumber++;
      observedStates.clear();
      while( !gameStates.empty() )
      {
        gameStates.pop();
      }
    }
    // case: we are done solving
    else
    {
      // set signal to stop solving
      keepSolving = false;
    }
  }

  // return 0 for successful program execution
  return 0;
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   FUNCTION IMPLEMENTATIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

int solveRushHour( int numCars, GameState initial,
                     queue< GameState > statesToEvaluate,
                     map< string, int > observedStates )
{
  // variables
  int numMoves = 0;
  int carNum = 0;
  GameState currentWork;
  string temp = initial.stringify();

  // prime the search loop
  statesToEvaluate.push( initial );
  observedStates.insert( pair< string, int >( temp, 0 ) );

  // search until a solution is found or there are no possibilities left
  while( !statesToEvaluate.empty() )
  {
    // pull a game state out of the queue
    currentWork = statesToEvaluate.front();
    statesToEvaluate.pop();

    // retrieve the number of moves up to that point
    temp = currentWork.stringify();
    numMoves = (*observedStates.find( temp )).second;

    // check to see if that state is a winning one
    if( currentWork.isWin() )
    {
      // if so, return the number of moves it took to find it the solution
      return numMoves;
    }

    // otherwise, make another round of moves
    numMoves++;
    for( carNum = 0; carNum < numCars; carNum++ )
    {
           // case: a forward movement attempt was successful
      if( currentWork.move( carNum, FORWARD ) )
      {
        // case: the new game state is not a winning one and it has not been
        //       encountered yet
        if( observedStates.find( currentWork.stringify() ) ==
                 observedStates.end() )
        {
          // enqueue the game state, add it to the map
          statesToEvaluate.push( currentWork );
          temp = currentWork.stringify();
          observedStates.insert( pair<string, int>( temp, numMoves ) );
        }
        // case: the new game state is not new nor a solution
          // do nothing

        // reset the game state for the next move
        currentWork.move( carNum, BACKWARD );
      }
      // case: a backward movement attempt is successful
      if( currentWork.move( carNum, BACKWARD ) )
      {
        // case: the new game state is not a winning one and it has not been
        //       encountered yet
        if( observedStates.find( currentWork.stringify() ) ==
                 observedStates.end() )
        {
          // enqueue the game state, add it to the map
          statesToEvaluate.push( currentWork );
          temp = currentWork.stringify();
          observedStates.insert( pair<string, int>( temp, numMoves ) );
        }
        // case: the new game state is not new nor a solution
          // do nothing

        // reset the car's position
        currentWork.move( carNum, FORWARD );
      }
    }
  }

  // if the search fails, return a designated error code
  return kUnsolvable;
}
