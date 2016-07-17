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


int solveRushHour( int numCars, GameState initial,
                     queue< GameState > statesToEvaluate,
                     map< string, int > observedStates );

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
================================================================================
                   MAIN PROGRAM IMPLEMENTATION
================================================================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

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
        cout << "Scenario: " << scenarioNumber << " requires "
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
