/**
    @file .cpp

    @author Terence Henriod

    Project Name

    @brief Class implementations declarations for...

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
  int numRead = 0;
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
  bool solutionFound = false;
  GameState currentWork;
  string temp = initial.stringify();

  statesToEvaluate.push( initial );
  observedStates.insert( pair< string, int >( temp, 0 ) );

  while( !statesToEvaluate.empty() )
  {
    currentWork = statesToEvaluate.front();
    statesToEvaluate.pop();
    temp = currentWork.stringify();
    numMoves = (*observedStates.find( temp )).second;
    if( currentWork.isWin() )
    {
      return numMoves;
    }
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

  // return the number of moves required for a win
  return kUnsolvable;

/*
  // prime the search for a solution
  statesToEvaluate.push( initial );
  observedStates.insert( pair< string, int >( temp, 0 ) );

  // check the initial game state to see if it is a winning one
  solutionFound = initial.isWin();

  // continue searching for a win until one is found
  while( !solutionFound && !statesToEvaluate.empty() )
  {
    // retrieve the move count we will be working with, then increment
    // to represent the current set of moves
    temp = statesToEvaluate.front().stringify();
    numMoves = (*observedStates.find( temp )).second;

    // generate all possible game states from this one
    for( carNum = 0; (carNum < numCars) && !solutionFound; carNum++ )
    {
      // case: a forward movement attempt was successful
      if( statesToEvaluate.front().move( carNum, FORWARD ) )
      {
        // case: the new game state is a winning one
        if( statesToEvaluate.front().isWin() )
        {
          // indicate such
          solutionFound = true;
        }
        // case: the new game state is not a winning one and it has not been
        //       encountered yet
        else if( observedStates.find( statesToEvaluate.front().stringify() ) ==
                 observedStates.end() )
        {
          // enqueue the game state, add it to the map
          statesToEvaluate.push( statesToEvaluate.front() );
          temp = statesToEvaluate.front().stringify();
          observedStates.insert( pair<string, int>( temp, numMoves ) );
        }
        // case: the new game state is not new nor a solution
          // do nothing

        // reset the game state for the next move
        statesToEvaluate.front().move( carNum, BACKWARD );
      }
      // case: a backward movement attempt is successful
      if( statesToEvaluate.front().move( carNum, BACKWARD ) )
      {
        // case: the new game state is a winning one
        if( statesToEvaluate.front().isWin() )
        {
          // indicate such
          solutionFound = true;
        }
        // case: the new game state is not a winning one and it has not been
        //       encountered yet
        else if( observedStates.find( statesToEvaluate.front().stringify() ) ==
                 observedStates.end() )
        {
          // enqueue the game state, add it to the map
          statesToEvaluate.push( statesToEvaluate.front() );
          temp = statesToEvaluate.front().stringify();
          observedStates.insert( pair<string, int>( temp, numMoves ) );
        }
        // case: the new game state is not new nor a solution
          // do nothing

        // move the car back
        statesToEvaluate.front().move( carNum, FORWARD );
      }
    }

    // processing this game state has concluded, throw it out
    statesToEvaluate.pop();

    // if the queue became empty, this indicates that the puzzle is
    // not solvable
    if( statesToEvaluate.empty() && !solutionFound )
    {
      // make numMoves indicate such
      numMoves = kUnsolvable;
    }
  }*/
}
