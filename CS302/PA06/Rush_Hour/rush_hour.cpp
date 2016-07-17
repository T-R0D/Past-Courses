////////////////////////////////////////////////////////////////////////////////
// 
//  Title:       
//  Created By:  Terence Henriod
//  Reviewed By:
//  Course:      CS 302: Data Structures
//
//  Summary:     
// 
//  Last Modified: 14:40   10/8/2013
//
////////////////////////////////////////////////////////////////////////////////

/**
@file rush_hour.cpp

@author Terence Henriod

@brief This program will solve a given rush hour puzzle using a bounded "try
every possible strategy" method via recursion. The reult that is found is the
least amount of necessary moves.

@version Original Code 1.00 (10/8/2013) - Terence Henriod
*/



//============================================================================//
//= Header Files =============================================================//
//============================================================================//

#include <cstdlib>
#include <iostream>
#include <cstring>
using namespace std;


//============================================================================//
//= Global Constants =========================================================//
//============================================================================//

const int BOARD_SIZE = 6;
const int CAR_LEN = 2;
const int TRUCK_LEN = 3;
const char HORIZONTAL = 'H';
const char VERTICAL = 'V';
const int MAX_MOVES = 10;
const bool FORWARD = true;
const bool BACKWARD = false;


//============================================================================//
//= Forward Declarations / User Defined Types ================================//
//============================================================================//

struct vehicle
{
  int number;
  int headX;
  int headY;
  char orientation;
  int length;
};

//============================================================================//
//= Function Prototypes ======================================================//
//============================================================================//

/**
This function will either return the signal to stop processing rush hour
puzzles or it wll set up the board in prepaation for the next puzzle and return
the number of cars that will be in the puzzle.

@param board[BOARD_SIZE][BOARD_SIZE]   a 2-D array that will represent the game
states as the puzzle is solved
@param carList   a pointer used to allocate/reference memory for an array that
will hold the vehicle structs that have the vehicle information for the puzzle
@return numCars   the number of cars the puzzle will have

@code
@endcode

@pre
-# vehicle* carList is a NULL pointer to an empty list
-# int board[BOARD_SIZE][BOARD_SIZE] may be in any state
@post
-# vehicle* carList will be allocated and will contain all the cars necessary
for soving the puzzle (if there is a puzzle to be soved)
-# int board[BOARD_SIZE][BOARD_SIZE] will contain the placement data of all
vehicles, with -1 representing unoccupied space (if there is a puzzle to be
solved)
@detail @bAlgorithm 
-# the number of cars for the next puzzle to be solved is read in
-# if there are no cars (no puzzle), this number is returned
-# if there are cars, the board is cleared (all spaces = -1) and car placement
occurs
-# each car's data is read in, stored in carList, an it is represented on the
board
*/
int setupBoard( int board[BOARD_SIZE][BOARD_SIZE], vehicle*& carList );

/**
This function will solve a rush hour puzzle using a semi-exhaustive "try all
possibilities" approach that is bounded by the lowest number of moves from a
previously successful strategy or predetermined upper limit.

@param board[BOARD_SIZE][BOARD_SIZE]   a 2-D array that will represent the game
states as the puzzle is solved
@param carList   a pointer used to allocate/reference memory for an array that
will hold the vehicle structs that have the vehicle information for the puzzle
@param numCars   the number of cars the puzzle will have
@param currentBound   the highest number of moves any strategy we will test
may use. decreases as new, more efficient, but successful strategies are found.
@param movesDone   the count of moves currently executed for a given strategy
path 
@return currentBound the highest number of moves any strategy we will test
may use. decreases as new, more efficient, but successful strategies are found.

@code
@endcode

@pre
-# int board[BOARD_SIZE][BOARD_SIZE] is initialized to an appropriate puzzle
state
-# vehicle* carList is a NULL pointer to a suitable list to fit the current
puzzle
-# int numCars matches to the size of carList
-# int movesDone is >= 0

@post
-# int currentBound will represent at least the best number of moves required
   to solve the current strategy path and at most the 
-# int board[BOARD_SIZE][BOARD_SIZE] will contain the placement data of all
vehicles, with -1 representing unoccupied space (if there is a puzzle to be
solved)
@detail @bAlgorithm 
-# first checks for an end of a strategy path either to see if the bound is
reached or if the game has been won. if the game has been won, if the score
is a new low score, the bound is updated
-# if the end of the strategy path has not been reached, each car is moved
-# first the car is moved forward and the function is called again to solve
   this new game state. after the reursive calls have resolved, the move is
   undone the..
-# next the car is moved backward and the function is called again to solve
   this new game state. after the reursive calls have resolved, the move is
   undone.
-# the lowest score is returned
*/
int solveJam( int board[BOARD_SIZE][BOARD_SIZE], vehicle*& carList,
              int numCars, int& currentBound, int movesDone );

/**
This function checks for the end of a strategy path either by comparing the
number of moves done in the current strategy, or by checking to see if the
0 car has escaped the jam

@param board[BOARD_SIZE][BOARD_SIZE]   a 2-D array that will represent the game
states as the puzzle is solved
@param carList   a pointer used to allocate/reference memory for an array that
will hold the vehicle structs that have the vehicle information for the puzzle
@param currentBound   the highest number of moves any strategy we will test
may use. decreases as new, more efficient, but successful strategies are found.
@param movesDone   the count of moves currently executed for a given strategy
path
@param car   the 0 car is passed so that the last space of the row car 0
occupies contains car 0
@return endFound   indicates the success of finding the end of the strategy
path

@code
@endcode

@pre
-# int board[BOARD_SIZE][BOARD_SIZE] is initialized to an appropriate puzzle
state
-# int movesDone is >= 0
-# int currentBound is >= movesDone
-# vehicle car is a valid car

@post
-# int currentBound will be >= movesDone. it will not be updated, this
occurs outside this function if necessary

@detail @bAlgorithm 
-# first compares movesDone to currentBound, if movesDone is >= currentBound
the end has been found
-# if car 0 made it to the rightmost column (it has escaped), the end has
been found
*/
bool checkForEnd( int board[BOARD_SIZE][BOARD_SIZE], int movesDone,
                  int currentBound, vehicle& car );

/**
This function moves a vehicle in the given direction (if possible) and updates
the vehicle's position on the game board.

@param board[BOARD_SIZE][BOARD_SIZE]   a 2-D array that will represent the game
states as the puzzle is solved
@param car   the car that is to be moved
@param direction   a flag to indicate which direction the car will be moved
@return moveSuccess   indicates whether or not the attempted move was completed

@code
@endcode

@pre
-# int board[BOARD_SIZE][BOARD_SIZE] is initialized to an appropriate puzzle
state
-# vehicle* car is a valid car

@post 
-# int board[BOARD_SIZE][BOARD_SIZE] will contain the placement data of all
vehicles, with -1 representing unoccupied space (if there is a puzzle to be
solved), including the just moved car
-# if the vehicle was not successfully moved, the board and car's states
remain unchanged
@detail @bAlgorithm 
-# first it is decided which direction the car is being moved
-# then it is determined if the space the vehicle would occupy will be valid
-# next a check to see that the intended position is clear to be moved into
-# if all of the above checks pass, the vehicle is then placed on the board in
its new location, and any previous/no longer valid spaces are cleared
*/
bool moveVehicle( int board[BOARD_SIZE][BOARD_SIZE], vehicle& car,
                  bool direction );

/**
Places a car on the given board, regardless of what may be occupying the spaces.

@param board[BOARD_SIZE][BOARD_SIZE]   a 2-D array that will represent the game
states as the puzzle is solved
@param car   the car that is to be moved
@param newX   the new x coordinate for the "head" of the car "vector"
@param newY   the new y coordinate for the "head" of the car "vector"
@return moveSuccess   indicates whether or not the attempted placement was
completed (in this version, placement is always successful)

@code
@endcode

@pre
-# int board[BOARD_SIZE][BOARD_SIZE] is initialized to an appropriate puzzle
state
-# vehicle* car is a valid/initialized car

@post 
-# int board[BOARD_SIZE][BOARD_SIZE] will contain the placement data of all
vehicles, with -1 representing unoccupied space (if there is a puzzle to be
solved), including the recently placed car
-# the vehicle will be placed wherever the function is told to place it
(BEWARE seg-faults!)
@detail @bAlgorithm 
-# the car's "head" is updated to it's new coordinates
-# placement on the board begins with the head coordinates
-# then the tail is placed based on the length of the car into the following
   spaces on the board (to the right for horizontal, down for vertical)
*/
bool placeCar( int board[BOARD_SIZE][BOARD_SIZE], vehicle& car,
               int newX, int newY );


//============================================================================//
//= Main Function ============================================================//
//============================================================================//

int main()
{
  // variables
  bool keepSolving = true;
  int board[BOARD_SIZE][BOARD_SIZE];
  vehicle* carList = NULL;
  int lowScore = MAX_MOVES;
  int numCars = 0;
  int counter = 1;

  // solve all puzzles that come
  while( keepSolving )
  {
    // setup the board (or receive signal to stop)
    numCars = setupBoard( board, carList ); 

    // case: signal to quit was given
    if( numCars == 0 )
    {
      // stop solving
      keepSolving = false;
      break;  
    }

    // reset the lowScore for the setup
    lowScore = MAX_MOVES;

    // solve the puzzle
    lowScore = solveJam( board, carList, numCars, lowScore, 0 );

    // output the lowest number of moves required to win
    cout << "Scenario " << counter << " requires " << lowScore << " moves" << endl;

    // destruct vehicle list
    delete [] carList;

    // increment the puzzle counter
    counter ++;
  }

  // return 0
  return 0;
}


//============================================================================//
//= Function Implementation ==================================================//
//============================================================================//

int setupBoard( int board[BOARD_SIZE][BOARD_SIZE], vehicle*& carList )
{
  // variables
  int numCars = 0;
  int carNdx = 0;
  int boardX = 0;
  int boardY = 0;

  // read number of cars (also signal to quit)
  cin >> numCars;

  // if there is a puzzle to solve, prepare
  if( numCars != 0 )
  {
    // initialize board
    for( boardY = (BOARD_SIZE - 1); boardY >= 0; boardY -- )   // optimization practice....
    {
      for(  boardX = (BOARD_SIZE - 1); boardX >= 0; boardX -- )
      {
        board[boardY][boardX] = -1;
      }
    }

    // allocate memory for car list
    carList = new vehicle [numCars];

    // read in all cars
    for( carNdx = 0; carNdx < numCars; carNdx ++ )
    {
      // read in car data
        // number
        carList[carNdx].number = carNdx;

        // length
        cin >> carList[carNdx].length;

        // orientation
        cin >> carList[carNdx].orientation;

        // y coordinate of head
        cin >> carList[carNdx].headY;

        // x coordinate of head
        cin >> carList[carNdx].headX;

      // place car on board
      placeCar( board, carList[carNdx], carList[carNdx].headX,
                carList[carNdx].headY );   
    }
  }

  // return the continue solving signal
  return numCars;
}


int solveJam( int board[BOARD_SIZE][BOARD_SIZE], vehicle*& carList,
              int numCars, int& currentBound, int movesDone )
{
  // variables
  int carNdx = 0;

  // case: we have found the end of the strategy tree (win or many moves)
  if( checkForEnd( board, movesDone, currentBound, carList[0] ) )
  {
    // case: the number of moves required was a record
    if( movesDone < currentBound )
    {
    currentBound = movesDone;
    }
  }
  // otherwise, continue solving
  else
  {
    // attempt to move all vehicles in all directions
    for( carNdx = 0; carNdx < numCars; carNdx ++ )
    {
      // attempt forward move
      if( moveVehicle( board, carList[carNdx], FORWARD ) )
      {
        // update move count
        movesDone ++;

        // solve puzzle with this game state
        solveJam( board, carList, numCars, currentBound, movesDone );

        // reset the game state to the one that was passed in
        movesDone --;
        moveVehicle( board, carList[carNdx], BACKWARD );
      }

     // attempt backward move
      if( moveVehicle( board, carList[carNdx], BACKWARD ) )
      {
        // update move count
        movesDone ++;

        // solve puzzle with this game state
        solveJam( board, carList, numCars, currentBound, movesDone );

        // reset the game state to the one that was passed in
        movesDone --;
        moveVehicle( board, carList[carNdx], FORWARD );
      }
    }
  }

  // return the lowest number of moves done
  return currentBound;
}


bool checkForEnd( int board[BOARD_SIZE][BOARD_SIZE], int movesDone, int currentBound, vehicle& car )
{
  // variables
  int xCoord = (BOARD_SIZE - 1);
  int yCoord = 0;
  bool endFound = false;

  // case: the number of moves performed is equal to the bound
  if( movesDone >= currentBound )
  {
    // end of this strategy path is found
    endFound = true;
  }
  // otherwise, check for your car escaping the jam
  else
  {
    // check the last column of the board
    if( board[car.headY][BOARD_SIZE - 1] == 0)
    {
      // the game has been won, end is found
      endFound = true;
    }
  }
 
  // return end status
  return endFound;
}


bool moveVehicle( int board[BOARD_SIZE][BOARD_SIZE], vehicle& car, bool direction )
{
  // variables
  bool moveSuccess = false;

  // case: moving forward
  if( direction == FORWARD )
  {
    // case: car is vertical
    if( car.orientation == VERTICAL )
    {
      // case: new index is valid
      if( (car.headY - 1) >= 0 )
      {
        // case: new index is empty
        if( board[car.headY - 1][car.headX] < 0 )
        {
           // perform move (modifies car head values)
           placeCar( board, car, car.headX, (car.headY - 1) );

           // clean up behind the car
           board[car.headY + car.length][car.headX] = -1;

           // update flag
           moveSuccess = true;
        }
      }
    }
    // case: car is horizontal
    else
    {
      // case: new index is valid
      if( (car.headX + car.length) < BOARD_SIZE )
      {
        // case: new index is empty
        if( board[car.headY][car.headX + car.length] < 0 )
        {
          // perform move (modifies car head values)
          placeCar( board, car, (car.headX + 1), car.headY );

          // clean up behind the car
          board[car.headY][car.headX - 1] = -1;

          // update flag
          moveSuccess = true;
        }
      }
    }
  }
  // case: moving backward
  else
  {
    // case: car is vertical
    if( car.orientation == VERTICAL )
    {
      // case: new index is valid
      if( (car.headY + car.length) < BOARD_SIZE )
      {
        // case: new index is empty
        if( board[car.headY + car.length][car.headX] < 0 )
        {
           // perform move (modifies car head values)
           placeCar( board, car, car.headX, (car.headY + 1) );

           // clean up behind the car
           board[car.headY - 1][car.headX] = -1;

           // update flag
           moveSuccess = true;
        }
      }
    }
    // case: car is horizontal
    else
    {
      // case: new index is valid
      if( (car.headX - 1) >= 0 )
      {
        // case: new index is empty
        if( board[car.headY][car.headX - 1] < 0 )
        {
           // perform move (modifies car head values)
           placeCar( board, car, (car.headX - 1), car.headY );

           // clean up behind the car
           board[car.headY][car.headX + car.length] = -1;

           // update flag
           moveSuccess = true;
        }
      }
    }
  }

  // return the movement's success
  return moveSuccess;
}


bool placeCar( int board[BOARD_SIZE][BOARD_SIZE], vehicle& car, int newX, int newY )
{
  // variables
  int counter = 0;
  bool successful = true;   // assume operation was sucessfull, may be used later

  // update head and tail values
  car.headX = newX;
  car.headY = newY;

  // star at head, place car in grid section by section
  for( counter = 0; counter < car.length; counter ++ )
  {
    board[newY][newX] = car.number;

    // case: orientation is horizontal
    if( car.orientation == HORIZONTAL )
    {
      // incremnt the horizontal coordinate
      newX ++;
    }
    // case: orientation is vertical
    else
    {
      // increment the vertical coordinate
      newY ++;
    }
  }

  // return success state
  return successful;
}


