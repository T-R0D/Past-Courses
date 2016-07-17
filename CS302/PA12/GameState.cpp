/**
    @file GameState.cpp

    @author Terence Henriod

    Project Name

    @brief Class implementations declarations for...

    @version Original Code 1.00 (10/29/2013) - T. Henriod
*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

// Class Declaration
#include "GameState.h"

// Other Dependencies
#include <cassert>
#include <iostream>


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
================================================================================
                   CLASS FUNCTION IMPLEMENTATIONS
================================================================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   CONSTRUCTOR(S) / DESTRUCTOR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
GameState

The default constructor for a game state. Constructs and initilizes an empty
GameState.

@pre
-# The GameState object is given an appropriate identifier.

@post
-# A new, empty GameState will be initialized.

@code
@endcode
*/
GameState::GameState()
{
  // initialize data members
  clear(); // initializes the board and tosses the cars

  // no return - constructor
}

/**
GameState

The copy constructor for a game state. Constructs and initilizes a GameState
equivalent to the given parameter GameState other.

@param   other   A previously created GameState. Will be cloned into *this.

@pre
-# GameState other is a valid instantiation of a GameState
-# The GameState object is given an appropriate identifier.

@post
-# *this will be an equivalent clone of other.

@code
@endcode
*/
GameState::GameState( const GameState& other )
{
  // initialize data members
    // none to initialize

  // call the overloaded assignment operator
  *this = other;

  // no return - constructor
}

/**
operator=

The assignment operator for the GameState class. Clones the given parameter
GameState other into *this.

@param   other   A valid GameState to be cloned into *this.

@return   *this

@pre
-# GameState other is a valid instantiation of a GameState
-# The GameState object is a valid instantiation of a GameState.

@post
-# *this will be an equivalent clone of other.

@code
@endcode
*/
GameState& GameState::operator=( const GameState& other )
{
  // variables
  int rndx = 0;
  int cndx = 0;

  // case: other is not the same object as *this
  if( this != &other )
  {
    // copy the game board of other
    // ( admittedly, an abuse of numCars )
    for( rndx = 0; rndx < kBoardSize; rndx++ )
    {
      for( cndx = 0; cndx < kBoardSize; cndx++ )
      {
        board[ rndx ][ cndx ] = other.board[ rndx ][ cndx ];
      }
    }

    // copy the car list
    for( numCars = 0; numCars < other.numCars; numCars++ )
    {
      carList[ numCars ] = other.carList[ numCars ];
    }
  }

  // return *this
  return *this;
}

/**
~GameState

The destructor for the GameState class. Currently doesn't do anything.

@pre
-# The GameState object is given an appropriate identifier.

@post
-# *this will be completely destructed.

@code
@endcode
*/
GameState::~GameState()
{
  // currently nothing to destruct
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   MUTATORS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
clear

Clears the GameState so that it is an empty one with no car data.

@pre
-# *this is a valid GameState object.

@post
-# *this will be a GameState with no cars and a clean board

@code
@endcode
*/
void GameState::clear()
{
  // variables
  int rndx = 0;
  int cndx = 0;

  // throw out all the cars
  numCars = 0;

  // clear the gameboard
  for( rndx = 0; rndx < kBoardSize; rndx++ )
  {
    for( cndx = 0; cndx < kBoardSize; cndx++ )
    {
      board[ rndx ][ cndx ] = '.';
    }
  }

  // no return - void
}

/**
readIn

A int description

@param

@return

@pre
-# 

@post
-# 

@detail @bAlgorithm
-# 

@exception

@code
@endcode
*/
bool GameState::move( const int whichCar, const bool direction )
{
/*  // variables
  bool result = false;
  int futureNdx = -8;

  // case: the car is a horizontal one
  if( carList[ whichCar ].orientation == kHorizontal )
  {
    // case: the car is moving forward
    if( direction == FORWARD )
    {
      // find the index of the space to be moved into
      futureNdx = ( ( carList[whichCar].yPos * kBoardSize ) +
                    ( carList[whichCar].xPos + carList[whichCar].length ) );

      // ensure there is room for the car to move forward (within a row)
      if( ( ( futureNdx % kBoardSize ) < kBoardSize ) &&
          ( ( futureNdx % kBoardSize ) >= ( carList[whichCar].length % kBoardSize ) ) )
      {
        // ensure the new space is open for the car to move into
        result = ( board[ futureNdx ] == '.' );

        // case: the movement can be carried out
        if( result )
        {
          // clean up the space the car will be leaving
          board[ futureNdx - carList[whichCar].length ] = '.';

          // update the car's postion
          carList[ whichCar ].xPos++;

          // place the car in the new position
          placeCar( whichCar );
        }
      }
    }
    // case: the car is moving backward
    else
    {
      // find the index of the space to be moved into
      futureNdx = ( ( carList[whichCar].yPos * kBoardSize ) +
                    ( carList[whichCar].xPos - 1) );

      // ensure there is room for the car to move backward (within a row)
      if( ( ( futureNdx % kBoardSize ) > 0 ) &&
          ( futureNdx <= ( kBoardSize - carList[whichCar].length ) ) )/////////////////////////////////////////////////////////////////////////
      {

        // ensure the new space is open for the car to move into
        result = ( board[ futureNdx ] == '.' );

        // case: the movement can be carried out
        if( result )
        {
          // clean up the space the car will be leaving
          board[ futureNdx + carList[whichCar].length ] = '.';

          // update the car's postion
          carList[whichCar].xPos--;

          // place the car in the new position
          placeCar( whichCar );
        }
      }
    }
  }
  // case: the car is a vertical one
  else
  {
    // case: the car is moving forward (down)
    if( direction == FORWARD )
    {
      // find the index that the car will be moving to
      futureNdx = ( ( ( carList[whichCar].yPos * kBoardSize ) +
                        carList[whichCar].length ) +
                    carList[whichCar].xPos );

      // ensure the car won't move off the board
      if( futureNdx < kBoardSizeSquared )
      {
        // ensure the new space is open for the car to move into
        result = ( board[ futureNdx ]
                 == '.' );

        // case: the movement can be carried out
        if( result )
        {
          // clean up the space the car will be leaving
          board[ futureNdx - carList[whichCar].length ] = '.';

          // update the car's postion
          carList[ whichCar ].yPos++;

          // place the car in the new position
          placeCar( whichCar );
        }
      }
    }
    // case: the car is moving backward (up)
    else
    {
      // find the index that the car will be moving to
      futureNdx = ( ( ( carList[whichCar].yPos - 1 ) * kBoardSize ) +
                    carList[whichCar].xPos );

      // ensure that the car won't be moving off the board
      if( futureNdx >= 0 )
      {
        // ensure the new space is open for the car to move into
        result = ( board[ futureNdx ] == '.' );

        // case: the movement can be carried out
        if( result )
        {
          // clean up the space the car will be leaving
          board[ ( ( carList[whichCar].yPos + carList[whichCar].length + 1 )
                    * kBoardSize ) + carList[whichCar].xPos ] = '.';

          // update the car's postion
          carList[ whichCar ].yPos--;

          // place the car in the new position
          placeCar( whichCar );
        }
      }
    }
*/
  // variables
  bool moveSuccess = false;

  // case: moving forward
  if( direction == FORWARD )
  {
    // case: car is vertical
    if( carList[whichCar].orientation == kVertical )
    {
      // case: new index is valid
      if( (carList[whichCar].yPos - 1) >= 0 )
      {
        // case: new index is empty
        if( board[carList[whichCar].yPos - 1][carList[whichCar].xPos] == '.' )
        {
           // perform move (modifies car head values)
           placeCar( whichCar, carList[whichCar].xPos, carList[whichCar].yPos - 1 );

           // clean up behind the car
           board[carList[whichCar].yPos + carList[whichCar].length]
                [carList[whichCar].xPos] = '.';

           // update flag
           moveSuccess = true;
        }
      }
    }
    // case: car is horizontal
    else
    {
      // case: new index is valid
      if( (carList[whichCar].xPos + carList[whichCar].length) < kBoardSize )
      {
        // case: new index is empty
        if( ( board[carList[whichCar].yPos][carList[whichCar].xPos +
              carList[whichCar].length] ) == '.' )
        {
          // perform move (modifies car head values)
          placeCar( whichCar, carList[whichCar].xPos + 1, carList[whichCar].yPos );

          // clean up behind the car
          board[carList[whichCar].yPos][carList[whichCar].xPos - 1] = '.';

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
    if( carList[whichCar].orientation == kVertical )
    {
      // case: new index is valid
      if( (carList[whichCar].yPos + carList[whichCar].length) < kBoardSize )
      {
        // case: new index is empty
        if( ( board[carList[whichCar].yPos +
              carList[whichCar].length][carList[whichCar].xPos] ) == '.' )
        {
           // perform move (modifies car head values)
           placeCar( whichCar, carList[whichCar].xPos, carList[whichCar].yPos + 1 );

           // clean up behind the car
           board[carList[whichCar].yPos - 1][carList[whichCar].xPos] = '.';

           // update flag
           moveSuccess = true;
        }
      }
    }
    // case: car is horizontal
    else
    {
      // case: new index is valid
      if( (carList[whichCar].xPos - 1) >= 0 )
      {
        // case: new index is empty
        if( board[carList[whichCar].yPos][carList[whichCar].xPos - 1] == '.' )
        {
           // perform move (modifies car head values)
           placeCar( whichCar, carList[whichCar].xPos - 1, carList[whichCar].yPos );

           // clean up behind the car
           board[ carList[whichCar].yPos ]
                [ carList[whichCar].xPos + carList[whichCar].length ] = '.';

           // update flag
           moveSuccess = true;
        }
      }
    }
  }

  // return the movement's success
  return moveSuccess;
}

/**
placeCar

A int description

@param

@return

@pre
-# 

@post
-# 

@detail @bAlgorithm
-# 

@exception

@code

@endcode
*/
bool GameState::placeCar( const int whichCar, int newX, int newY )
{
  // variables
  int counter = 0;
  bool successful = true;   // assume operation was sucessfull, may be used later

  // update head and tail values
  carList[whichCar].xPos = newX;
  carList[whichCar].yPos = newY;

  // start at head, place car in grid section by section
  for( counter = 0; counter < carList[whichCar].length; counter ++ )
  {
    board[newY][newX] = char( whichCar + '0' );

    // case: orientation is horizontal
    if( carList[whichCar].orientation == kHorizontal )
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

/*

  // variables
  int carX = carList[ whichCar ].xPos;
  int carY = carList[ whichCar ].yPos;
  int carLength = carList[ whichCar ].length;
  int ndx = 0;
  int counter = 0;

  // place the car at the designated location
  // case: the car is a horizontal one
  if( carList[ whichCar ].orientation == kHorizontal )
  {
    // place an X on the gameboard where the car belongs for each position
    for( ndx = ( ( carY * kBoardSize ) + carX );
         counter < carLength;
         ndx++, counter++ )
    {
      board[ ndx ] = char( whichCar + '0' );
    }
  }
  // case: the car is a vertical one
  else
  {
    // place an X on the gameboard where the car belongs for each position
    for( ndx = ( ( carY * kBoardSize ) + carX );
         counter < carLength;
         ndx += kBoardSize, counter++ )
    {
      board[ ndx ] = char( whichCar + '0' );
    }
  }*/

}

/**
readIn

Reads in data from the standard in to initialize a game state using the given
data.

@pre
-# *this is a valid instantiation of a game state.
-# *this is a cleared games state to prevent data corruption, or at least this
   function will perform the clearing operation.

@post
-# *this will contain the data of a valid game state.

@detail @bAlgorithm
-# *this is cleared.
-# The car data is read in sequentially.
-# The cars are then placed in the array

@code
@endcode
*/
void GameState::readIn()
{
  // assert pre-conditions
  assert( numCars > 0 );

  // variables
  int currentCar = 0;

  // read in each vehicle's data
  for( currentCar = 0; currentCar < numCars; currentCar++ )
  {
    // read in each piece of car data
    cin >> carList[ currentCar ].length >> carList[ currentCar ].orientation
        >> carList[ currentCar ].yPos >> carList[ currentCar ].xPos; 
  }

  // place each car
  for( currentCar = 0; currentCar < numCars; currentCar++ )
  {
    placeCar( currentCar, carList[currentCar].xPos, carList[currentCar].yPos );
  }

  // no return - void
}

/**

readIn

Reads in data from the standard in to initialize a game state using the given
data.

@pre

-# *this is a valid instantiation of a game state.
-# *this is a cleared games state to prevent data corruption, or at least this
   function will perform the clearing operation.

@post
-# *this will contain the data of a valid game state.

@detail @bAlgorithm
-# *this is cleared.
-# The car data is read in sequentially.

-# The cars are then placed in the array


@code
@endcode

*/
void GameState::setNumCars( const int newNumCars )
{
  // assert pre-conditions
  assert( newNumCars > 0 );

  // set the number of cars
  numCars = newNumCars;

  // no return - void
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   ACCESSORS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
isWin

Indicates if the GameState pertains to a winning one.

@return isWin   A boolean value indicating if the game has been won, true if it
                has, false otherwise.

@pre
-# *this is a valid instantiation of a GameState.
-# *this must actually contain a GameState in order for this result to produce a
   meaningful result.

@post
-# The status of whether or not the game has been won is returned.

@code
@endcode
*/
bool GameState::isWin() const
{
  // return the truth of the farthest reaching part of car 0 being in the
  // escape column
  return ( ( carList[0].xPos + carList[0].length ) == kBoardSize );
}

/**
stringify

Converts the current GameState to a string for GameState sharing, searching, and
comparison.

@return gameString   A string class object containing the data of the game board
                     inside it.

@pre
-# *this is a valid instantiation of a GameState.
-# *this must actually contain a GameState in order for this result to produce a
   meaningful result.

@post
-# The GameState will remain unchanged.
-# A string class object with the data pertaining to the game board is returned.

@code
@endcode
*/
string GameState::stringify() const
{
  // variables
  int ndx = 0;
  string carListString;

  // add each data piece of the car list to the string
  for( ndx = 0; ndx < numCars; ndx++ )
  {
    carListString += char( carList[ndx].xPos + '0' );
    carListString += char( carList[ndx].yPos + '0' );
    carListString += char( carList[ndx].orientation + '0' );
    carListString += char( carList[ndx].length + '0' );
  }

  // return the string
  return carListString;
}



void GameState::printBoard()
{
for( int rndx = 0; rndx < kBoardSize; rndx++ )
{
  for( int cndx = 0; cndx < kBoardSize; cndx++ )
  {
    cout << board[rndx][cndx];
  }
  cout << endl;
}

cout << endl << endl;
} 
