/**
    @file GameState.h

    @author Terence Henriod

    Rush Hour: Breadth First Search

    @brief Class declarations for a Rush Hour Game State


    @version Original Code 1.00 (10/29/2013) - T. Henriod
*/


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   PREPROCESSOR DIRECTIVES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#ifndef ___GAMESTATE_H___
#define ___GAMESTATE_H___


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#include <cassert>
#include <iostream>
#include <string>
using namespace std;


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   GLOBAL CONSTANTS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

const bool FORWARD = true;
const bool BACKWARD = false;
const char kHorizontal = 'H';
const char kVertical = 'V';
const char kEmpty = '.';
const int kBoardSize = 6;
const int kBoardSizeSquared = ( kBoardSize * kBoardSize );
const int kMaxCars = 18;
const int kCarXpos = 0;
const int kCarYpos = 1;
const int kLengthPos = 3;
const int kOrientationPos = 4;
const int kCarDataSize = 4;


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
================================================================================
                   CLASS DEFINITION(S)
================================================================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
@class GameState

Contains the full amount of data required to represent and manipulate a Rush
Hour Game state. Includes a representation of a game board, the ability to
move cars, and overloaded copy constructor and assignment operator for use with
the STL queue.
*/
class GameState
{
 public:
  /*---   Constructor(s) / Destructor   ---*/
  GameState();
  GameState( const GameState& other );
  GameState& operator=( const GameState& other );
  ~GameState();

  /*---   Mutators   ---*/
  void clear();
  bool move( const int whichCar, const bool direction );
  bool placeCar( const int whichCar, int newX, int newY );
  void readIn();
  void setNumCars( const int newNumCars );

  /*---   Accessors   ---*/
  bool isWin() const;
  string stringify() const;
  void printBoard() const;

 private:
  /*---   Data Members   ---*/
  /**
  @struct car

  Contains all the pertinent data for representing a car, with a pair of
  coordinates for the car's head, its length, and its orientation.  
  */
  struct car
  {
    int xPos;
    int yPos;
    int length;
    char orientation;
  };
  char board[ kBoardSize ][ kBoardSize ]; // used to simulate moving
  car carList[ kMaxCars ];                // keeps track of all car data
  int numCars;                            // represents the number of cars on
                                          // the board
};

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   TERMINATING PREPROCESSOR DIRECTIVES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#endif		// #ifndef BSTREE_H



