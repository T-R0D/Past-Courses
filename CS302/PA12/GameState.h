/**
    @file .h

    @author Terence Henriod

    Project Name

    @brief Class declarations for...


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
@class ClassName

int Description
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
  void printBoard();

 private:
  /*---   Data Members   ---*/
  struct car
  {
    int xPos;
    int yPos;
    int length;
    char orientation;
  };
  char board[ kBoardSize ][ kBoardSize ];
  car carList[ kMaxCars ];
  int numCars;
};

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   TERMINATING PREPROCESSOR DIRECTIVES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#endif		// #ifndef BSTREE_H



