#ifndef ___GAMECLASS_H___
#define ___GAMECLASS_H___

////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      gameClass.h
//  Created By: Terence Henriod
//  Course:     CS202
//
//  Summary:    
// 
//  Last Modified: 
//
////////////////////////////////////////////////////////////////////////////////


//============================================================================//
//= Header Files =============================================================//
//============================================================================//

// headers/namespaces

#include <iostream>
#include <fstream>
#include <conio.h>
#include <stdlib.h>
#include <assert.h>
#include <stack>
#include "card.h"
#include "hand.h"
#include "deck.h"
#include "player.h"
#include "dealer.h"
#include "displayHandClass.h"
#include "Ttools.h"




#pragma warning (disable : 4996)   // disables strcpy, strcat warnings in Visual


//============================================================================//
//= Class Definition =========================================================//
//============================================================================//

  // Constants /////////////////////////////////////////////////////////////////
  const int MAX_P = 3; 

class gameClass
   {
   
private:
  // Data Members //////////////////////////////////////////////////////////////
  static int gameNumber;
  deck theDeck;
  int numP;
  player* currPlayer;
  dealer* theHouse;
  tm* startTime;
  struct result
    {
    bool played;
    char pName[25];
    hand pHand;
    int score;
    char outcome;
    };
  stack<result> summary;

public:
  // Constructor(s) ////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: gameClass   [0]
// Summary:       The default constructor. Creates a new instance of a game 
//                engine class.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  gameClass();


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: myIntListClass   [0]
// Summary:       The default constructor. Creates a new instance of a doubly
//                linked list with no nodes.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////



  // Destructor ////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: ~gameClass
// Summary:       The destructor. Currently detucts nothing.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  ~gameClass();

private:
  // Internal/Maintenance //////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: gameStartTime
// Summary:       Simply gets the current system time and stores it in the
//                class's data member for later use.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  void gameStartTime();  

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: startDeal
// Summary:       Performs the initial deal at the start of the game. Manages
//                data only.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  void startDeal();

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: playerTurn
// Summary:       Manages all tasks associated with a player's turn, both in 
//                data and display capacities.
//
// Parameters:    int playerNum   Used to indicate which player is taking their
//                                turn.
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  void playerTurn( int playerNum );

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: dealerTurn
// Summary:       Manages the dealer's turn. Dealer is automated and plays to 
//                basic strategy (i.e. hits until score >= 17). Displays only
//                the dealer's final hand.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  void dealerTurn();

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: hitMenu
// Summary:       Displays the player turn menu (hit or stand) and collects the
//                user's input.
//
// Parameters:    none
// Returns:       char   The user's input. This is handled by the calling
//                       function.
//
////////////////////////////////////////////////////////////////////////////////
  char hitMenu();

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: scoreGame
// Summary:       Determines the game results for each player and stores the
//                data in a stack for later output.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  void scoreGame();

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: showOutcomes
// Summary:       Outputs a simple display of the game results at the end of 
//                game play so that it can be seen how each player fared.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  void showOutcomes();

public:
  // Accessors /////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: play
// Summary:       Executes all functions associated with gameplay.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  void play();

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: recordGameData
// Summary:       Outputs the result data to the output file. Data includes:
//                the number of the game played (as far as the current prigram 
//                run), the time the game began, player name, hand contents &
//                score, and the outcome of the game for that player.
//
// Parameters:    char* outFile   A char* to the name of the outputfile to be 
//                                used.
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  void recordGameData( const char* outFile );

  // Mutators //////////////////////////////////////////////////////////////////

  // Overloaded Operators //////////////////////////////////////////////////////

   };

#endif