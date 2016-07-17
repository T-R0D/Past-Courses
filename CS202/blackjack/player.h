#ifndef ___PLAYER_H___
#define ___PLAYER_H___

////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      player.h
//  Created By: Terence Henriod
//  Course:     CS202
//
//  Summary:    This is a class derived from the abstract participant class 
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
#include "Ttools.h"
#include "participant.h"
#include "card.h"
using namespace std;



#pragma warning (disable : 4996)   // disables strcpy, strcat warnings in Visual



//============================================================================//
//= Class Definition =========================================================//
//============================================================================//

  // Constants /////////////////////////////////////////////////////////////////


class player : public participant
   {
   
private:
  // Data Members //////////////////////////////////////////////////////////////


public:
  // Constructor(s) ////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: player [0]
// Summary:       The default constructor. Creates a new instance of a player.
//
// Parameters:    int playerNumber (defaults to zero)   Provides the number for 
//                                                      which the player is 
//                                                      named    
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  player( int playerNumber = 0 );


  // Destructor ////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: ~player
// Summary:       The destructor. Currently does nothing.
//
// Parameters:    none    
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  ~player();

private:
  // Internal/Maintenance //////////////////////////////////////////////////////


public:
  // Accessors /////////////////////////////////////////////////////////////////


  // Mutators //////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: setName
// Summary:       Defines the player's name based on the number argument
//                provided.
//
// Parameters:    int playerNumber (defaults to zero)   Provides the number for 
//                                                      which the player is 
//                                                      named    
// Returns:       void
//
////////////////////////////////////////////////////////////////////////////////
virtual void setName( int playerNumber = 0 ) const;

  // Overloaded Operators //////////////////////////////////////////////////////

   };

#endif