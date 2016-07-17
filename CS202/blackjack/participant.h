#ifndef ___PARTICIPANT_H___
#define ___PARTICIPANT_H___

////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      participant.h
//  Created By: Terence Henriod
//  Course:     CS202
//
//  Summary:    Defines the participant class, the abstract base for any kind of
//              player: dealer or actual player.
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
#include "deck.h"
#include "hand.h"
using namespace std;



#pragma warning (disable : 4996)   // disables strcpy, strcat warnings in Visual



//============================================================================//
//= Class Definition =========================================================//
//============================================================================//

  // Constants /////////////////////////////////////////////////////////////////


class participant
   {
   
private:
  // Data Members //////////////////////////////////////////////////////////////
protected:
  char* pName;

private:
  hand pHand;

public:
  // Constructor(s) ////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: participant   [0]
// Summary:       The default constructor. This is an abstract class and cannot
//                be instantiated. Completes the instantiation of an object of 
//                a derived class.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  participant();

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: participant   [1]
// Summary:       An overloaded constructor. ***This should be removed.***
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  participant( char* name );
 
////////////////////////////////////////////////////////////////////////////////
//
// Function Name: participant   [2]
// Summary:       An overloaded constructor. ***This should be removed.***
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  participant( int num );

  // Destructor ////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: ~participant
// Summary:       The destructor. Returns dynamic memory for the name string.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  ~participant();

private:
  // Internal/Maintenance //////////////////////////////////////////////////////


public:
  // Accessors /////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: score
// Summary:       Returns the score of the player's current hand.
//
// Parameters:    none
// Returns:       int   The score of the player's current hand.
//
////////////////////////////////////////////////////////////////////////////////
  int score(); 
 
////////////////////////////////////////////////////////////////////////////////
//
// Function Name: name
// Summary:       Returns a character pointer to the participant's name.
//
// Parameters:    none
// Returns:       char*   Points to the participant's name.
//
////////////////////////////////////////////////////////////////////////////////
  char* name();

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: vHand
// Summary:       Grants access to (view) the participant's hand.
//
// Parameters:    none
// Returns:       hand   The participant's current hand.
//
////////////////////////////////////////////////////////////////////////////////
  hand vHand();

  // Mutators //////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: hit
// Summary:       Adds a card taken from the source parameter to the 
//                participant's current hand.
//
// Parameters:    deck &source   The source of the card to be added to the 
//                               participant's hand.
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  void hit( deck &source );

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: setName
// Summary:       Pure virtual function to make the class abstract. Should be 
//                overwritten in any derived class.
//
// Parameters:    int playerNumber   The parameter that should determine the
//                                   behavior of the function in derived 
//                                   classes.
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  virtual void setName( int playerNumber = 0 ) const = 0;   // pure, virtual


  // Overloaded Operators //////////////////////////////////////////////////////

   };

#endif