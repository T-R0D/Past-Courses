#ifndef ___DECK_H___
#define ___DECK_H___

////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      deck.h
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
#include <vector>
#include <queue>
#include "Ttools.h"
#include "card.h"
#include "displayHandClass.h"
using namespace std;



#pragma warning (disable : 4996)   // disables strcpy, strcat warnings in Visual
#pragma warning (disable : 4244)   // disables time(NULL) warnings in Visual


//============================================================================//
//= Class Definition =========================================================//
//============================================================================//

  // Constants /////////////////////////////////////////////////////////////////


class deck
   {
   
private:
  // Data Members //////////////////////////////////////////////////////////////
vector<card> unshuffledCards;
queue<card> cardsToUse;

public:
  // Constructor(s) ////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: deck   [0]
// Summary:       The default constructor. Creates a new instance of a deck
//                object. By default this object will be instantiated as a 
//                shuffled deck. The deck will be ordered when instantiated
//                a false argument.
//
// Parameters:    bool shuffled(=true)   Indicates whether or not the deck will
//                                       begin as shuffled or not.
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  deck( bool shuffled = true);   // shuffles by default


  // Destructor ////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: ~deck
// Summary:       The destructor. Currently nothing to destruct.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  ~deck();

private:
  // Internal/Maintenance //////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: genCards
// Summary:       Generates a set of 52 playing cards. Stores them in the 
//                vector member intended for internal operations.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  void genCards();

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: loadShuffled
// Summary:       Sends all cards held in the "internal" vector to the 
//                "external" queue in random order so as to perform shuffling.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  void loadShuffled();

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: loadUnshuffled
// Summary:       Sends all the cards in the "internal" vector to the 
//                "external" queue in such that order is preserved.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  void loadUnshuffled();

public:
  // Accessors /////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: peekAtTop
// Summary:       Allows the next card to be dealt (front of the queue) to be
//                viewed (on screen). Use at own risk. Programmer not 
//                responsible for shot out kneecaps, pool cues to the back of 
//                the head, ring imprints on face, etc.
//
// Parameters:    none
// Returns:       card   A copy of the card at the front of the queue.
//
////////////////////////////////////////////////////////////////////////////////
  card peekAtTop() const;   // for cheating

  // Mutators //////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: deal
// Summary:       Copies the card at the front of the queue and returns it.
//                eliminates it from the current deck.
//
// Parameters:    none
// Returns:       card   The card at the front of the queue.
//
////////////////////////////////////////////////////////////////////////////////
  card deal();

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: reShuffle
// Summary:       ***NOT YET IMPLEMENTED*** Allows the cards currently held in
//                the deck to be reshuffled.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  void reShuffle(); 

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: addCards
// Summary:       Adds a set of 52 shuffled cards to the deck whenever called.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  void addCards();

  // Overloaded Operators //////////////////////////////////////////////////////

   };

#endif