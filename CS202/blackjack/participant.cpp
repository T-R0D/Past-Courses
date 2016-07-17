////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      participant.cpp
//  Created By: Terence Henriod
//  Course:     CS202
//
//  Summary:    The implementation of the class defined in cardClass.h
// 
//  Last Modified: 
//
////////////////////////////////////////////////////////////////////////////////


//============================================================================//
//= Header Files =============================================================//
//============================================================================//

// class definition header
#include "participant.h"

// other headers
using namespace std;


//============================================================================//
//= Global (Static) Variables ================================================//
//============================================================================//


//============================================================================//
//= Function Implementation ==================================================//
//============================================================================//


// Constructor(s) //////////////////////////////////////////////////////////////
participant::participant()
   {
   // initialize the participant name to a null string
   pName = new char [15];
     memset(pName, '\0', 15 * sizeof(char));

   // no return - constructor
   }

participant::participant( char* name )
   {
   // initialize the participant with the given name
   strcpy(pName, name);

   // no return - constructor
   }

// Destructor //////////////////////////////////////////////////////////////////
participant::~participant()
   {
   // return the dynamically allocated name string
   delete [] pName;

   // no return - destructor
   }

// Internal/Maintenance ////////////////////////////////////////////////////////


// Accessors ///////////////////////////////////////////////////////////////////
int participant::score()   // accesses the score of the current hand
   {
   return pHand.score();
   }

char* participant::name()  // accesses the player's name
   {
   return pName;
   }

hand participant::vHand()
   {
   return pHand;
   }

// Mutators ////////////////////////////////////////////////////////////////////
void participant::hit(deck &source)
   {
   // add the next card in the deck to the hand
   pHand += source.deal();
   
   // no return - void
   }

// Overloaded Operators ////////////////////////////////////////////////////////

