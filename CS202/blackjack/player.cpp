////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      player.cpp
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
#include "player.h"

// other headers
using namespace std;


//============================================================================//
//= Global (Static) Variables ================================================//
//============================================================================//


//============================================================================//
//= Function Implementation ==================================================//
//============================================================================//


// Constructor(s) //////////////////////////////////////////////////////////////
player::player(int playerNumber) : participant()
   {
   setName(playerNumber);
   }

// Destructor //////////////////////////////////////////////////////////////////
player::~player()
   {
   // currently nothing to destruct
   }

// Internal/Maintenance ////////////////////////////////////////////////////////


// Accessors ///////////////////////////////////////////////////////////////////


// Mutators ////////////////////////////////////////////////////////////////////
void player::setName(int playerNumber) const
   {
   // vars
   char num = (char(playerNumber) + '1');
   char numStr[2] = {num, '\0'};      // convert the parameter to a string
   char temp[50] = "";
   strcat(temp, "PLAYER ");

   // concatenate "PLAYER" and the number
   strcat(temp, numStr);

   // place the player name in the member string
   strcpy(pName, temp);    

   // no return - void
   }

// Overloaded Operators ////////////////////////////////////////////////////////

