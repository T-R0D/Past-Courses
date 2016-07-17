#ifndef ___DEALER_H___
#define ___DEALER_H___

////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      dealer.h
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
#include "Ttools.h"
#include "card.h"
#include "participant.h"
using namespace std;


#pragma warning (disable : 4996)   // disables strcpy, strcat warnings in Visual
#pragma warning (disable : 4244)   // disables time(NULL) warnings in Visual


//============================================================================//
//= Class Definition =========================================================//
//============================================================================//

  // Constants /////////////////////////////////////////////////////////////////


class dealer : public participant
   {
   
private:
  // Data Members //////////////////////////////////////////////////////////////

/* all data members should be managed in the participant class */

public:
  // Constructor(s) ////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: dealer   [0]
// Summary:       The default constructor. Creates a new instance of a dealer.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  dealer();


  // Destructor ////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: ~dealer
// Summary:       The destructor. Currently nothing to destruct.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  ~dealer();

private:
  // Internal/Maintenance //////////////////////////////////////////////////////


public:
  // Accessors /////////////////////////////////////////////////////////////////


  // Mutators //////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: setName
// Summary:       Overrides the pure virtual function in the participant class.
//                Simply sets this participant's name to "DEALER"
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  virtual void setName( int playerNumber = 0 ) const; 

  // Overloaded Operators //////////////////////////////////////////////////////

   };

#endif