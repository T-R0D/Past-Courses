#ifndef ___DISPLAYHANDCLASS_H___
#define ___DISPLAYHANDCLASS_H___

////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      displayHandClass.h
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
#include "hand.h"
using namespace std;




#pragma warning (disable : 4996)   // disables strcpy, strcat warnings in Visual
#pragma warning (disable : 4244)   // disables time(NULL) warnings in Visual



//============================================================================//
//= Class Definition =========================================================//
//============================================================================//

  // Constants /////////////////////////////////////////////////////////////////

class participant;  // forward declaration - for some reason including 
                    // participant.h screws everything up

class displayHandClass
   {
   
private:
  // Data Members //////////////////////////////////////////////////////////////
  int numCards;
  
public:
  // Constructor(s) ////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: displayHandClass   [0]
// Summary:       The default constructor. Creates a new instance of a 
//                "displayer" object.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  displayHandClass();


  // Destructor ////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: ~displayHandClass
// Summary:       The destructor. Currently nothing to destruct.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  ~displayHandClass();

private:
  // Internal/Maintenance //////////////////////////////////////////////////////


public:
  // Accessors /////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: disp
// Summary:       Outputs a graphical display (via command line output) to the
//                terminal. Also displays the hand score.
//
// Parameters:    hand &theHand   The hand to be displayed.
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  void disp( hand &theHand );

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: simpleDisp
// Summary:       An alternate format for graphically displaying a hand.
//                Similar to disp. Allows the first card of the hand to be 
//                hidden. Will likely replace disp altogether.
//
// Parameters:    participant* individual   A pointer to the participant whose
//                                          hand will be displayed.
//                bool hideFirst            Indicates whether or not the first 
//                                          card should be displayed face down.
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  void simpleDisp(participant* individual, bool hideFirst);  

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: oneCard
// Summary:       Graohically displays just a single card.
//
// Parameters:    card &theCard   The card to be put on screen.
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  void oneCard( card &theCard );

  // Mutators //////////////////////////////////////////////////////////////////


  // Overloaded Operators //////////////////////////////////////////////////////

   };

#endif