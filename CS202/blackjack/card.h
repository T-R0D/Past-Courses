#ifndef ___CARD_H___
#define ___CARD_H___

////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      card.h
//  Created By: Terence Henriod
//  Course:     CS202
//
//  Summary:    Defines a class whose objects represent a typical playing card.
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
using namespace std;




#pragma warning (disable : 4996)   // disables strcpy, strcat warnings in Visual
#pragma warning (disable : 4244)   // disables time(NULL) warnings in Visual




//============================================================================//
//= Class Definition =========================================================//
//============================================================================//

  // Constants /////////////////////////////////////////////////////////////////

// define the suits
const char HEART = char(3);
const char DIAMOND = char (4);
const char CLUB = char(5);
const char SPADE = char(6);

// define the colors
enum COLOR
   {
   NONE,
   BLACK,
   RED 
   };


class card
   {
   
private:
  // Data Members //////////////////////////////////////////////////////////////
char rankVal;
int pointVal;
char suitVal;
int colorVal;

public:
  // Constructor(s) ////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: card   [0]
// Summary:       The default constructor. Creates a new instance of a card
//                with null data.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
card();

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: card   [1]
// Summary:       An alternate constructor that allows the card object to be
//                defined with rank and suit from instantiation.
//
// Parameters:    char r   A character representing the rank of the card
//                char s   A character representing the suit of the card
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
card( char r, char s );

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: card   [2]
// Summary:       A card object copy constructor.
//
// Parameters:    card &rhs   The card to be copied.
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
card( const card &rhs );

  // Destructor ////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: ~card
// Summary:       The destructor. Currently nothing to destruct.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
~card();

private:
  // Internal/Maintenance //////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: updatePoints
// Summary:       Updates the pint value of the card based on the rank of the 
//                card. Generally used for when the card's rank is changed.
//
// Parameters:    char r   The rank of the card.
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
void updatePoints(char r);

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: updateColor
// Summary:       Updates the color of the card based on its suit. Typically 
//                called after the suit of the card is updated.
//
// Parameters:    char s   The suit for which the color should correspond.
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
void updateColor(char s);


public:
  // Accessors /////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: rank
// Summary:       Returns the rank of the card.
//
// Parameters:    none
// Returns:       char   The character representing the rank of the card.
//
////////////////////////////////////////////////////////////////////////////////
char rank() const;

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: points
// Summary:       Returns the current pointvalue held by the card
//
// Parameters:    none
// Returns:       int   The point value of the card.
//
////////////////////////////////////////////////////////////////////////////////
int points() const;

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: suit
// Summary:       Returns the suit of the card.
//
// Parameters:    none
// Returns:       char   The character indicating the suit of the card.
//
////////////////////////////////////////////////////////////////////////////////
char suit() const;

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: color
// Summary:       Returns the color code of the card.
//
// Parameters:    none
// Returns:       int   The integer corresponding to the card's color.
//
////////////////////////////////////////////////////////////////////////////////
int color() const;

  // Mutators //////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: setRank
// Summary:       Changes the rank of the card.
//
// Parameters:    char r   The rank value the card will take.
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
void setRank(const char r);

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: setRank
// Summary:       An overloaded version of setRank so that integers can be 
//                used.
//
// Parameters:    int r   The integer version of the rank value the card will
//                        take.
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
void setRank(const int r);

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: setSuit
// Summary:       Changes the suit value of the card to that of the given
//                argument.
//
// Parameters:    char s   The character code for the suit the card's value
//                         will be updated to.
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
void setSuit(const char s);

  // Overloaded Operators //////////////////////////////////////////////////////
  card operator = ( const card & rhs );

  friend ostream& operator<< ( ostream &out, const card &object );

   };

#endif