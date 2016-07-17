////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      hand.cpp
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
#include "hand.h"

// other headers
using namespace std;


//============================================================================//
//= Global (Static) Variables ================================================//
//============================================================================//


//============================================================================//
//= Function Implementation ==================================================//
//============================================================================//


// Constructor(s) //////////////////////////////////////////////////////////////
hand::hand()
   {
   // currently nothing to construct, the vector members should start empty
   }

// Destructor //////////////////////////////////////////////////////////////////
hand::~hand()
   {
   // currently nothing to destruct
   }

// Internal/Maintenance ////////////////////////////////////////////////////////
void hand::sort()
   {
   // implement later
   }

void hand::updateVals()
   {
   // vars
   unsigned int i;
   bool changeMade = false;

   // if score is a bust, attempt to change the score of one ace
   if(score() > 21){
     // iterate through the card values vector, if 11 is found, change it
     for(i = 0; i < cardValues.size() && !changeMade; i ++){
       // check the value of the element
       if(cardValues[i] == 11){
         cardValues[i] = 1;
         changeMade = true;
       }
     }
   } 
  
   // no return - void   
   }

// Accessors ///////////////////////////////////////////////////////////////////
unsigned int hand::score() const
   {
   // vars
   int score = 0;
   unsigned int i = 0;

   // sum the values stored in the vector of individual card scores
   for(i = 0; i < cardValues.size(); i ++){ 
   score += cardValues[i];
   }

   // return the summed score
   return unsigned int(score);
   }

int hand::numC()
   {
   return theCards.size();
   }
 
 card hand::selectCard(const int ndx)
   {
   // vars
   card pick;

   // retreive the sought card
   pick = theCards[ndx];

   // return the chosen card
   return pick;   
   }

// Mutators ////////////////////////////////////////////////////////////////////
void hand::addCard(const card &newCard)
   {
   // push a card into the hand
   theCards.push_back(newCard);
 
   // update the score value vector
   cardValues.push_back(newCard.points());

      // update the values in case of "Ace dilemma"
      updateVals();      

   // no return - void
   }

// Overloaded Operators ////////////////////////////////////////////////////////
void hand::operator += ( const card &rhs )
   {
   // add the card being offered as an argument
   addCard(rhs);

   // no return - void
   }

ostream& operator<< (ostream &out, hand &object)
   {
   // vars 
   int i = 0;

   // output the card values
   for(i = 0; i < object.numC(); i++){
     out << object.theCards[i] << "  ";
   }

   // return the ofstream object
   return out;
   }
