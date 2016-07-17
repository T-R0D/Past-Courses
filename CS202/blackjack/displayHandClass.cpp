////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      displayHandClass.cpp
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
#include "displayHandClass.h"
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
displayHandClass::displayHandClass()
   {
   // currently nothing to construct
   }

// Destructor //////////////////////////////////////////////////////////////////
displayHandClass::~displayHandClass()
   {
   // currently nothing to destruct
   }

// Internal/Maintenance ////////////////////////////////////////////////////////


// Accessors ///////////////////////////////////////////////////////////////////
void displayHandClass::disp(hand &theHand)
   {
   // vars
   numCards = theHand.numC();
   int counter = 0;
   int sndx = 0;
   card currCard;
   const int blankH = 6;
   const char* first = "++++++++";
   const char* second = "++";
   const char* side1 = "+ ";
   const char* side2 = " +";
   const char* smallBlank = "    ";
   const char* blankSpace = "      ";

   // inform the player of their score
   cout << "The hand score is: " << theHand.score() << endl;

   // notify the player that this is their hand
   cout << "The hand consists of:" << endl << endl;

   // display the cards, one ASCII row at a time
     // top row first
     for(counter = 0; counter < numCards; counter ++){
       cout << first;
     }
     cout << second << endl;

     // then the card id row
     for(counter = 0; counter < numCards; counter ++){
       currCard = theHand.selectCard(counter);
       cout << side1 << currCard << smallBlank;
     }
     cout << side2 << endl;

     // the rest of the card body
     for(sndx = 0; sndx < blankH; sndx ++){
       for(counter = 0; counter < numCards; counter ++){
          cout << side1 << blankSpace;
        }
       cout << side2 << endl;
     }

     // finally, the bottom of the cards
     for(counter = 0; counter < numCards; counter ++){
       cout << first;
     }
     cout << second << endl;

   // the program should hold via the external operation
 
   // no return - void
   }



void displayHandClass::simpleDisp( participant* individual, bool hideFirst = false)
   {
   // vars
   numCards = individual->vHand().numC();
   int counter = 0;
   int sndx = 0;
   card currCard;
   const int blankH = 6;
   const char* first = "++++++++";
   const char* second = "++";
   const char* side1 = "+ ";
   const char* side2 = " +";
   const char* smallBlank = "    ";
   const char* blankSpace = "      ";
   const char* shBack = "+XXXXXXX";
   const char* wdBack = "+XXXXXXXXX+";

/* Ideal layout of a single card
   cout << "+++++++++++"  << endl
        << "+ Ah      +"  << endl
        << "+         +"  << endl
        << "+         +"  << endl
        << "+         +"  << endl
        << "+         +"  << endl
        << "+         +"  << endl
        << "+++++++++++"  << endl; */

   // indentify the player
   cout << "  " << individual->name() << "'s hand:" << endl << endl;

   // print the cards
   // display the cards, one ASCII row at a time

   // top row first
   for(counter = 0; counter < numCards; counter ++){
     cout << first;
   }
   cout << second << endl;

   // take special measures if the first card should be hidden (else)
   if(!hideFirst){
     // then the card id row
     for(counter = 0; counter < numCards; counter ++){
       currCard = individual->vHand().selectCard(counter);
       cout << side1 << currCard << smallBlank;
     }
     cout << side2 << endl;

     // the rest of the card body
     for(sndx = 0; sndx < blankH; sndx ++){
       for(counter = 0; counter < numCards; counter ++){
          cout << side1 << blankSpace;
       }
       cout << side2 << endl;
     }
   }
   else{
     // card id row should be different
     for(counter = 0; counter < numCards; counter ++){
       if(counter == 0){
       cout << shBack;
       }
       else{
       currCard = individual->vHand().selectCard(counter);
       cout << side1 << currCard << smallBlank;
       }
     }
     cout << side2 << endl;

     // card body should be different
     for(sndx = 0; sndx < blankH; sndx ++){
       for(counter = 0; counter < numCards; counter ++){
         if(counter == 0){
           cout << shBack;
         }
         else{
           cout << side1 << blankSpace;
         }
       }
       cout << side2 << endl;
     }
   }

   // finally, the bottom of the cards
   for(counter = 0; counter < numCards; counter ++){
       cout << first;
   }
   cout << second << endl;

   // display the hand's score if it should be seen
   if(!hideFirst){
     cout << "     Score: " << individual->vHand().score() << endl << endl;   
   }


   //no return - void
   } 

void displayHandClass::oneCard(card &theCard)
   {
   // display the single card
   cout << "   ++++++++++"  << endl
        << "   + " << theCard << "     +"  << endl
        << "   +        +"  << endl
        << "   +        +"  << endl
        << "   +        +"  << endl
        << "   +        +"  << endl
        << "   +        +"  << endl
        << "   +        +"  << endl
        << "   ++++++++++"  << endl;

   // hold program, clear screen
   holdProg();                      // change to brief wait
   clrScr();
   
   // no return - void
   }

// Mutators ////////////////////////////////////////////////////////////////////

// Overloaded Operators ////////////////////////////////////////////////////////
   
