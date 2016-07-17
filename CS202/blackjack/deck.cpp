////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      deck.cpp
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
#include "deck.h"

// other headers
using namespace std;


//============================================================================//
//= Global (Static) Variables ================================================//
//============================================================================//


//============================================================================//
//= Function Implementation ==================================================//
//============================================================================//


// Constructor(s) //////////////////////////////////////////////////////////////
deck::deck(bool shuffled)
   {
   // add an initial set of cards to the deck
   if(shuffled == true){
     addCards();
   }
   else{
     genCards();
     loadUnshuffled();
   }
  
   // no return - constructor
   }

// Destructor //////////////////////////////////////////////////////////////////
deck::~deck()
   {
   // currently nothing to destruct
   }

// Internal/Maintenance ////////////////////////////////////////////////////////
void deck::genCards()
   {
   // vars
   int sndx = 0;
   char suit = CLUB;
   int rndx = 0;
   char rank = 'A';
   card temp;

   // use nested loops to generate a card of every rank, of every suit
   // then load them into the vector member
   // generate by suit
   for(sndx = 0; sndx < 4; sndx ++){
      // decide the suit
      switch(sndx){
        case 0:
          suit = CLUB;
          break;

        case 1:
          suit = DIAMOND;
          break;

        case 2:
          suit = HEART;
          break;

        case 3:
          suit = SPADE;
          break;
      }
       
      // generate by rank
      for(rndx = 0; rndx < 13; rndx ++){
        // set the rank value with a long switch statement
        switch(rndx){
          case 0:
            rank = 'A';
            break;

          case 1:
            rank = '2';
            break;

          case 2:
            rank = '3';
            break;

          case 4:
            rank = '5';
            break;

          case 5:
            rank = '6';
            break;

          case 6:
            rank = '7';
            break;

          case 7:
            rank = '8';
            break;

          case 8:
            rank = '9';
            break;

          case 9:
            rank = 'T';
            break;

          case 10:
            rank = 'J';
            break;

          case 11:
            rank = 'Q';
            break;

          case 12:
            rank = 'K';
            break;
        }

        // create the card and add to vector
        temp.setRank(rank);
        temp.setSuit(suit);
        unshuffledCards.push_back(temp);
      }
   }

   // no return - void
   }

void deck::loadShuffled()
   {
   // vars
   int i = 0;

  // grab a random element of the vector, push into queue until full/empty
   while(!unshuffledCards.empty()){
     i = randBetw(0, (unshuffledCards.size() - 1));
     cardsToUse.push(unshuffledCards[i]);
     unshuffledCards.erase(unshuffledCards.begin() + i);   // consider the int vs. iterator thing
     
     // update i
     i++;
   }

   // no return - void
   }

void deck::loadUnshuffled()
   {
   // vars

   // pop elements from vector and place in queue sequentially
   while( !unshuffledCards.empty() ){
     cardsToUse.push( unshuffledCards[0] );
     unshuffledCards.pop_back();
   }

   // no return - void
   }

// Accessors ///////////////////////////////////////////////////////////////////
card deck::peekAtTop() const
   {
   // vars
   card temp;
   displayHandClass* disp;

   // display the message
   cout << endl << endl << endl
        << "   SHHH! Don't let anyone know you're cheating!" << endl << endl;

   // copy the next card to be dealt
   temp = cardsToUse.front();
   
   // display the card
   disp = new displayHandClass;
   disp->oneCard(temp);
   delete disp;   

   // return the card at the front of the queue
   return temp;
   }

// Mutators ////////////////////////////////////////////////////////////////////
card deck::deal() 
   {
   // vars
   card dealt;

   // pop a card off the front of the queue
   dealt = cardsToUse.front();
   cardsToUse.pop();

   // get new deck if the deck as <= 20 cards left
   if(cardsToUse.size() <= 20){
     // discard the current deck by popping all the remaining elements
     while( !cardsToUse.empty() ){
       cardsToUse.pop();
     }

     // reload the deck
     addCards();      
   }

   // return the card
   return dealt;
   }

void deck::reShuffle()   // in case anyone wants it
   {
   // implement later
   }

void deck::addCards()    // adds a set of 52 for use when multiple decks are required or when the playing deck needs to be refilled
   {
   // create the cards, load them into the queue
   genCards();
   loadShuffled();

   // no return - void   
   }

// Overloaded Operators ////////////////////////////////////////////////////////

