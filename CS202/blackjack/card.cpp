////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      card.cpp
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

#include "card.h"


// other headers

using namespace std;


//============================================================================//
//= Global (Static) Variables ================================================//
//============================================================================//


//============================================================================//
//= Function Implementation ==================================================//
//============================================================================//


// Constructor(s) //////////////////////////////////////////////////////////////
card::card()
   {
   // initialize data members to null
   rankVal = NULL;
   pointVal = 0;
   suitVal = NULL;
   updateColor(NULL);

   // no return - constructor
   }

card::card(char r, char s)
   {
   // initialize data members to null
   setRank(r);
   setSuit(s);

   // no return - constructor
   }

card::card(const card &rhs)
   {
   // copy the members over
   rankVal = rhs.rankVal;
     updatePoints(rankVal);
   suitVal = rhs.suitVal;
     updateColor(suitVal);

   // no return - constructor
   }

// Destructor //////////////////////////////////////////////////////////////////
card::~card()
   {
   // currently no action
   
   // no return - destructor
   }

// Internal/Maintenance ////////////////////////////////////////////////////////
void card::updatePoints(char r)
   {
   // update data member
   switch(r){
     case 'A':
       pointVal = 11;
     break;

     case '2':
       pointVal = 2;
     break;

     case '3':
       pointVal = 3;
     break;

     case '4':
       pointVal = 4;
     break;

     case '5':
       pointVal = 5;
     break;

     case '6':
       pointVal = 6;
     break;

     case '7':
       pointVal = 7;
     break;

     case '8':
       pointVal = 8;
     break;

     case '9':
       pointVal = 9;
     break;

     case 'T':
     case 'J':
     case 'Q':
     case 'K':
       pointVal = 10;
     break;

     default:
       pointVal = 0;
     break;
   }

   // no return - void
   }

void card::updateColor(char s)
   {
   // decide which color to use depending on suit
   switch(s){
     case DIAMOND:
     case HEART:
       colorVal = RED;
     break;

     case CLUB:
     case SPADE: 
       colorVal = BLACK;
     break;
  
     default:
       colorVal = NONE;
     break;
   }

   // no return - void   
   }

// Accessors ///////////////////////////////////////////////////////////////////
char card::rank() const
   {
   return rankVal;
   }

int card::points() const
   {
   return pointVal;
   }

char card::suit() const
   {
   return suitVal;
   }

int card::color() const
   {
   return colorVal;
   }

// Mutators ////////////////////////////////////////////////////////////////////
void card::setRank(const char r)
   {
   // update the data value
   switch(r){
     case 'a':
     case 'A':
     case '1':
       rankVal = 'A';
     break;

     case '2':
       rankVal = '2';
     break;

     case '3':
       rankVal = '3';
     break;

     case '4':
       rankVal = '4';
     break;

     case '5':
       rankVal = '5';
     break;

     case '6':
       rankVal = '6';
     break;

     case '7':
       rankVal = '7';
     break;

     case '8':
       rankVal = '8';
     break;

     case '9':
       rankVal = '9';
     break;

     case 't':
     case 'T':
       rankVal = 'T';
     break;

     case 'j':
     case 'J':
       rankVal = 'J';
     break;

     case 'q':
     case 'Q':
       rankVal = 'Q';
     break;

     case 'k':
     case 'K':
       rankVal = 'K';
     break;

     default:
       rankVal = NULL;
     break;
   }

   // update points
   updatePoints(rankVal);

   // no return - void
   }

void card::setRank(const int r)
   {
   // update the data value
   switch(r){
     case 1:
     case 11:
       rankVal = 'A';
     break;

     case 2:
       rankVal = '2';
     break;

     case 3:
       rankVal = '3';
     break;

     case 4:
       rankVal = '4';
     break;

     case 5:
       rankVal = '5';
     break;

     case 6:
       rankVal = '6';
     break;

     case 7:
       rankVal = '7';
     break;

     case 8:
       rankVal = '8';
     break;

     case 9:
       rankVal = '9';
     break;

     case 10:
       rankVal = 'T';
     break;

     default:
       rankVal = NULL;
     break;
   }

   // update points
   updatePoints(rankVal);

   // no return - void
   }

void card::setSuit(const char s)
   {
   // update the data member
   switch(s){
     case 'd':
     case 'D':
     case DIAMOND:
       suitVal = DIAMOND;
     break;

     case 'c':
     case 'C':
     case CLUB:
       suitVal = CLUB;
     break;

     case 'h':
     case 'H':
     case HEART:
       suitVal = HEART;
     break;

     case 's':
     case 'S':
     case SPADE:
       suitVal = SPADE;
     break;

     default:
       suitVal = NULL;
     break;
   } 
    
   // update the card's color
   updateColor(suitVal);

   // no return - void
   }

// Overloaded Operators ////////////////////////////////////////////////////////
card card::operator = (const card &rhs)
   {
   // copy the data over
   rankVal = rhs.rankVal;
   suitVal = rhs.suitVal;

   // return rhs
   return *this;
   }

ostream& operator<< (ostream &out, const card &object)
   {
   // output the card values
   out << object.rank() << object.suit();

   // return the ofstream object
   return out;
   }