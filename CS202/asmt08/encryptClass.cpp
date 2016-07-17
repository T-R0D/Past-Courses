////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      encryptClass.cpp
//  Created By: Terence Henriod
//  Course:     CS202
//
//  Summary:    The implementation of the class defined in abstractBase.h
// 
//  Last Modified: 4/30/2013 18:00
//
////////////////////////////////////////////////////////////////////////////////


//============================================================================//
//= Header Files =============================================================//
//============================================================================//


// class definition header
#include "encryptClass.h"


// other headers
using namespace std;


//============================================================================//
//= Global (Static) Variables ================================================//
//============================================================================//


//============================================================================//
//= Function Implementation ==================================================//
//============================================================================//


// Constructor(s) //////////////////////////////////////////////////////////////


// Destructor //////////////////////////////////////////////////////////////////
encryptClass::~encryptClass()
   {
   // currently nothing to destruct

   // no return - destructor
   }

// Internal/Maintenance ////////////////////////////////////////////////////////


// Accessors ///////////////////////////////////////////////////////////////////
unsigned int encryptClass::viewKey() const
   {
   return key;
   }

// Mutators ////////////////////////////////////////////////////////////////////
void encryptClass::set_key(const unsigned int k)
   {
   if(k != 0)
     {
     key = k;
     }
   else
     {
     key = randBetw(1, ALPHARANGE);
     }
 
   // no return - void
   }

char encryptClass::transform( char ch ) const
   {
   // vars
   char newChar = ch;

   // offset currChar
   if( isupper(newChar) )
     {
     if( (newChar + char(key)) <= 'Z' )
       {
       newChar += char(key);
       }
     else
       {
       newChar = ('A' + (key - ('Z' - newChar) ) );
       }
     }
   else if( islower(newChar) )
     {
     if( (newChar + char(key)) <= 'z' )
       {
       newChar += char(key);
       }
     else
       {
       newChar = ('a' + (key - ('z' - newChar) ) );
       }
     }
   else if( isdigit(newChar) )
     if( (newChar + char(key)) <= '9' )
       {
       newChar += char(key);
       }
     else
       {
       newChar = ('0' + (key - ('9' - newChar) ) );
       }   
    
   // decide what to do with non-alphanumerics 

   // return the transformed char
   return newChar;
   }

// Overloaded Operators ////////////////////////////////////////////////////////



