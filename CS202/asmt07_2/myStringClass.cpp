////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      myStringClass.cpp
//  Created By: Terence Henriod
//  Course:     CS202
//
//  Summary:    The implementation of the class defined in myStringClass.h
// 
//  Last Modified: 4/18/13 20:00
//
////////////////////////////////////////////////////////////////////////////////

//============================================================================//
//= Header Files =============================================================//
//============================================================================//

// class definition header
#include "myStringClass.h"
 
// other headers
using namespace std;


//============================================================================//
//= Global (Static) Variables ================================================//
//============================================================================//


//============================================================================//
//= Function Implementation ==================================================//
//============================================================================//

// Constructor(s) //////////////////////////////////////////////////////////////
myStringClass::myStringClass()
   {
      myString = new char;
        if(myString == 0)   // implement better error handling
          {
          exit(1);
          }
        myString[0] = '\0';
      myLen = 0;
   // no return - constructor
   }

myStringClass::myStringClass( char* newStr )
   {
   myStringCopy( newStr );
   myLen = myStrLen( myString );

   // no return - constructor
   }

// Destructor //////////////////////////////////////////////////////////////////
myStringClass::~myStringClass()
   {
   // return dynamic memory
   delete [] myString;                                         // fix this

   // no return - destructor
   }

// Internal/Maintenance ////////////////////////////////////////////////////////

int myStringClass::myStrLen( const char* String )
   {
   // vars 
   int i = 0;

   // iterate through string, counting
   while( myString[i] != '\0' )
     {
     i ++;
     }

   // return length
   return i;
   }

// Accessors ///////////////////////////////////////////////////////////////////


// Mutators ////////////////////////////////////////////////////////////////////

void myStringClass::myStringCopy( char* newStr )
   {
   // vars
   char* oldLoc = myString;
   char* tempLoc = new char [1000];
     if( tempLoc = NULL )    // replace with proper exception handling
       {
       exit(1);
       }
     memset(tempLoc, '\0', sizeof(char)*1000);
   char* newLoc;
   int newSize = myStrLen( newStr );
   int i = 0;

   // copy the characters by iteration
   while( newStr[i] != '\0' )
     {
     tempLoc[i] = newStr[i];
     i ++;
     }
   
   // copy the string in the temporary loc to appropriatley sized newLoc
   newLoc = new char [newSize + 1];   
     if( newLoc == 0 )     // replace with proper exception handling
       {
       exit(1);
       }
     memset(newLoc, '\0', (newSize + 1) * sizeof(char)); 

   // copy the characters by iteration
   i = 0;
   while( tempLoc[i] != '\0' )
     {
     newLoc[i] = tempLoc[i];
     i ++;
     }

   // set myString to newString, delete old loc
   myString = newLoc;
   delete [] oldLoc;
   delete [] tempLoc;

   // update size
   myLen = myStrLen( myString );

   // no return - void
   }

void myStringClass::myStringCopy( myStringClass newStr )
   {
   // copy the newStr member to myString
   myStringCopy(newStr.myString);

   // update size
   myLen = myStrLen( myString );

   // no return - void
   }

void myStringClass::myStringCat( const char* suffix )
   {
   // vars 
   char* oldLoc = myString;
   char* newLoc;
   int newSize = (myStrLen(myString) + myStrLen(suffix));
   int i = 0;
   int j = 0;

   // get new space
   newLoc = new char [newSize + 5];
     memset(newLoc, '\0', (newSize + 5) * sizeof(char) );    // figure out why it has to be 5!!!!!!

   // copy over string, then suffix
   while( myString[i] != '\0' )
     {
     newLoc[i] = myString[i];
     i ++;
     }
   //cout << "i = " << i << endl;
   //  assert( i < myStrLen(myString) );
   while( suffix[j] != '\0' )
     {
     newLoc[i] = suffix[j];
     i ++;
     j ++;
     }
   
   // terminate with null
   newLoc[i] = '\0';

   // delete old location 
   myString = newLoc;
   delete [] oldLoc;                                    // fix this so it works in all cases

   // update myLen
   myLen = myStrLen( myString );

   // no return - void
   }

void myStringClass::myStringCat( const myStringClass suffix )
   {
   // concatenate the member string from newStr to myString
   myStringCat(suffix.myString);

   // update myLen
   myLen = myStrLen( myString );

   // no return - void
   }

// Overloaded Operators ////////////////////////////////////////////////////////

void myStringClass::operator+ ( char* rhs )
   {
   myStringCat( rhs );

   // no return - void
   }

void myStringClass::operator+ ( const myStringClass rhs )
   {
   myStringCat(rhs.myString);

   // no return - void
   }

void myStringClass::operator= ( char* rhs )
   {
   // copy the string to the myString member
   myStringCopy( rhs );

   // update length
   myLen = myStrLen(myString);

   // no return - void
   }

void myStringClass::operator= ( myStringClass &rhs )
   {
   // copy the rhs string member to the myString member
   myStringCopy( rhs.myString );

   // update length
   myLen = myStrLen(myString);

   // no return - void
   }

bool myStringClass::operator== ( const char* rhs )
   {
   // vars
   bool result = false;   // result - is equal
   int i = 0;

   // test each subsequent letter until a difference is reached
   while(myString[i] == rhs[i])   
     {
     i ++;

     if( (myString[i] == '\0') && (rhs[i] == '\0') )
       {
       result = true;
       } 
     }

   // return result
   return result;
   }

bool myStringClass::operator== ( const myStringClass &rhs )
   {
   // vars
   bool result = false;   // result - is equal

   // return result
   return result;
   }

bool myStringClass::operator!= ( const char* rhs )
   {
   // vars
   bool result = false;   // result - is not equal
   int i = 0;

   // test each subsequent letter until a difference is reached
   while(myString[i] != rhs[i])   
     {
     i ++;

     if( (myString[i] == '\0') && (rhs[i] == '\0') )
       {
       result = true;
       } 
     }

   // return result
   return result;
   }

bool myStringClass::operator!= ( const myStringClass &rhs )
   {
   // vars
   bool result = false;   // result - is not equal
   int i = 0;

   // test each subsequent letter until a difference is reached
   while(myString[i] != rhs.myString[i])   
     {
     i ++;

     if( (myString[i] == '\0') && (rhs.myString[i] == '\0') )
       {
       result = true;
       } 
     }
   // return result
   return result;
   }

ostream& operator<< ( ostream &out, const myStringClass &object )
   {
   // vars                         This is a skeleton for later improvement
                                   // it is functional though

   // output the string
   cout << object.myString;

   // return ofstream object
   return out;
   }

istream& operator>> ( istream &in, const myStringClass &object )
   {
   // vars                         This is a skeleton for later improvement
                                   // it is functional though

   

   // return the stream object
   return in;
   }
