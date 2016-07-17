#ifndef ___MYSTRINGCLASS_H___
#define ___MYSTRINGCLASS_H___

////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      myIntListClass.h
//  Created By: Terence Henriod
//  Course:     CS202
//
//  Summary:    The declaration of the myStringClass string class.
// 
//  Last Modified: 4/18/2013 20:00
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

//============================================================================//
//= Class Definition =========================================================//
//============================================================================//

class myStringClass
   {

private:
  // Data Members //////////////////////////////////////////////////////////////
char* myString;
int myLen;

public:
  // Constructor(s) ////////////////////////////////////////////////////////////
  myStringClass();

  myStringClass( char* newStr );

  // Destructor ////////////////////////////////////////////////////////////////
  ~myStringClass();


private:
  // Internal/Maintenance //////////////////////////////////////////////////////
int myStrLen( const char* String );

public:
  // Accessors /////////////////////////////////////////////////////////////////
inline int myStringClass::length() const
   { return myLen; }


  // Mutators //////////////////////////////////////////////////////////////////
void myStringCopy( char* newStr );

void myStringCopy( myStringClass newStr );

void myStringCat( const char* suffix );

void myStringCat( const myStringClass suffix );


  // Overloaded Operators //////////////////////////////////////////////////////
void operator+ ( char* rhs );

void operator+ ( const myStringClass rhs );

void operator= ( char* rhs );

void operator= ( myStringClass &rhs );

bool operator== ( const char* rhs );

bool operator== ( const myStringClass &rhs );

bool operator!= ( const char* rhs );

bool operator!= ( const myStringClass &rhs );

friend ostream& operator<< ( ostream &out, const myStringClass &object );

friend istream& operator>> ( istream &in, const myStringClass &object );

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: checkArgs
// Summary:       Verifies that the cmd line arguments are suitable for normal 
//                program execution
// Parameters:    int argc       The count of the cmd line arguments the program
//                               was called with
//                char ** argv   Character pointers to the string values of the 
//                               cmd line arguments
// Returns:       void
//
////////////////////////////////////////////////////////////////////////////////

   };



#endif
