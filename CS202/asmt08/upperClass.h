#ifndef ___UPPERCLASS_H___
#define ___UPPERCLASS_H___

////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      upperClass.h
//  Created By: Terence Henriod
//  Course:     CS202
//
//  Summary:    
// 
//  Last Modified: 4/30/2013 18:00
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
#include <iomanip>
#include <assert.h>
#include "Ttools.h"
using namespace std;

// parent class header
#include "f_filterBase.h"

//============================================================================//
//= Class Definition =========================================================//
//============================================================================//

class upperClass : public f_filterBase
   {
   
private:
  // Data Members //////////////////////////////////////////////////////////////


public:
  // Constructor(s) ////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: upperClass
// Summary:       The default constructor. Creates a new instance of an
//                uppercasing file filter with a default name for the output
//                file.
//
// Parameters:    none
//
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
upperClass() : f_filterBase()
   { set_inF("uppercased.txt"); }

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: myIntListClass   [0]
// Summary:       An overloaded constructor. Creates a new instance of an
//                uppercasing object that uses the given file name parameters.
//
// Parameters:    string &inF    A string containing the name of the input 
//                               file to be used
//                string &outF   A string containing the name of the output
//                               file to be created    
//
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
upperClass(const string &inF, const string &outF) : f_filterBase(inF, outF)
   {}


  // Destructor ////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: ~upperClass
// Summary:       The destructor. carries out any necessary destruction for an 
//                upperClass object.
//
// Parameters:    none
//
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
virtual ~upperClass()
   { /* currently empty */ };

private:
  // Internal/Maintenance //////////////////////////////////////////////////////


public:
  // Accessors /////////////////////////////////////////////////////////////////


  // Mutators //////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: transform
// Summary:       If a character is a lowercase letter, it is capitalized and 
//                returned.
//
// Parameters:    char ch   The character to (possibly) be capitalized
//
// Returns:       char   The character after the (possible) transformation
//
////////////////////////////////////////////////////////////////////////////////
virtual char transform( char ch ) const;

  // Overloaded Operators //////////////////////////////////////////////////////

   };

#endif


