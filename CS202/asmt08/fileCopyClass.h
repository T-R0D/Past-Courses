#ifndef ___FILECOPYCLASS_H___
#define ___FILECOPYCLASS_H___

////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      fileCopyClass.h
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

class fileCopyClass : public f_filterBase
   {
   
private:
  // Data Members //////////////////////////////////////////////////////////////


public:
  // Constructor(s) ////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: fileCopyClass
// Summary:       The default constructor. Creates a new instance of a file
//                copying class. Sets the output file name by default.
//
// Parameters:    none
//
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
fileCopyClass() : f_filterBase()
   { set_inF("copy.txt"); }

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: fileCopyClass
// Summary:       An overloaded constructor. Instantiates a fileCopyClass
//                object with the given file name parameters.
//
// Parameters:    string &inF    A string containing the name of the input
//                               file to be used
//                string &outF   A string containing the name of the output
//                               file to be created
//
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
fileCopyClass(const string &inF, const string &outF) : f_filterBase( inF, outF)
   {}

  // Destructor ////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: ~fileCopyClass
// Summary:       The destructor. Destroys a fileCopyClass object.
//
// Parameters:    none
//
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
virtual ~fileCopyClass()
   { /* currently empty */ };

private:
  // Internal/Maintenance //////////////////////////////////////////////////////


public:
  // Accessors /////////////////////////////////////////////////////////////////


  // Mutators //////////////////////////////////////////////////////////////////

protected:
////////////////////////////////////////////////////////////////////////////////
//
// Function Name: transform
// Summary:       Preserves the character being read (i.e. copies it).
//
// Parameters:    char ch   The character that has been read
//
// Returns:       char   The unaltered character ch
//
////////////////////////////////////////////////////////////////////////////////
virtual char transform( char ch ) const
   { /* do nothing, preserve char */ return ch; };

  // Overloaded Operators //////////////////////////////////////////////////////

   };

#endif


