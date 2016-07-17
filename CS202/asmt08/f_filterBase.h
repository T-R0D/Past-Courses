#ifndef ___F_FILTERBASE_H___
#define ___F_FILTERBASE_H___

////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      f_filterBase.h
//  Created By: Terence Henriod
//  Course:     CS202
//
//  Summary:    An abstract base class for file filtering operations.
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
#include <string>
#include "Ttools.h"
using namespace std;

// parent class header


//============================================================================//
//= Class Definition =========================================================//
//============================================================================//

class f_filterBase
   {
   
private:
  // Data Members //////////////////////////////////////////////////////////////

string inFName;
string outFName;

protected:
char currChar;  // protected so only derived classes can act on this,
                // preferably via the overridden transform function

public:
  // Constructor(s) ////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: f_filterBase
// Summary:       The default constructor. Creates a new instance of the
//                abstract base class as necessary for instntiating derived
//                class objects
//
// Parameters:    none
//
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
f_filterBase();

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: f_filterBase
// Summary:       An overloaded constructor. Creates a new instance of the
//                abstract base class as necessary for instntiating derived
//                class objects using the given file names as parameters.
//
// Parameters:    string &inF    The name of the input file to be filtered
//                string &outF   The name of the output file to be created
//
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
f_filterBase( const string &inF, const string &outF );

  // Destructor ////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: ~f_filterBase
// Summary:       The destructor. Performs any necessary destruction of the 
//                abstract base class after the derived classes are destructed
//
// Parameters:    none
//
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
virtual ~f_filterBase();

private:
  // Internal/Maintenance //////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: readWrite
// Summary:       Performs the reading and writing actions of the file 
//                filtering. Utilizes fstream objects passed from outside the 
//                object.
//
// Parameters:    ifstream &in    The reading file stream object
//                ifstream &out   The outputting file stream object
//
// Returns:       bool   Indicates whether or not filtering was successful
//                       (at least one character read/written) 
//
////////////////////////////////////////////////////////////////////////////////
bool readWrite(ifstream &in, ofstream &out);


public:
  // Accessors /////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: view_inF
// Summary:       Grants access to/returns the input file name member.
//
// Parameters:    none
//
// Returns:       string   A string conatining the current input file name.
//
////////////////////////////////////////////////////////////////////////////////
string view_inF() const;

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: view_outF
// Summary:       Grants access to/returns the output file name member.
//
// Parameters:    none
//
// Returns:       string   A string conatining the current output file name.
//
////////////////////////////////////////////////////////////////////////////////
string view_outF() const;

  // Mutators //////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: set_inF
// Summary:       Updates the input file name string to be used.
//
// Parameters:    string &inf   A string class object containing the name to
//                              used
//
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
void set_inF( const string &inF );

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: set_outF
// Summary:       Updates the ouput file name string to be used.
//
// Parameters:    string &outf   A string class object containing the name to
//                               used
//
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
void set_outF( const string &outF );

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: doFilter
// Summary:       Utilizes externally provided fstream objects to conduct the
//                file filtering process. Should only execute after the input
//                file name has been deemed valid.
//
// Parameters:    ifstream &in    The reading file stream object
//                ofstream &out   The writing file stream object
//
// Returns:       bool   Indicates the success of the operation (at least one
//                       character filtered)
//
////////////////////////////////////////////////////////////////////////////////
bool doFilter(ifstream &in, ofstream &out);

protected:
////////////////////////////////////////////////////////////////////////////////
//
// Function Name: transform
// Summary:       The pure abstract function that makes the class pure and 
//                abstract. This function, when overridden, will perform
//                any necessary character transformation. THIS WILL BE
//                OVERRIDDEN IN DERIVED CLASSES.
//
// Parameters:    char ch   The character to be transformed
//
// Returns:       char   The transformed character
//
////////////////////////////////////////////////////////////////////////////////
virtual char transform( char ch ) const = 0;  //  <- makes class pure, abstract

public:
  // Overloaded Operators //////////////////////////////////////////////////////

   };

#endif


