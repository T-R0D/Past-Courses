#ifndef ___ENCRYPTCLASS_H___
#define ___ENCRYPTCLASS_H___

////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      encryptClass.h
//  Created By: Terence Henriod
//  Course:     CS202
//
//  Summary:    This class is a derived class of the pure abstract file filter
//              base class. This class is capable of creating an encrypted copy
//              of a given text file using an encryption key that is simply a
//              character offset for alphanumeric characters.
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

const int ALPHARANGE = 26;
const int DEFAULTKEY = 13;

class encryptClass: public f_filterBase
   {
   
private:
  // Data Members //////////////////////////////////////////////////////////////

unsigned int key;   // the character offset amount for encryption

public:
  // Constructor(s) ////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: encryptClass
// Summary:       The default constructor. Creates a new instance of a file
//                encryption class.
//
// Parameters:    none
//
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
encryptClass() : f_filterBase()
   {
   key = DEFAULTKEY;
   set_inF("encryptedFile.txt");   // default output name
   }

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: encryptClass
// Summary:       An overloaded constructor. Creates a new instance of a file
//                encryption class with the given file names and a given 
//                encryption key.
//
// Parameters:    unsigned int encryptKey   The encryption key to be used
//                string inF                The name of the input file to be 
//                                          used
//                string outF               The name of the output file to be 
//                                          used
//
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
encryptClass( unsigned int encryptKey, string &inF, string &outF ) 
            : f_filterBase( inF, outF )
   {
   // set key to a value within the alphabet range
   key = (encryptKey % ALPHARANGE);

   // if the given key will not perform encryption, set it to the default value
   if( key == 0 )
     {
     key = DEFAULTKEY;
     }
   }

  // Destructor ////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: ~encryptClass
// Summary:       The destructor for an encryptClass object. Currently does 
//                nothing
//
// Parameters:    none
//
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
virtual ~encryptClass();

private:
  // Internal/Maintenance //////////////////////////////////////////////////////


public:
  // Accessors /////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: viewKey
// Summary:       Grants access to the encryption key of the object to a scope
//                outside that of the object. Simply returns the key.
//
// Parameters:    none
//
// Returns:       unsigned int   The current value of the encryption key
//
////////////////////////////////////////////////////////////////////////////////
unsigned int viewKey() const;

  // Mutators //////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: set_key
// Summary:       Allows the encryption key to be set with the given parameter.
//                If the value will not result in encryption, a random one
//                be provided.
//
// Parameters:    unsigned int k   The value for the encryption key
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
void set_key(const unsigned int k);


protected:

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: transform
// Summary:       The default constructor. Creates a new instance of a doubly
//                linked list with no nodes.
//
// Parameters:    char ch   The character to be transformed 
//
// Returns:       char the character after it has been transformed
//
////////////////////////////////////////////////////////////////////////////////
virtual char transform( char ch ) const;

  // Overloaded Operators //////////////////////////////////////////////////////

   };

#endif


