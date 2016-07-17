#ifndef ___Ttools_h___
#define ___Ttools_h___


/////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      T-R0D's Toolbox
//  Created By: Terence Henriod
//
//  This header file and the accompanying .cpp file are a collection of self
//  written functions that are used relatively often.
//
/////////////////////////////////////////////////////////////////////////////////


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//~ Header Files ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <iostream>
#include <fstream>
#include <iomanip>
#include <conio.h>
#include <stdlib.h>
#include <string>
using namespace std;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//~ Macros //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#define ARR_SIZE( a ) (sizeof( a ) / sizeof( a[ 0 ] ))  // a MUST == an array


#pragma warning (disable : 4996)   // disables strcpy, strcat warnings in Visual

/////////////////////////////////////////////////////////////////////////////////
/// Functions ///////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//~ "system()" replacers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/////////////////////////////////////////////////////////////////////////////////
//
// Function Name: holdProg
// Summary:       Holds the program for the user without the use of the "evil"
//                use of system().
// Parameters: none
// Returns:    void
//
///////////////////////////////////////////////////////////////////////////////// 
void holdProg();

/////////////////////////////////////////////////////////////////////////////////
//
// Function Name: clrScr
// Summary:       Clears the terminal screen without the use of the "evil"
//                use of system().
// Parameters: none
// Returns:    void
//
///////////////////////////////////////////////////////////////////////////////// 
void clrScr();


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//~ Miscelaneous ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/////////////////////////////////////////////////////////////////////////////////
//
// Function Name: randBetw
// Summary:       Generates a random integer based on the given range. Generates
//                a number from the "inclusive range" (includes both end 
//                numbers).
// Parameters: int lo   The lower limit of the range
//             int hi   The higher limit of the range
// Returns:    int      The random integer generated
//
///////////////////////////////////////////////////////////////////////////////// 
int randBetw( int lo, int hi );


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: checkExt
// Summary:       This function can be used to check the end of a string for a 
//                string representing a file extension. Can be used generally to 
//                check the end of a string for the existence of any string,
//                provided that extension is shorter than fname
// Parameters:    char* fname            A C-string representing a file name to
//                                       be tested
//                const char* extension  A constant C-string representing a file 
//                                       extension
// Returns:       the boolean result of whether the extension is at the end of 
//                fname
//
////////////////////////////////////////////////////////////////////////////////
bool checkExt( char* fname, const char* extension );

bool openFile( const char* fname, ifstream &fobj );

bool openFile( const string fname, ifstream &fobj );

bool openFile( const char* fname, ofstream &fobj );

bool openFile( const string fname, ofstream &fobj );

void closeFile( ifstream &fobj );

void closeFile( ofstream &fobj );

#endif
