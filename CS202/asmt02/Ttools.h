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
using namespace std;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//~ Macros //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#define ARR_SIZE( a ) (sizeof( a ) / sizeof( a[ 0 ] ))  // a MUST == an array


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


#endif
