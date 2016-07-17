///////////////////////////////////////////////////////////////////////////
/*
    Program:             Rectangle_3.cpp
    Author:              Jamie L. Student
    Update By:           Jamie L. Student
    Description:         Assignment 1 - program draws a rectangle made
                         from asterisks and spaces
  
    Program Written:     08/1/2008
    Most Recent Update:  08/15/2008 - 11:30 a.m.
    Date Due for Review: 08/26/2008
*/
///////////////////////////////////////////////////////////////////////////
/*
INSTRUCTIONS:
    
   This file can be used for the first laboratory.  It has several comments
   in it that explain how a program should look.  Comments are text parts of
   the program that are not compiled or used as part of the program. See
   comment information below.

   The area above these instructions is for programmer information; it is
   not required that you use this, but it is a good idea
   
*/

// There are two kinds of comments:

// Single-line comments can be typed like this with two forward slashes

/* 
Multiple-line comments can use the "slash-star" characters for the beginning, 
and the "star-slash" characters for the end.
*/

/*
END OF INSTRUCTIONS
*/

/*
Header files are necessary if you are using previously generated code
from other sources.  For example, you will need iostream to be able to
implement input and output actions with cin and cout.  You will need cmath
when you use functions that generate random numbers or functions that
calculate square roots, sines, cosines, etc.  You will also need to type
the "using namespace std:" after your include files
*/

// Header files
#include "formatted_console_io_v17.h"

using namespace std;

/*
Your global constants can go immediately below your header file declarations.
Note that global constants are an excellent professional practice in
software development; global variables on the other hand, are a very bad
practice, and should rarely, if ever, be used.  While there might be some
value to using global variables in other places, they should NEVER be used
in the CS 135 Course.
*/

// Global Constants
const char ASTERISK = '*';

/*
Function prototypes must be placed before they are used in the program, so
this is the best location for them, just before the main function. The 
compiler refers to these prototypes to make sure they are used correctly
in the program, so it must see them before it sees the functions being used.
*/

// Global Function Prototypes
   // none required
/*
The main function is the actual program. All actions start and end with
the main function, including other functions that will likely be called by
it. There can only be one main function in any program.
*/

int main()
   {
    /* All variables should be declared at the beginning of the function
       in which they are used. The C++ programming language allows them
       to be declared later in functions, but this is not a good practice.
       Note that the variables declared below are not used in this program;
       they are placed in the code as examples.
    */

    // Function initialization / Variable definition
       // Examples are shown, but commented out 
       // since they are not used in this program
       // However, it is necessary to initialize Curses
    // int myInteger;
    // double myDouble;
    startCurses();
 
    /* Program statements are made as needed in the program.
       Every major action in the program should have a comment
       in front of it for clarity.
    */

    // Print first box left justified at location 40
    // Output line 1
    printStringAt( 15, 4, "*", "CENTER" );

    // Output line 2
    printStringAt( 15, 5, "***", "CENTER" );

    // Output line 3
    printStringAt( 15, 6, "*****", "CENTER" );
   
    // Output line 4
    printStringAt( 15, 7, "*******", "CENTER" );

    // Output line 5
    printStringAt( 15, 8, "*****", "CENTER" );

    // Output line 6
    printStringAt( 15, 9, "***", "CENTER" );

    // Output line 7
    printStringAt( 15, 10, "*", "CENTER" );

    
    // End program
    //   hold the program, then end curses
    //   then end the program
    waitForInput( FIXED_WAIT );
    endCurses();

    /* The main function should always return 0 unless the program
       fails for some reason.
    */

    return 0;  // Return 0 to the operating system
               // to indicate succesful program completion
   }  //  end of main function

/*
All functions that you write for your program will be placed down below
the main function.  Since you are only using these functions and not
having to write them, you do not have to do anything with the following
functions. However, when you begin writing functions, you will be placing
them here.

When you start writing functions, you will also be required to write
some supporting comments for each one; this will be explained later.
*/ 

// Supporting Function Implementations



