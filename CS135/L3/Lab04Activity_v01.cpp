// Header Files

#include "formatted_cmdline_io_v08.h"

// Global Constant Definitions

const int NO_BLOCK_SIZE = 0;
const int ONE_LINE = 1;
const int TWO_LINES = 2;

// Function Prototypes

/*
name: printTitle
process: display title with underline
Function Input/parameters: none
Function Output/parameters: none
Function Output/returned: none
Device Input: none
Device Output/Monitor: title displayed
dependencies: formatted command line I/O tools
*/
void printTitle();

/*
name: getInput
process: prompt user for input, acquire input, return
Function Input/parameters: name of input - e.g., "first", "second" (string)
Function Output/parameters: none
Function Output/returned: user input (int)
Device Input/Keyboard: user input
Device Output/Monitor: display of prompt
dependencies: formatted command line I/O tools
*/
int getInput( const string &inputName );

/*
name: calcProduct
process: calculate product of two numbers, return result
Function Input/parameters: two values (int)
Function Output/parameters: none
Function Output/returned: product of two numbers (int)
Device Input: none
Device Output: none
dependencies: none
*/
int calcProduct( int valOne, int valTwo );

/*
name: displayResult
process: display program result, including two input values
Function Input/parameters: two input values, product (int)
Function Output/parameters: none
Function Output/returned: none
Device Input: none
Device Output/Monitor: result with values displayed
dependencies: formatted command line I/O tools
*/
void displayResult( int valOne, int valTwo, int result );

// Main Program
int main()
   {
    // initialize program

       // initialize variables
       int firstNum, secondNum, product;

       // display title
          // function: iostream <<
       cout << "Multiplication Program" << endl;
       cout << "======================" << endl << endl;

    // prompt user for input

       // prompt for, acquire first value
          // function: iostream <<, >>
       cout << "Enter first number: ";
       cin >> firstNum;

       // prompt for, acquire second value
          // function: iostream <<, >>
       cout << "Enter first number: ";
       cin >> secondNum;

    // calculate product
       // operation: math
    product = firstNum * secondNum;

    // display result
       // function: iostream <<
    cout << endl << "The product of " << firstNum 
         << " and " << secondNum << " is: " << product
         << endl << endl;

    // end program

          // hold screen
             // function: system/pause
          system( "pause" );

       // return success to OS
       return 0;
   }

// Supporting Function Implementation

void printTitle()
   {
    // initialize function/variables
       // none

    // display first line of title
       // function: printString, printEndLines

    // display second line of title, with extra vertical space
       // function: printString, printEndLines
   }

int getInput( const string &inputName )
   {
    // initialize function/variables

    // print "Enter " pretext
       // function: printString

    // print input name ("first" or "second")
       // function: printString

    // prompt user for input with last part of string ("number: ")
       // function: promptForInt

    // return input value
    return 0; // temporary stub return
   }

int calcProduct( int valOne, int valTwo )
   {
    // initialize function/variables

    // calculate product
       // operation: math

    // return product of math
    return 0; // temporary stub return
   }

void displayResult( int valOne, int valTwo, int result )
   {
    // initialize function/variables
       // none
   
    // print "The product of " pretext
       // function: printString

    // print first integer
       // function: printInt

    // print " and " intermediate text
       // function: printString

    // print second integer
       // function: printInt

    // print " is: " pretext
       // function: printString

    // print result
       // function: printInt
   }


