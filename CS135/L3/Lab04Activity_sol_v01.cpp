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
          // function: printTitle
       printTitle();

    // prompt user for input

       // prompt for, acquire first value
          // function: getInput
       firstNum = getInput( "first" );

       // prompt for, acquire second value
          // function: getInput
       secondNum = getInput( "second" );

    // calculate product
       // function: calcProduct
    product = calcProduct( firstNum, secondNum );

    // display result
       // function: displayResult
    displayResult( firstNum, secondNum, product );

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
    printString( "Multiplication Program", NO_BLOCK_SIZE, "LEFT" );
    printEndLines( ONE_LINE );

    // display second line of title, with extra vertical space
       // function: printString, printEndLines
    printString( "======================", NO_BLOCK_SIZE, "LEFT" );
    printEndLines( TWO_LINES );
   }

int getInput( const string &inputName )
   {
    // initialize function/variables
    int response;

    // print "Enter " pretext
       // function: printString
    printString( "Enter ", NO_BLOCK_SIZE, "LEFT" );

    // print input name ("first" or "second")
       // function: printString
    printString( inputName, NO_BLOCK_SIZE, "LEFT" );

    // prompt user for input with last part of string ("number: ")
       // function: promptForInt
    response = promptForInt( " number: " );

    // return input value
    return response;
   }

int calcProduct( int valOne, int valTwo )
   {
    // initialize function/variables
    int product;

    // calculate product
       // operation: math
    product = valOne * valTwo;

    // return product of math
    return product;
   }

void displayResult( int valOne, int valTwo, int result )
   {
    // initialize function/variables
       // none

    // print vertical space
       // function: printEndLines
    printEndLines( ONE_LINE );

    // print "The product of " pretext
       // function: printString
    printString( "The product of ", NO_BLOCK_SIZE, "LEFT" );

    // print first integer
       // function: printInt
    printInt( valOne, NO_BLOCK_SIZE, "LEFT" );

    // print " and " intermediate text
       // function: printString
    printString( " and ", NO_BLOCK_SIZE, "LEFT" );

    // print second integer
       // function: printInt
    printInt( valTwo, NO_BLOCK_SIZE, "LEFT" );

    // print " is: " pretext
       // function: printString
    printString( " is: ", NO_BLOCK_SIZE, "LEFT" );

    // print result
       // function: printInt
    printInt( result, NO_BLOCK_SIZE, "LEFT" );

    // end line and add extra vertical space
       // function: printEndLines
    printEndLines( TWO_LINES );
   }


