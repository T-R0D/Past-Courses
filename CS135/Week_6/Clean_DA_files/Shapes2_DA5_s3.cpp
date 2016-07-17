// Header Files
   #include "formatted_cmdline_io_v08.h"
   #include <cmath> 
   using namespace std ;

// Global Constant Definitions

   // spacing constants

   // numerical constants for calculations


   // table piece strings

   // character and other string constants

// Global Function Prototypes

/* 
Name: printTitle 
Process: prints the program title and its underline
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: title displayed
Dependencies: formatted command line I/O tools
*/
void printTitle() ;


/* 
Name: printRequiredData
Process: displays the possibly shapes whose area can be calculated, the 
         necessary input required for said calculations, and any necessary 
         instructions, all in tabular form. 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: displays the "RequiredData" table
Dependencies: formatted command line I/O tools
*/
void printRequiredData() ;


/* 
Name: printOneMeasurementLine 
Process: prints the information, columnar and lower box border table parts, and prepares
         for a new line
Function Input/Parameters: string shape         the name of the shape to be displayed
                           string instructions  the specification of the necessary information 
                                                required to calculate the area of the 
                                                previously specified shape
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: displays each row of the "RequiredData" table
Dependencies: formatted command line I/O tools
*/
void printOneMeasurementLine( string shape, string instructions ) ;


/* 
Name: calcArea
Process: tests the input shape name to determine whether or not it falls
         within the scope of the program or if the area result is a
         non-physical solution.
Function Input/Parameters: string shape_name     The desired shape name input
                                                 by the user.
                           double area           The calculated result of the 
                                                 area of the desired shape.
Function Output/Parameters: bool true/false
Function Output/Returned: none
Device Input: none
Device Output: displays "Error Screen."
Dependencies: formatted command line I/O tools
*/
bool displayErrorScreen( string shape_name, double area ) ;


/* 
Name: calcArea
Process: accepts user input values and calculated the desired shape's area
Function Input/Parameters: string shape_name     The desired shape name input
                                                 by the user.
                           double length         The length (or similar) measurement
                                                 value entered by the user.
                           double width          The width (or similar) measurement
                                                 value entered by the user.
                           double height         The height (or similar) measurement
                                                 value entered by the user.
                           double shape_area     The area calculated for the desired
                                                 shape to be displayed.
Function Output/Parameters: none
Function Output/Returned: calculated value for the desired shape's area
Device Input: none
Device Output: title displayed
Dependencies: formatted command line I/O tools, cmath
*/
double calcArea( string shape_name, double length, double width, double height ) ;


/* 
Name: displayResults
Process: Accepts user input and displays the results table in
         accordance with the user's input
Function Input/Parameters: string shape_name     The desired shape name input
                                                 by the user.
                           double length         The length (or similar) measurement
                                                 value entered by the user.
                           double width          The width (or similar) measurement
                                                 value entered by the user.
                           double height         The height (or similar) measurement
                                                 value entered by the user.
                           double shape_area     The area calculated for the desired
                                                 shape to be displayed.
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: displays the "Results Table"
Dependencies: formatted command line I/O tools
*/
void displayResults( string shape_name, double length, double width, double height, double shape_area) ;


/* 
Name: printOneResultLine
Process: accepts the results of the appropriate computation and 
Function Input/Parameters: string shape_name     the text that will be displayed
                                                 in the first cloumn of the 
                                                 table.
                           double shape_area     the value that will appear in
                                                 the second column of the table.
                                                 *Ideally* corresponds to the 
                                                 text in the first column.
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: displays the shape name and result for the area calculation.
Dependencies: formatted command line I/O tools
*/
void printOneResultLine( string shape_name, double shape_area ) ;


/* 
Name: printThinDividerLine
Process: prints a thin line that includes a pipe (to divide the columns) to 
         separate rows from one another.
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: prints a thin line with a column divider
Dependencies: formatted command line I/O tools
*/
void printThinDividerLine() ;


/* 
Name: printThickDividerLine
Process: prints a thick line that includes a pipe (to divide the columns) to 
         separate rows from one another.
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: prints a thick line with a column divider
Dependencies: formatted command line I/O tools
*/
void printThickDividerLine() ;


/* 
Name: printThickSolidLine
Process: prints a thick line to 
         separate rows from one another, or begin or end a table.
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: prints a thick table line
Dependencies: formatted command line I/O tools
*/
void printThickSolidLine() ;


/*
name: toUpperCaseLetter
process: if letter passed is lower case, converts to upper case,
         has no effect otherwise
Function Input/parameters: letter character (char)
Function Output/parameters: none
Function Output/returned: upper case letter character, as appropriate (char)
Device Input: none
Device Output: none
dependencies: none
*/
char toUpperCaseLetter( char letterChar );


/*
name: toUpperCaseWord
process: if word passed has any lower case letters, 
         converts to individual letters to upper case,
         has no effect otherwise
Function Input/parameters: word (string)
Function Output/parameters: none
Function Output/returned: upper case letter word, as appropriate (string)
Device Input: none
Device Output: none
dependencies: toUpperCaseLetter, string management tools
*/
string toUpperCaseWord( const string &word );



// Main Program Definition
int main()
   { 
   
   // initialize program

      // initialize variables

      // print program title
         // function: printTitle
  
   // display screen 1

      // display informational table that specifies the shapes the program can accomodate,
      // as well as the necessary data for each shape    
         // function: printRequiredData 
  
   // collect data

      // prompt user for desired shape type ---> convert string to capitalized statement
         // functions: printString, promptForString, toUpperCaseWord

      // prompt user for length or bottom base data
         // functions: printString, promptForDouble

      // prompt user for width, span, or top base data
         // functions: printString, promptForDouble
 
      // prompt user for height or radius data  
         // functions: printString promptForDouble

   // calculate desired shape's area
      // function: calcArea

   // verify that shape name and area fall within the scope of program, if not
   // terminate the program
      // function: displayErrorScreen, if, exit


   // display result

      // display results table 
         // function: displayResults   

   // end program

      // hold program for user viewing
         // function: system

      // return 0
         return 0;
   }



// Supporting function implementations

void printTitle()
   {
      // no return - void
   }


void printRequiredData() 
   {
      // no return - void
   }


void printOneMeasurementLine( string shape, string instructions ) 
   {
      // no return - void
   }


bool displayErrorScreen( string shape_name, double area )
   {
   return 0; // temporary stub return
   }


double calcArea( string shape_name, double length, double width, double height )
   {
   return 0; // temproary stub return 
   }


void displayResults(  string shape_name, double length, double width, double height, double shape_area  ) 
   {
      // no return - void
   }


void printOneResultLine( string shape_name, double shape_area ) 
   {
      // no return - void
   }


void printThinDividerLine() 
   {
      // no return - void
   }


void printThickDividerLine() 
   {
   // intitialize function/variables
      // none

   // print the divider line
     // function: printString

   // print an EndLine
     // function: printEndLines

   // end function
      // no return - void
   }


void printThickSolidLine()
   {
      // no return - void
   }


// ========================================================================== //
//  functions given to students - start ===================================== //
// ========================================================================== //

char toUpperCaseLetter( char letterChar )
   {
    // initialize function/variables
       // none

    // check for lower case letter
    if( letterChar >= 'a' && letterChar <= 'z' )
       {
        // change lower case letter to upper case
        letterChar = letterChar - 'a' + 'A';
       }

    // return character
    return letterChar;
   }

string toUpperCaseWord( const string &word )
   {
    // initialize function/variables

       // create new local word
       string upperCaseWord;

       // initialize index to zero
       unsigned index = 0;

    // iterate to end of word    
    while( index < word.length() )
       {
        // set lower case letter to upper case, if needed
           // function: toUpperCaseLetter
        upperCaseWord += toUpperCaseLetter( word.at( index ) );

        // increment index
        index++;
       }

    // return upper case word
    return upperCaseWord;
   }

// ========================================================================== //
//  functions given to students - end ======================================= //
// ========================================================================== //



