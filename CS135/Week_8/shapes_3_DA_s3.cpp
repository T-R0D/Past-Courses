// Header Files/////////////////////////////////////////////////////////////////

   #include "formatted_cmdline_io_v08.h"
   #include <cmath> 
   using namespace std ;


// Global Constant Definitions//////////////////////////////////////////////////

   // character constants

   // string constants

   // numerical constants

   // spacing constants

   // table piece constants


// Global Function Prototypes///////////////////////////////////////////////////

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
Name:
Process: 
Function Input/Parameters: 
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: displays the "Results Table"
Dependencies: formatted command line I/O tools
*/
double calcConeV(                   gjlkglkjskfl                                   )


/* 
Name:
Process: 
Function Input/Parameters: 
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: displays the "Results Table"
Dependencies: formatted command line I/O tools
*/
double calcCubeV(                   gjlkglkjskfl                                   )


/* 
Name:
Process: 
Function Input/Parameters: 
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: displays the "Results Table"
Dependencies: formatted command line I/O tools
*/
double calcCylindV(                   gjlkglkjskfl                                   )


/* 
Name:
Process: 
Function Input/Parameters: 
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: displays the "Results Table"
Dependencies: formatted command line I/O tools
*/
double calcSphereV(                   gjlkglkjskfl                                   )


/* 
Name: 
Process: 
Function Input/Parameters: 
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: displays the "Results Table"
Dependencies: formatted command line I/O tools
*/
bool displayErrorScreen( string shape_name, double area ) ;


/* 
Name: displayResults
Process: 
Function Input/Parameters: 
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








// Main Program Definition//////////////////////////////////////////////////////
int main()
   { 

   // Initialize program and variables

      // initialize variables

         // char and string variables

         // numerical variables

      // print title
         // function: printTitle

   // prompt for desired shape

      // prompt user for shape name
         // function: promptForString, printEndLines

      // capitalize shape name
         // function: toUpperCaseWord, printEndLines

   // get desired shape's measurements

      // if entered shape is cone
        
         // prompt for radius of base
            // functions: promptForDouble, printEndLines

         // prompt for height of cone
            // functions: promptForDouble, printEndLines

      // if entered shape is cube/rectangular prism

         // prompt for length
            // functions: propmptForDouble, printEndLines

         // prompt for width
            // functions: propmptForDouble, printEndLines

         // prompt for height
            // functions: propmptForDouble, printEndLines

      // if entered shape is cylinder

         // prompt for radius
            // functions: propmptForDouble, printEndLines

         // prompt for height
            // functions: propmptForDouble, printEndLines

      // if entered shape is sphere

         // prompt for radius
            // functions: propmptForDouble, printEndLines

   // calculate volume of shape

      // if entered shape is cone
         // function: calcConeV

      // if entered shape is cube/rectangular prism
         // function: calcCubeV

      // if entered shape is cylinder
         // function: calcCylindV

      // if entered shape is sphere
         // function: calcShpereV

   // display result

      // display results table

   // hold screen for user, return 0

      // hold system for user

      // return 0
      return 0;
   
   }

// Supporting function implementations//////////////////////////////////////////


void printTitle() 
   {

   }

double calcConeV(                   gjlkglkjskfl                                   )
   {

   }


double calcCubeV(                   gjlkglkjskfl                                   )
   {

   }


double calcCylindV(                   gjlkglkjskfl                                   )
   {

   }


double calcSphereV(                   gjlkglkjskfl                                   )
   {

   }


bool displayErrorScreen( string shape_name, double area ) 
   {

   }


void displayResults( string shape_name, double length, double width, double height, double shape_area) 
   {

   }


void printOneResultLine( string shape_name, double shape_area ) 
   {


   }


void printThinDividerLine() 
   {

   }


void printThickDividerLine() 
   {

   }


void printThickSolidLine() 
   {

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








