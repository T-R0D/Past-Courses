// Header Files/////////////////////////////////////////////////////////////////

   #include "formatted_cmdline_io_v08.h"
   #include <cmath> 
   using namespace std ;


// Global Constant Definitions//////////////////////////////////////////////////

   // character constants
      const char PIPE  = '|' ;
      const char SPACE = ' ' ;

   // string constants
      const string FIRST_PIPE = "     | " ;
      const string COLON      = ": "      ;
      const string ERROR      = "ERROR: Shape not found or input data incorrect - Program aborted" ;
   // numerical constants
      const double PI = 3.14159265359 ;
      const int PREC  = 2             ;

   // spacing constants
      const int NO_BLOCK     = 0  ;
      const int PROG_TIT     = 46 ;
      const int TITLE_BLOCK  = 37 ;
      const int COL_ONE      = 19 ;
      const int COL_TWO      = 17 ;
      const int ONE_LIN      = 1  ;
      const int TWO_LIN      = 2  ;

   // table piece constants
      const string THICK_SOLID = "     |======================================|" ;
      const string THICK_DIV   = "     |====================|=================|" ;
      const string THIN_DIV    = "     |--------------------|-----------------|" ;


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
Name: promptForLength
Process: prompts user for a length measurement
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: double     user input for length
Device Input: none
Device Output: none
Dependencies: formatted command line I/O tools
*/
double promptForLength();


/* 
Name: promptForWidth
Process: prompts user for a width measurement
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: double     user input for width
Device Input: none
Device Output: none
Dependencies: formatted command line I/O tools
*/
double promptForWidth();


/* 
Name: promptForHeight
Process: prompts user for a height measurement
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: double     user input for height
Device Input: none
Device Output: none
Dependencies: formatted command line I/O tools
*/
double promptForHeight();


/* 
Name: promptForRadius
Process: prompts user for a radius measurement
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: double     user input for radius
Device Input: none
Device Output: none
Dependencies: formatted command line I/O tools
*/
double promptForRadius();


/* 
Name: calcConeSA
Process: Calculates the surface area of a cone using given measurements
Function Input/Parameters: double radius     The radius of the circular base of the cone 
                           double height     The height of the cone when set on its base
Function Output/Parameters: none
Function Output/Returned: double     calculated surface area of the cone 
Device Input: none
Device Output: none
Dependencies: none
*/
double calcConeSA( double radius, double height) ;


/* 
Name: calcCubeSA
Process: Calculates the volume of a cube (or rectangular prism) using given measurements
Function Input/Parameters: double length     The length of the x edge of the cube
                           double width      The height of the y edge of the cube
                           double height     The radius of the z edge of the cube 
Function Output/Parameters: none
Function Output/Returned: double     calculated volume of the cube (rectangular prism)
Device Input: none
Device Output: none
Dependencies: none
*/
double calcCubeSA( double length, double width, double height ) ;


/* 
Name: calcCylindSA
Process: Calculates the surface area of a cylinder using the given measurements
Function Input/Parameters: double radius     The radius of the circular base of the cylinder 
                           double height     The height of the cylinder when set on its base
Function Output/Parameters: none
Function Output/Returned: double     the calculated surface area of the cylinder
Device Input: none
Device Output: none
Dependencies: none
*/
double calcCylindSA( double radius, double height ) ;


/* 
Name: calcSphereSA
Process: calculates the surface area of a sphere using the given measurements
Function Input/Parameters: double radius     The radius of the sphere
Function Output/Parameters: none
Function Output/Returned: double     the calculated surface area of the sphere
Device Input: none
Device Output: none
Dependencies: none
*/
double calcSphereSA( double radius ) ;


/* 
Name: displayErrorMess
Process: checks user input for shape and the calculated surface area to see if
         they are valid 
Function Input/Parameters: const string &shape     the input name of the shape
                                                   the surface area is calculated for
                           double surfaceA         the calculated surface area for the 
                                                   desired shape
Function Output/Parameters: none
Function Output/Returned: bool     returns true/false depending on whether or not
                                   invalid input/data has been passed to the function
Device Input: none
Device Output: displays the error message if invalid input/data is present,
               otherwise, does nothing
Dependencies: formatted command line I/O tools
*/
bool displayErrorMess( const string &shape, double surfaceA ) ;


/* 
Name: displayResults
Process: accepts the shape name, measurements, and calculated surface area and displays the values
         in a table; assumes that the input parameters are valid
Function Input/Parameters: const string &shape_name     The name of the shape input by the user 
                           double length                The length measuremententered by the user;
                                                        if no such value is entered, it will not 
                                                        be processed/displayed
                           double width                 The width measuremententered by the user;
                                                        if no such value is entered, it will not 
                                                        be processed/displayed
                           double height                The height measuremententered by the user;
                                                        if no such value is entered, it will not 
                                                        be processed/displayed
                           double radius                The radius measuremententered by the user;
                                                        if no such value is entered, it will not 
                                                        be processed/displayed
                           double surfaceA              The surface area calculated for the input shape
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: displays the "Results Table"
Dependencies: formatted command line I/O tools
*/
void displayResults( const string &shape, double length, double width, double height, double radius, double surfaceA ) ;


/* 
Name: printOneResultLine
Process: accepts the results of the appropriate computation and 
Function Input/Parameters: string shape          the text that will be displayed
                                                 in the first cloumn of the 
                                                 table.
                           double value          the value that will appear in
                                                 the second column of the table.
                                                 *Ideally* corresponds to the 
                                                 text in the first column.
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: displays the shape name and result for the entered value.
Dependencies: formatted command line I/O tools
*/
void printOneResultLine( const string &shape, double value ) ;


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


// Instructor Provided Functions //////////////////////////////////////////////

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

// End Instructor Provided Functions ///////////////////////////////////////////



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
         // function: printString, promptForString, printEndLines
 
      // capitalize shape name
         // function: toUpperCaseWord

   // get desired shape's measurements
      // operations: if, else

         // if entered shape is cone

            // prompt for radius of base
               // function: promptForRadius

            // prompt for height of cone
               // function: promptForHeight

         // if entered shape is cube/rectangular prism
        
            // prompt for length
               // function: propmptForLength

            // prompt for width
               // function: propmptForWidth

            // prompt for height
               // function: propmptForHeight

         // if entered shape is cylinder

            // prompt for radius of base
               // function: promptForRadius

            // prompt for height of cone
               // function: promptForHeight
           

         // if entered shape is sphere
       
            // prompt for radius
               // function: propmptForRadius
       
   // calculate volume of shape
      // operations: if, else
         // if entered shape is cone

           // function: calcConeV


         // if entered shape is cube/rectangular prism
        
            // function: calcCubeV

         // if entered shape is cylinder

           // function: calcCylindV

         // if entered shape is sphere
      
            // function: calcShpereV
           
           // else statement is fulfilled by the coming if statement in error checking

   // check for errors
      // operations: if, else
     
         // if shape name is incorrect or measurements produce non-physical result
            // function: displayErrorScreen

              // display error message
                 // displayed by function already

              // hold system for user
                 // function: system
                 
              // end program by returning 0

         // otherwise, program proceeds normally
              // do nothing, program continues normally

   // display result

      // display results table
         // function: displayResults

   // hold screen for user, return 0
      
      // hold system for user
         // function: system

      // return 0
      return 0;
   
   }

// Supporting function implementations//////////////////////////////////////////


void printTitle() 
   {
   // no return - void
   }

double promptForLength()
   {
   return 0; // temporary stub return    
   }

double promptForWidth()
   {
   return 0; // temporary stub return    
   }

double promptForHeight()
   {
   return 0; // temporary stub return    
   }

double promptForRadius()
   {
   return 0; // temporary stub return    
   }

double calcConeSA( double radius, double height )
   {
    return 0; // temporary stub return
   }


double calcCubeSA( double length, double width, double height )
   {
    return 0; // temporary stub return
   }


double calcCylindSA( double radius, double height )
   {
    return 0; // temprorary stub return
   }


double calcSphereSA( double radius )
   {
    return 0; // temporary stub return
   }


bool displayErrorMess( const string &shape, double surfaceA ) 
   {
   return true; // temporary stub return
   }


void displayResults( const string &shape, double length, double width, double height, double radius, double surfaceA ) 
   {
   // no return - void
   }


void printOneResultLine( const string &shape, double value ) 
   {
      // no return - void
   }


void printThinDividerLine() 
   {
      // no return - void
   }


void printThickDividerLine() 
   {

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








