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
Process: Calculates the volume of a cube using given measurement
Function Input/Parameters: double length     The length of any edge of the cube 
Function Output/Parameters: none
Function Output/Returned: double     calculated volume of the cube
Device Input: none
Device Output: none
Dependencies: none
*/
double calcCubeSA( double length ) ;


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
         string shape  = "NoNe EnTeReD";

         // numerical variables
         double length   = -999.9;
         double width    = -999.9;
         double height   = -999.9;
         double radius   = -999.9;
         double surfaceA = -999.9;

      // print title
         // function: printTitle
         printTitle();

   // prompt for desired shape

      // prompt user for shape name
         // function: printString, promptForString, printEndLines
         printString( "Enter name of shape", NO_BLOCK, "LEFT" );
         shape = promptForString( COLON );
         printEndLines( ONE_LIN );
 
      // capitalize shape name
         // function: toUpperCaseWord, printEndLines
         shape = toUpperCaseWord( shape );

   // get desired shape's measurements
      // operations: if, else

         // if entered shape is cone
         if( shape == "CONE" )
           {
            // prompt for radius of base
               // function: promptForRadius
               radius = promptForRadius();

            // prompt for height of cone
               // function: promptForHeight
               height = promptForHeight();
           }

         // if entered shape is cube
         if( shape == "CUBE" )
           {
            // prompt for length
               // function: propmptForLength
               length = promptForLength();
           }

         // if entered shape is cylinder
         if( shape == "CYLINDER" )
           {
            // prompt for radius of base
               // function: promptForRadius
               radius = promptForRadius();

            // prompt for height of cone
               // function: promptForHeight
               height = promptForHeight();
           }

         // if entered shape is sphere
         if( shape == "SPHERE" )
           {
            // prompt for radius
               // function: propmptForRadius
               radius = promptForRadius();
           }

   // calculate volume of shape
      // operations: if, else
         // if entered shape is cone
         if( shape == "CONE" )
           {   
           // function: calcConeV
           surfaceA = calcConeSA( radius, height );
           }

         // if entered shape is cube
         if( shape == "CUBE" )
           {
            // function: calcCubeV
            surfaceA = calcCubeSA( length );
           }

         // if entered shape is cylinder
         if( shape == "CYLINDER" )
           {
           // function: calcCylindV
           surfaceA = calcCylindSA( radius, height );
           }

         // if entered shape is sphere
         if( shape == "SPHERE" )
           {
            // function: calcShpereV
            surfaceA = calcSphereSA( radius );
           }

         else
           {
           // else statement is fulfilled by later error checking and following error message
           }

   // check for errors
      // operations: if, else
     
         // if shape name is incorrect or measurements produce non-physical result
            // function: displayErrorScreen
            if( displayErrorMess( shape, surfaceA ) )
              {
              // display error message
                 // displayed by function already

              // hold system for user
                 // function: system
                 system( "PAUSE" );
                 
              // end program by returning 0
              return 0;
              }

         // otherwise, program proceeds normally
            else
              {
              // do nothing, program continues normally
              }

   // display result

      // display results table
         // function: displayResults
         displayResults( shape, length, width, height, radius, surfaceA );
   // hold screen for user, return 0
      
      // hold system for user
         // function: system
         system( "PAUSE" );

      // return 0
      return 0;
   }


// Supporting function implementations//////////////////////////////////////////

void printTitle() 
   {
   // intitialize function/variables
      // none

   // print title text
      // functions: printString, printEndLines
         printString( "SHAPE CALCULATING FUNCTION", PROG_TIT, "CENTER" );  
         printEndLines( ONE_LIN );

   // print underline
      // functions: printString, printEndLines
         printString( "==========================", PROG_TIT, "CENTER" );
         printEndLines( TWO_LIN );

   // end function
      // no return - void
   }

double promptForLength()
   {
   // initialize variables
      double entry = -888.8;

   // prompt user for length
      // functions: printString, promptForDouble, printEndLines
      printString( "Enter length", NO_BLOCK, "LEFT" );
      entry = promptForDouble( COLON );
      printEndLines( ONE_LIN );

   // return entry
   return entry;    
   }

double promptForWidth()
   {
   // initialize variables
      double entry = -888.8;

   // prompt user for width
      // functions: printString, promptForDouble, printEndLines
      printString( "Enter width", NO_BLOCK, "LEFT" );
      entry = promptForDouble( COLON );
      printEndLines( ONE_LIN );

   // return entry
   return entry;    
   }

double promptForHeight()
   {
   // initialize variables
      double entry = -888.8;

   // prompt user for height
      // functions: printString, promptForDouble, printEndLines
      printString( "Enter height", NO_BLOCK, "LEFT" );
      entry = promptForDouble( COLON );
      printEndLines( ONE_LIN );

   // return entry
   return entry;    
   }

double promptForRadius()
   {
   // initialize variables
      double entry = -888.8;

   // prompt user for radius
      // functions: printString, promptForDouble, printEndLines
      printString( "Enter radius", NO_BLOCK, "LEFT" );
      entry = promptForDouble( COLON );
      printEndLines( ONE_LIN );

   // return entry
   return entry;    
   }

double calcConeSA( double radius, double height )
   {
    // initialize variables
    double result     = -888.8;
    double baseA      = -888.8;
    double hypotenuse = -888.8;
    double shaftA     = -888.8;

    // perform calculation
       // operations: math
       baseA = (PI * pow( radius, 2.0 ) );
       hypotenuse = pow( ( pow(radius, 2.0) + pow(height, 2.0) ), .5 );
       shaftA = PI * radius * hypotenuse;
       result = baseA + shaftA;

    // error check: if an inout parameter results in a non-physical shape, invalidate 
    // the calculation
       // operation: if
       if( radius <= 0 || height <= 0 )
         {
          return 0;
         }
 
    // return result
    return result;
   }


double calcCubeSA( double length )
   {
    // initialize variables
    double result = -888.8;

    // perform calculation
       // operations: math

       result = ( 6 * pow( length, 2.0 ) );

    // return result
    return result;
   }


double calcCylindSA( double radius, double height )
   {
    // initialize variables
    double result = -888.8;
    double baseA  = -888.8;
    double shaftA = -888.8;

    // perform calculation
       // operations: math
       baseA = (PI * pow( radius, 2.0 ) );
       shaftA = 2 * PI * radius * height ;
       result = (2 * baseA) + shaftA ;

    // error check: if an inout parameter results in a non-physical shape, invalidate 
    // the calculation
       // operation: if
       if( radius <= 0 || height <= 0 )
         {
          return 0;
         }
 
    // return result
    return result;
   }


double calcSphereSA( double radius )
   {
    // initialize variables
    double result = -888.8;

    // perform calculation
       // operations: math
       result = 4 * PI * pow( radius, 2.0 );
 
    // return result
    return result;
   }


bool displayErrorMess( const string &shape, double surfaceA ) 
   {
   // intialize variables
   bool result = true;

   // test the input
      // operations: if, else

      // if bad input is entered
      if( ( shape != "CONE" && shape != "CUBE" && shape != "CYLINDER"
         && shape != "SPHERE" ) || surfaceA <= 0 )
        {
         // print error message
            // functions: printString, printEndLines
            printEndLines( ONE_LIN );
            printString( ERROR, NO_BLOCK, "LEFT" );
            printEndLines( TWO_LIN );
            printEndLines( TWO_LIN );
         // return true so program can continue knowing an error is present
         result = true;
         return result;
        }

      // otherwise
      else
        {
        // return false to indicate no errors are present
        result = false;
        return result;
        }
   }


void displayResults( const string &shape, double length, double width, double height, double radius, double surfaceA ) 
   {
    // print title box
       // functions: printThickSolidLine, printEndLines, printString, printChar, printThickDividerLine
       printThickSolidLine();
       printString( FIRST_PIPE, NO_BLOCK, "LEFT" );
       printString( "SHAPE CALCULATION DISPLAY", TITLE_BLOCK, "CENTER" );
       printChar( PIPE );
       printEndLines( ONE_LIN );
       printThickDividerLine();
 
    // print subtitles
       // functions: printThickDividerLine, printString, printChar, printEndLines, printThinDividerLine
       printThickDividerLine();
       printString( FIRST_PIPE, NO_BLOCK, "LEFT" );
       printString( "Measurement", COL_ONE, "CENTER" );
       printChar( PIPE );
       printString( "Value", COL_TWO, "CENTER" );
       printChar( PIPE );
       printEndLines( ONE_LIN );
       printThinDividerLine();
     
    // print input measurements
       // operation: if

          // if valid radius was entered
          if( radius > 0 )
            {
            // print measurement line
               // function: printOneResultLine
               printOneResultLine( "Radius", radius );
            }

          // if valid length was entered
          if( length > 0 )
            {
            // print measurement line
               // function: printOneResultLine
               printOneResultLine( "Length", length );
            }

          // if valid width was entered
          if( width > 0 )
            {
            // print measurement line
               // function: printOneResultLine
               printOneResultLine( "Width", width );
            }

          // if valid height was entered
          if( height > 0 )
            {
            // print measurement line
               // function: printOneResultLine
               printOneResultLine( "Height", height );
            }

    // print calculated volume
       // functions: printThickDividerLine, printString, printChar, printEndLines, printThinDividerLine, printOneResultLine
       printThickDividerLine();
       printString( FIRST_PIPE, NO_BLOCK, "LEFT" );
       printString( "Shape", COL_ONE, "CENTER" );
       printChar( PIPE );
       printString( "Surface Area", COL_TWO, "CENTER" );
       printChar( PIPE );
       printEndLines( ONE_LIN );
       printThinDividerLine();
       printOneResultLine( shape, surfaceA );
       printEndLines( TWO_LIN );
   }


void printOneResultLine( const string &shape, double value ) 
   {
   // intitialize function/variables
      // none

   // print first text block with vertical borders
      // functions: printString, printChar
         printString( FIRST_PIPE, NO_BLOCK, "LEFT" );
         printString( shape, COL_ONE, "LEFT" );
         printChar( PIPE );

   // print second block
      // printDouble, printChar, printEndLines
         printDouble( value, PREC, COL_TWO, "CENTER" );
         printChar( PIPE );
         printEndLines( ONE_LIN );

   // print row divider line
      // printThinDividerLine
         printThickDividerLine();

   // end function
      // no return - void
   }


void printThinDividerLine() 
   {
   // intitialize function/variables
      // none

   // print the divider line
     // function: printString
        printString( THIN_DIV, NO_BLOCK, "LEFT" );
   
   // print an EndLine
     // function: printEndLines
        printEndLines( ONE_LIN );

   // end function
      // no return - void
   }


void printThickDividerLine() 
   {
   // intitialize function/variables
      // none

   // print the divider line
     // function: printString
        printString( THICK_DIV, NO_BLOCK, "LEFT" );

   // print an EndLine
     // function: printEndLines
        printEndLines( ONE_LIN );

   // end function
      // no return - void
   }


void printThickSolidLine()
   {
   // intitialize function/variables
      // none

   // print the divider line
     // function: printString
        printString( THICK_SOLID, NO_BLOCK, "LEFT" );

   // print an EndLine
     // function: printEndLines
        printEndLines( ONE_LIN );

   // end function
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








