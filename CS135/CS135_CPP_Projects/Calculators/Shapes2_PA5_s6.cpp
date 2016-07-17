// Header Files
   #include "formatted_cmdline_io_v08.h"
   #include <cmath> 
   using namespace std ;

// Global Constant Definitions

   // spacing constants
      const int NO_BLOCK     = 0  ;
      const int PROG_TIT     = 66 ;
      const int TITLE_BLOCK  = 54 ;
      const int PROMPT_BLOCK = 47 ;
      const int COL_ONE      = 19 ;
      const int COL_TWO      = 34 ;
      const int ONE_LIN      = 1  ;
      const int TWO_LIN      = 2  ;

   // numerical constants for calculations
      const double PI = 3.14159265359 ;
      const int PREC  = 2             ;

   // table piece strings
      const string THICK_SOLID = "     |=======================================================|" ;
      const string THICK_DIV   = "     |====================|==================================|" ;
      const string THIN_DIV    = "     |--------------------|----------------------------------|" ;

   // character and other string constants
      const char PIPE = '|' ;
      const string FIRST_PIPE = "     | " ;
      const char SPACE = ' ';
      const string COLON = ": " ;

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
         string shape_desired = "None entered!" ; 
         double length        = -999 ;
         double width         = -999 ;
         double height        = -999 ;
         double area          = -999 ;
         bool errorTest       = false;
            /* Note: these values are just arbitrary initial values to ensure the 
               variables are not storing garbage, which will ideally help in 
               preventing any confusion if debugging needs to be performed.   */

      // print program title
         // function: printTitle
            printTitle();

   // display screen 1

      // display informational table that specifies the shapes the program can accomodate,
      // as well as the necessary data for each shape    
         // function: printRequiredData 
            printRequiredData(); 

   // collect data

      // prompt user for desired shape type ---> convert string to capitalized statement
         // functions: printString, promptForString, toUpperCaseWord
            printString( "     Enter shape name (e.g., TRIANGLE, CIRCLE) ", PROMPT_BLOCK, "LEFT" );
            shape_desired = promptForString( COLON );
            shape_desired = toUpperCaseWord( shape_desired );


      // prompt user for length or bottom base data
         // functions: printString, promptForDouble
            printString( "     Enter length or bottom base of shape ", PROMPT_BLOCK, "LEFT" );
            length = promptForDouble( COLON );

      // prompt user for width, span, or top base data
         // functions: printString, promptForDouble
            printString( "     Enter width, span, or top base of shape ", PROMPT_BLOCK, "LEFT" );
            width = promptForDouble( COLON );

      // prompt user for height or radius data  
         // functions: printString promptForDouble
            printString( "     Enter height or radius of shape ", PROMPT_BLOCK, "LEFT" );
            height = promptForDouble( COLON );


   // calculate desired shape's area
      // function: calcArea
         area = calcArea( shape_desired, length, width, height );

   // verify that shape name and area fall within the scope of program, if not
   // terminate the program
      // function: displayErrorScreen
         errorTest = displayErrorScreen( shape_desired, area );
         if( errorTest )
           {
           // end program
           return 0;
           }
         else
           {
            // do nothing
           }

   // display result

      // display results table 
         // function: displayResults   
            displayResults(  shape_desired, length, width, height, area );

   // end program

      // hold program for user viewing
         // function: system
            system( "PAUSE" );

      // return 0
         return 0;
   }



// Supporting function implementations

void printTitle()
   {
   // intitialize function/variables
      // none

   // print title text
      // functions: printString, printEndLines
         printString( "TWO DIMENSIONAL SHAPE AREA CALCULATOR", PROG_TIT, "CENTER" );  
         printEndLines( ONE_LIN );

   // print underline
      // functions: printString, printEndLines
         printString( "=====================================", PROG_TIT, "CENTER" );
         printEndLines( TWO_LIN );

   // end function
      // no return - void
   }


void printRequiredData() 
   {
   // initialize function/variables
      // none

   // print title row of table
      // functions: printThickSolidLine, printString, printChar,
      //            printEndLines, printThinDividerLine
         printThickSolidLine();
         printString( FIRST_PIPE, NO_BLOCK, "LEFT");
         printString( "Data Required for Each Object", TITLE_BLOCK, "CENTER" );
         printChar( PIPE );
         printEndLines( ONE_LIN );
         printThinDividerLine();

   // print subtitles
     // functions: printString, printChar, printEndLines,
     //            printThickDividerLine
        printString( FIRST_PIPE, NO_BLOCK, "LEFT" );
        printString( "Shape Name", COL_ONE, "CENTER" );
        printChar( PIPE );
        printString( "Measurement Needed", COL_TWO, "CENTER" );
        printChar( PIPE );
        printEndLines( ONE_LIN );
        printThickDividerLine();

   // print rectangle row
     // function: printOneMeasurementLine 
        printOneMeasurementLine( "Rectangle", "Length, Width, Height" );

   // print triangle row
     // function: printOneMeasurementLine 
        printOneMeasurementLine( "Triangle", "Bottom Base, Height" );

   // print trapezoid row
     // function: printOneMeasurementLine 
        printOneMeasurementLine( "Trapezoid", "Bottom Base, Top Base, Height" );

   // print parallelogram row
     // function: printOneMeasurementLine 
        printOneMeasurementLine( "Parallelogram", "Bottom Base, Height" );

   // print octaogon row
     // function: printOneMeasurementLine 
        printOneMeasurementLine( "Octagon", "Bottom Base, Span" );

   // print circle row
     // function: printOneMeasurementLine 
        printOneMeasurementLine( "Circle", "Radius" );

   // print instruction row
     // functions: printString, printChar, printEndLines, printThickSolidLine
        printString( FIRST_PIPE, NO_BLOCK, "LEFT" );
        printString( "If data not needed for shape, enter zero", TITLE_BLOCK, "CENTER" );
        printChar( PIPE );
        printEndLines( ONE_LIN );
        printThickSolidLine();
        printEndLines( ONE_LIN );

   // end function
      // no return - void
   }


void printOneMeasurementLine( string shape, string instructions ) 
   {
   // intitialize function/variables

   // print table first text block and vertical borders
      // functions: printString, printChar
         printString( FIRST_PIPE, NO_BLOCK, "LEFT" );
         printString( shape, COL_ONE, "LEFT" );
         printChar( PIPE );

   // print second block
      // functions: printString, printChar, printEndLines
         printString( instructions, COL_TWO, "CENTER" );
         printChar( PIPE );
         printEndLines( ONE_LIN );

   // print bottom divider line
      // function: printThinDividerLine
         printThinDividerLine();

   // end function
      // no return - void
   }


bool displayErrorScreen( string shape_name, double area )
   {
    // Test shape name, display error screen if input is invalid
    if( ( (shape_name != "RECTANGLE" ) && (shape_name != "TRIANGLE") && (shape_name != "TRAPEZOID")
           && (shape_name != "PARALLELOGRAM") && (shape_name != "OCTAGON") 
           && (shape_name != "CIRCLE") ) || (area <= 0) )
       {
        // display error screen
           // functions: printThickSolidLine, printString, printChar, printEndLines, system
              system( "CLS" );
              printThickSolidLine();
              printString( FIRST_PIPE, NO_BLOCK, "LEFT" );
              printString( "!!! ERROR !!!", TITLE_BLOCK, "CENTER" );
              printChar( PIPE );
              printEndLines( ONE_LIN );
              printString( FIRST_PIPE, NO_BLOCK, "LEFT" );
              printString( "Check the shape name or measurements, a mispelled", TITLE_BLOCK, "CENTER" );
              printChar( PIPE );
              printEndLines( ONE_LIN );
              printString( FIRST_PIPE, NO_BLOCK, "LEFT" );
              printString( "shape name or non-physical area may be present...", TITLE_BLOCK, "CENTER" );
              printChar( PIPE );
              printEndLines( ONE_LIN );
              printString( FIRST_PIPE, NO_BLOCK, "LEFT" );
              printString( "!!! PROGRAM WILL NOW TERMINATE !!!", TITLE_BLOCK, "CENTER" );
              printChar( PIPE );
              printEndLines( ONE_LIN );
              printThickSolidLine();
              printEndLines( TWO_LIN );
              system( "PAUSE" );

          // return true
             return true;
       }
    else
       {
       // do nothing, program will terminate in main
       return false;
       }
   }


double calcArea( string shape_name, double length, double width, double height )
   {
   // identify shape desired and calculate area

      /* Note: to make the math less complicated, I simply used 1 variable to
               correspond to each of the 3 prompts in the program. It should 
               be noted that:
               length represents the length or bottom base;
               width represents the width, span or top base;
               the height represents the height or radius.
               This means that the variable names may not match the proper 
               formula, but the code should result in the proper area 
               calculation.   */

       // initialize any necessary variables
          double area          = -999 ;
          double oct_perimeter = -999 ;
          double apothem       = -999 ;

      // if string is "RECTANGLE"
         // function: if
         // operations: math
         if( shape_name == "RECTANGLE" )
           {
            area = (length * width);
            return area;
           }

      // if string is "TRIANGLE"
         // function: if
         // operations: math
         if( shape_name == "TRIANGLE" )
           {
            area = 0.5 *(length * height);
            return area;
           }

      // if string is "TRAPEZOID"
         // function: if
         // operations: math
         if( shape_name == "TRAPEZOID" )
           {
            area = (.5 * ( length + width ) * height);
            return area;
           }

      // if string is "PARALLELOGRAM"
         // function: if
         // operations: math
         if( shape_name == "PARALLELOGRAM" )
           {
            area = (length * height) ;
            return area;
           }

      // if string is "OCTAGON"
         // function: if
         // operations: math
         if( shape_name == "OCTAGON" )
           {
            oct_perimeter = (8 * length);
            apothem = (.5 * width );
            area = (.5 * apothem * oct_perimeter );
            return area;
           }

      // if string is "CIRCLE"
         // function: if
         // operations: math
         if( shape_name == "CIRCLE" )
           {
            area = (PI * pow( height, 2.0 )) ;
            return area;
           }
   }


void displayResults(  string shape_name, double length, double width, double height, double shape_area  ) 
   {
   // intitialize function/variables
      // none

   // clear screen
      // function: system
         system( "CLS" );
   
   // print top row of table
      // functions: printThickSolidLine, printString, printChar, 
      //            printEndLines, printThinDividerLine
         printThickSolidLine();
         printString( FIRST_PIPE, NO_BLOCK, "LEFT" );
         printChar( SPACE );
         printString( "Area of :", COL_ONE, "RIGHT" );
         printString( shape_name, COL_TWO, "CENTER" );
         printChar( PIPE );
         printEndLines( ONE_LIN );
         printThinDividerLine();
         printThinDividerLine();
 
   // print subtitles
     // functions: printString, printChar, 
     //            printEndLines, printThinDividerLine
        printString( FIRST_PIPE, NO_BLOCK, "LEFT" );
        printString( "Measurement", COL_ONE, "CENTER" );
        printChar( PIPE );
        printString( "Value", COL_TWO, "CENTER" );
        printChar( PIPE );
        printEndLines( ONE_LIN );
        printThinDividerLine();

   // print inputted data row(s) and area 

      // protocol if "RECTANGLE" is entered
         // functions: if, printOneResultLine, printThinDividerLiane, printString, printChar, printEndLines
         if( shape_name == "RECTANGLE" )
           {
            printOneResultLine( "LENGTH", length);
            printOneResultLine( "WIDTH", width );
            printThinDividerLine();
            printString( FIRST_PIPE, NO_BLOCK, "LEFT" );
            printString( "Shape", COL_ONE, "CENTER" );
            printChar( PIPE );
            printString( "Area", COL_TWO, "CENTER" );
            printChar( PIPE );
            printEndLines( ONE_LIN );
            printThinDividerLine();
            printOneResultLine( "RECTANGLE", shape_area );
           }
    
      // protocol if "TRIANGLE" is entered
         // functions: if, printOneResultLine, printThinDividerLiane, printString, printChar, printEndLines
         if( shape_name == "TRIANGLE" )
           {
            printOneResultLine( "BOTTOM BASE", length);
            printOneResultLine( "HEIGHT", height );
            printThinDividerLine();
            printString( FIRST_PIPE, NO_BLOCK, "LEFT" );
            printString( "Shape", COL_ONE, "CENTER" );
            printChar( PIPE );
            printString( "Area", COL_TWO, "CENTER" );
            printChar( PIPE );
            printEndLines( ONE_LIN );
            printThinDividerLine();
            printOneResultLine( "TRIANGLE", shape_area );
           }

      // protocol if "TRAPEZOID" is entered
         // functions: if, printOneResultLine, printThinDividerLiane, printString, printChar, printEndLines
         if( shape_name == "TRAPEZOID" )
           {
            printOneResultLine( "BOTTOM BASE", length);
            printOneResultLine( "TOP BASE", width );
            printOneResultLine( "HEIGHT", height );
            printThinDividerLine();
            printString( FIRST_PIPE, NO_BLOCK, "LEFT" );
            printString( "Shape", COL_ONE, "CENTER" );
            printChar( PIPE );
            printString( "Area", COL_TWO, "CENTER" );
            printChar( PIPE );
            printEndLines( ONE_LIN );
            printThinDividerLine();
            printOneResultLine( "TRAPEZOID", shape_area );
           }

      // protocol if "PARALLELOGRAM" is entered
         // functions: if, printOneResultLine, printThinDividerLiane, printString, printChar, printEndLines
         if( shape_name == "PARALLELOGRAM" )
           {
            printOneResultLine( "BOTTOM BASE", length);
            printOneResultLine( "HEIGHT", height );
            printThinDividerLine();
            printString( FIRST_PIPE, NO_BLOCK, "LEFT" );
            printString( "Shape", COL_ONE, "CENTER" );
            printChar( PIPE );
            printString( "Area", COL_TWO, "CENTER" );
            printChar( PIPE );
            printEndLines( ONE_LIN );
            printThinDividerLine();
            printOneResultLine( "PARALLELOGRAM", shape_area );
           }

      // protocol if "OCTAGON" is entered
         // functions: if, printOneResultLine, printThinDividerLiane, printString, printChar, printEndLines
         if( shape_name == "OCTAGON" )
           {
            printOneResultLine( "BOTTOM BASE", length);
            printOneResultLine( "SPAN", width );
            printThinDividerLine();
            printString( FIRST_PIPE, NO_BLOCK, "LEFT" );
            printString( "Shape", COL_ONE, "CENTER" );
            printChar( PIPE );
            printString( "Area", COL_TWO, "CENTER" );
            printChar( PIPE );
            printEndLines( ONE_LIN );
            printThinDividerLine();
            printOneResultLine( "OCTAGON", shape_area );
           }

      // protocol if "CIRCLE" is entered
         // functions: if, printOneResultLine, printThinDividerLiane, printString, printChar, printEndLines
         if( shape_name == "CIRCLE" )
           {
            printOneResultLine( "RADIUS", height);
            printThinDividerLine();
            printString( FIRST_PIPE, NO_BLOCK, "LEFT" );
            printString( "Shape", COL_ONE, "CENTER" );
            printChar( PIPE );
            printString( "Area", COL_TWO, "CENTER" );
            printChar( PIPE );
            printEndLines( ONE_LIN );
            printThinDividerLine();
            printOneResultLine( "CIRCLE", shape_area );
           }

   // close table
     // function: printThickSolidLine, printEndLines 
        printThickSolidLine();
        printEndLines( ONE_LIN );

   // end function
      // no return - void
   }


void printOneResultLine( string shape_name, double shape_area ) 
   {
   // intitialize function/variables
      // none

   // print first text block with vertical borders
      // functions: printString, printChar
         printString( FIRST_PIPE, NO_BLOCK, "LEFT" );
         printString( shape_name, COL_ONE, "LEFT" );
         printChar( PIPE );

   // print second block
      // printDouble, printChar, printEndLines
         printDouble( shape_area, PREC, COL_TWO, "CENTER" );
         printChar( PIPE );
         printEndLines( ONE_LIN );

   // print row divider line
      // printThinDividerLine
         printThinDividerLine();

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



