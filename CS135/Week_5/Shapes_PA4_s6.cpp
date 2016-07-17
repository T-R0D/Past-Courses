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
Name: displayResults
Process: accepts calculated areas and user input specifying the desired shape 
         area to be calculated and displays them in a table 
         containing the values.
Function Input/Parameters: string shape_desired   The desired shape whose area 
                                                  is to be calculated, as specified  
                                                  by the user.
                           double A_rect          The area calculated for a 
                                                  rectangle based on the input.
                           double A_tri           The area calculated for a 
                                                  triangle based on the input.
                           double A_trap          The area calculated for a 
                                                  trapezoid based on the input.
                           double A_para          The area calculated for a 
                                                  parallelogram based on the input.
                           double A_oct           The area calculated for an 
                                                  octagon based on the input.
                           double A_circ          The area calculated for a 
                                                  circle based on the input.
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: displays the "Results Table"
Dependencies: formatted command line I/O tools
*/
void displayResults( string shape_desired, double A_rect, double A_tri, double A_trap, double A_para, double A_oct, double A_circ) ;

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



// Main Program Definition
int main()
   { 
   
   // initialize program

      // initialize variables
         string shape_desired = "None entered!" ; 
         double length        = 8 ;
         double width         = 8 ;
         double height        = 8 ;
         double A_rect        = 8 ;
         double A_tri         = 8 ;
         double A_trap        = 8 ;
         double A_para        = 8 ;
         double A_oct         = 8 ;
         double oct_perimeter = 8 ;
         double apothem       = 8 ;
         double A_circ        = 8 ;
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

      // prompt user for desired chape type
         // functions: printString, promptForString
            printString( "     Enter shape name (e.g., TRIANGLE, CIRCLE) ", PROMPT_BLOCK, "LEFT" );
            shape_desired = promptForString( COLON );

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

   // perform computations

      /* Note: to make the math less complicated, I simply used 1 variable to
               correspond to each of the 3 prompts in the program. It should 
               be noted that:
               length represents the length or bottom base;
               width represents the width, span or top base;
               the height represents the height or radius.
               This means that the variable names may not match the proper 
               formula, but the code should result in the proper area 
               calculation.   */

      // compute area of rectangle
         // operation: math
            A_rect = (length * width) ;

      // compute area of triangle
         // operation: math
            A_tri = (.5 * length * height) ;
            // bottom base * height

      // compute area of trapezoid
         // operation: math
            A_trap = (.5 * ( length + width ) * height) ;
            // (.5 * (bottom base + top base) * height)

      // compute area of parallelogram
         // operation: math
            A_para = (length * height) ;
            // bottom base * height 
          
      // compute area of octagon 
         // operation: math
            oct_perimeter = (8 * length);
            apothem = (.5 * width );
            A_oct = (.5 * apothem * oct_perimeter );
            // length ==> bottom base, width ==> span            

      // compute area of circle
         // operation: math
            A_circ = (PI * pow( height, 2.0 )) ;
            // height ==> radius

   // display result

      // display results table 
         // function: displayResults   
            displayResults( shape_desired, A_rect, A_tri, A_trap, A_para, A_oct, A_circ);

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


void displayResults( string shape_desired, double A_rect, double A_tri, double A_trap, double A_para, double A_oct, double A_circ ) 
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
         printString( "Possible areas for :", COL_ONE, "LEFT" );
         printString( shape_desired, COL_TWO, "CENTER" );
         printChar( PIPE );
         printEndLines( ONE_LIN );
         printThinDividerLine();
         printThinDividerLine();
 
   // print subtitles
     // functions: printString, printChar, 
     //            printEndLines, printThinDividerLine
        printString( FIRST_PIPE, NO_BLOCK, "LEFT" );
        printString( "Shape Name", COL_ONE, "CENTER" );
        printChar( PIPE );
        printString( "Calculated Area", COL_TWO, "CENTER" );
        printChar( PIPE );
        printEndLines( ONE_LIN );
        printThickDividerLine();

   // print rectangle row
     // function: printOneResultLine 
        printOneResultLine( "Rectangle", A_rect );

   // print triangle row
     // function: printOneResultLine 
        printOneResultLine( "Triangle", A_tri );

   // print trapezoid row
     // function: printOneResultLine 
        printOneResultLine( "Trapezoid", A_trap );

   // print parallelogram row
     // function: printOneResultLine 
        printOneResultLine( "Parallelogram", A_para );

   // print octaogon row
     // function: printOneResultLine 
        printOneResultLine( "Octagon", A_oct );

   // print circle row
     // function: printOneResultLine 
        printOneResultLine( "Circle", A_circ );

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






