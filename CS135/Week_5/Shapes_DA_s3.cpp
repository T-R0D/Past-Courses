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

      // print program title
         // function: printTitle

   // display screen 1

      // display informational table that specifies the shapes the program can accomodate,
      // as well as the necessary data for each shape    
         // function: printRequiredData  

   // collect data

      // prompt user for desired chape type
         // functions: printString, promptForString

      // prompt user for length or bottom base data
         // functions: printString, promptForDouble

      // prompt user for width, span, or top base data
         // functions: printString, promptForDouble

      // prompt user for height or radius data  
         // functions: printString promptForDouble

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

      // compute area of triangle
         // operation: math

            // bottom base * height

      // compute area of trapezoid
         // operation: math
;
            // (.5 * (bottom base + top base) * height)

      // compute area of parallelogram
         // operation: math
;
            // bottom base * height 
          
      // compute area of octagon 
         // operation: math
;
            // length ==> bottom base, width ==> span            

      // compute area of circle
         // operation: math
            // height ==> radius

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


void displayResults( string shape_desired, double A_rect, double A_tri, double A_trap, double A_para, double A_oct, double A_circ ) 
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
      // no return - void
   }


void printThickSolidLine()
   {
      // no return - void
   }






