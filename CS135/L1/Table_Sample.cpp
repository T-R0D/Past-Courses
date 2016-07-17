// Header Files
#include "formatted_cmdline_io_v07.h"	// for command-line I/O

   using namespace std;

// Global Constant Definitions

   const char PIPE = '|';
   const int ONE_LINE = 1;
   const int TWO_LINES = 2;
   const int NO_BLOCK_SIZE = 0;
   const int NAME_COL_WIDTH = 19;
   const int AGE_COL_WIDTH = 10;
   const int PAYRATE_COL_WIDTH = 9;
   const int PRECISION = 2;

// Global Function Prototypes

    // none

// Main Program Definition
int main()
   {
    // initialize program

       // initialize variables with data
       int age = 16;
       double payRate = 7.95;
       string name = "Bill Smith";

    // output data

       // make vertical space
         // function: printEndLines
       printEndLines( TWO_LINES );

       // print sub titles
          // function: printString, printEndLines
       printString( "|~~~~~~~~~~~~~~~~~~~~|----------|----------|", NO_BLOCK_SIZE, "LEFT" );
       printEndLines( ONE_LINE );
       printString( "|   NAME             |    AGE   | PAY RATE |", NO_BLOCK_SIZE, "LEFT" );
       printEndLines( ONE_LINE );
       printString( "|~~~~~~~~~~~~~~~~~~~~|----------|----------|", NO_BLOCK_SIZE, "LEFT" );
       printEndLines( ONE_LINE );

       // print data set
          // function: printChar, printInt, printString, printDouble, printEndLines
       printString( "| ", NO_BLOCK_SIZE, "LEFT" );
       printString( name, NAME_COL_WIDTH, "LEFT" );
       printChar( PIPE );
       printInt( age, AGE_COL_WIDTH, "CENTER" );
       printChar( PIPE );
       printDouble( payRate, PRECISION, PAYRATE_COL_WIDTH, "RIGHT" );
       printString( " |", NO_BLOCK_SIZE, "LEFT" );
       printEndLines( ONE_LINE );

       // print bottom line
          // function: printString, printEndLines
       printString( "|--------------------|----------|----------|", NO_BLOCK_SIZE, "LEFT" );
       printEndLines( ONE_LINE );

    // end program

       // print vertical space
          // function: printEndLines
       printEndLines( TWO_LINES );

       // hold screen for user
          // function: system/pause
       system( "PAUSE" );

       // return zero
       return 0;
   }

// Supporting function implementations

    // none



