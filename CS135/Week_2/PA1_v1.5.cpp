// Header Files
#include "formatted_cmdline_io_v07.h"	// for command-line I/O

   using namespace std;

// Global Constant Definitions

   const char PIPE = '|';
   const int ONE_LINE = 1;
   const int TWO_LINES = 2;
   const int NO_BLOCK_SIZE = 0;
   const int TEST_NUM_COL_WIDTH = 6;
   const int WT_TYPE_COL_WIDTH = 11;
   const int DISTANCE_COL_WIDTH = 12;
   const int TIME_COL_WIDTH = 8;
   const int SPEED_COL_WIDTH = 9;
   const int TABLE_WIDTH = 50;
   const int PRECISION = 2;
   const string EMPTY_BLOCK =   "                                                  ";
   const string TABLE_TOP =    "|------|-----------|------------|--------|---------|";
   const string TABLE_HORIZ =  "|------|-----------|------------|--------|---------|";
   const string TABLE_BOTTOM = "|------|-----------|------------|--------|---------|";

// Global Function Prototypes

    // none

// Main Program Definition
int main()
   {
    // initialize program

       // initialize variables
         // Introduce necessary variables --> variables are self-documenting
int firstCase;
string weight1; 
double distance1; 
double time1; 
double speed1; 
int secondCase;
string weight2;
double distance2;
double time2;
double speed2;

       // show title
          // function: iostream <<
             cout << "Speed Calculator Program" << endl 
                  << "========================" << endl
                  << endl;
    // input Data

       // get data for first test

          // print first test header
             // function: iostream <<
                cout << "Enter data for 1st case:" << endl;

          // get test number for first test
             // function: iostream <<, >>
                
                cout << "Enter case #: ";
                cin >> firstCase;

          // get weight type for first test
             // function: iostream <<, >>
                
                cout << "Enter weight type: ";
                cin >> weight1;

          // get distance for first test
             // function: iostream <<, >>
                
                cout << "Enter distance traveled: ";
                cin >> distance1;

          // get time for first test
             // function: iostream <<, >>
                
                cout << "Enter time elapsed: ";
                cin >> time1;
                   
printEndLines( ONE_LINE ); // Just separating the data entry prompts for aesthetic purposes

       // get data for second test

          // print second test header
             // function: iostream <<
                cout << "Enter data for 2nd case" << endl;

          // get test number for second test
             // function: promptForInt
                secondCase = promptForInt( "Enter case #: " );
                
          // get weight type for second test
             // function: promptForString
                weight2 = promptForString( "Enter weight type: " ); 

          // get distance for second test
             // function: promptForDouble
                distance2 = promptForDouble( "Enter distance traveled: " );

          // get time for second test
             // function: promptForDouble
                time2 = promptForDouble( "Enter time elapsed: " );  

    // calculate speeds

       // calculate speed one
          // operation: math
             speed1 = distance1 / time1 ; 
             
       // calculate speed two
          // operation: math
             speed2 = distance2 / time2 ; 

    // output Data

       // make vertical space
         // function: printEndLines
            printEndLines( TWO_LINES );
       // print top line
          // function: printString, printEndLines
             printString( TABLE_TOP, NO_BLOCK_SIZE, "CENTER" );
             printEndLines( ONE_LINE ); 
             
       // print main title with end pipes
          // function: printChar, printString, printEndLines

             /*printChar( PIPE );
             printString( EMPTY_BLOCK, NO_BLOCK_SIZE, "CENTER" );
             printChar( PIPE );
             printEndLines( ONE_LINE );  */

             printChar( PIPE );
             printString( "DATA  PRESENTATION", TABLE_WIDTH, "CENTER" );                              
             printChar( PIPE );
             printEndLines( ONE_LINE );

             /*printChar( PIPE );
             printString( EMPTY_BLOCK, NO_BLOCK_SIZE, "CENTER" );
             printChar( PIPE );            
             printEndLines( ONE_LINE );  */


       // print sub titles with horizontal dividers and end pipes
          // function: printString, printEndLines
             printString( TABLE_HORIZ, NO_BLOCK_SIZE, "LEFT" );
             printEndLines( ONE_LINE );
             printChar( PIPE );
             printString(  " CASE |  WT TYPE  |  DISTANCE  |  TIME  |  SPEED  ", NO_BLOCK_SIZE, "CENTER" );
             printChar( PIPE );
             printEndLines( ONE_LINE ); 
             printString( TABLE_HORIZ, NO_BLOCK_SIZE, "LEFT" );
             printEndLines( ONE_LINE );

       // print first data set with divider pipes
          // function: printChar, printInt, printString, printDouble, printEndLines
             printChar( PIPE );
             printInt( firstCase, TEST_NUM_COL_WIDTH, "CENTER" );
             printChar( PIPE );
             printString( weight1, WT_TYPE_COL_WIDTH, "CENTER" );
             printChar( PIPE );
             printDouble( distance1, PRECISION, DISTANCE_COL_WIDTH, "CENTER" );
             printChar( PIPE );
             printDouble( time1, PRECISION, TIME_COL_WIDTH, "CENTER" );
             printChar( PIPE );
             printDouble( speed1, PRECISION, SPEED_COL_WIDTH, "CENTER" );
             printChar( PIPE );
             printEndLines( ONE_LINE );


       // print second data set with divider pipes
          // function: printChar, printInt, printString, printDouble, printEndLines

             /*printString( TABLE_HORIZ, NO_BLOCK_SIZE, "LEFT" );
             printEndLines( ONE_LINE );  */

             printChar( PIPE );
             printInt( secondCase, TEST_NUM_COL_WIDTH, "CENTER" );
             printChar( PIPE );
             printString( weight2, WT_TYPE_COL_WIDTH, "CENTER" );
             printChar( PIPE );
             printDouble( distance2, PRECISION, DISTANCE_COL_WIDTH, "CENTER" );
             printChar( PIPE );
             printDouble( time2, PRECISION, TIME_COL_WIDTH, "CENTER" );
             printChar( PIPE );
             printDouble( speed2, PRECISION, SPEED_COL_WIDTH, "CENTER" );
             printChar( PIPE );
             printEndLines( ONE_LINE );

       // print bottom line
          // function: printString, printEndLines
            printString( TABLE_BOTTOM, NO_BLOCK_SIZE, "CENTER" );
            printEndLines( TWO_LINES );

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



