// Header Files
#include "formatted_cmdline_io_v07.h"	// for command-line I/O

   using namespace std;

// Global Constant Definitions

   const char PIPE = '|';
   const int ONE_LINE = 1;
   const int TWO_LINES = 2;
   const int NO_BLOCK_SIZE = 0;
   const int TEST_NUM_COL_WIDTH = 8;
   const int WT_TYPE_COL_WIDTH = 11;
   const int DISTANCE_COL_WIDTH = 12;
   const int TIME_COL_SPEED = 8;
   const int SPEED_COL_WIDTH = 9;
   const int PRECISION = 2;
   string TABLE_TOP = "X~~~~~~|~~~~~~~~~~~|~~~~~~~~~~~~|~~~~~~~~|~~~~~~~~~X";
   string TABLE_BOTTOM = "X~~~~~~|~~~~~~~~~~~|~~~~~~~~~~~~|~~~~~~~~|~~~~~~~~~X";

// Global Function Prototypes

    // none

// Main Program Definition
int main()
   {
    // initialize program

       // initialize variables

       // show title
          // function: iostream <<
             cout << "Speed Calculator Program" << endl 
                  << "++++++++++++++++++++++++" << endl
                  << endl;
    // input Data

       // get data for first test

          // print first test header
             // function: iostream <<
                cout << "Enter data for 1st case:" << endl;

          // get test number for first test
             // function: iostream <<, >>
                int firstCase; // Introduce 1st case variable
                cout << "Enter case #: ";
                cin >> firstCase;

          // get weight type for first test
             // function: iostream <<, >>
                string weight1 ; // Introduce weight1 variable
                cout << "Enter weight type: ";
                cin >> weight1;

          // get distance for first test
             // function: iostream <<, >>
                double distance1 ; // Introduce distance1 variable
                cout << "Enter distance traveled: ";
                cin >> distance1;

          // get time for first test
             // function: iostream <<, >>
                double time1; // Introduce time1 variable
                cout << "Enter time elapsed: ";
                cin >> time1;
                   
printEndLines( ONE_LINE ); // Just separating the data entry prompts for aesthetic purposes

       // get data for second test

          // print second test header
             // function: iostream <<
                cout << "Enter data for 2nd case" << endl;

          // get test number for second test
             // function: promptForInt
                






                int secondCase; // Introduce 2nd case variable
                promptForint secondCase;










          // get weight type for second test
             // function: promptForString
                






                string weight2; // Introduce weight2 variable
                promptForString ;









                


          // get distance for second test
             // function: promptForDouble
                






                int secondCase; // Introduce 2nd case variable
                promptForint secondCase;










          // get time for second test
             // function: promptForDouble
                






                int secondCase; // Introduce 2nd case variable
                promptForint secondCase;










    // calculate speeds

       // calculate speed one
          // operation: math

       // calculate speed two
          // operation: math

    // output Data

       // make vertical space
         // function: printEndLines
printEndLines( TWO_LINES );
       // print top line
          // function: printString, printEndLines
            printString( TABLE_TOP, NO_BLOCK_SIZE, "LEFT" );
            printEndLines( ONE_LINE ); 
             
       // print main title with end pipes
          // function: printChar, printString, printEndLines

       // print sub titles with horizontal dividers and end pipes
          // function: printString, printEndLines

       // print first data set with divider pipes
          // function: printChar, printInt, printString, printDouble, printEndLines

       // print second data set with divider pipes
          // function: printChar, printInt, printString, printDouble, printEndLines

       // print bottom line
          // function: printString, printEndLines
            printString( TABLE_BOTTOM, NO_BLOCK_SIZE, "CENTER" );
            printEndLines( TWO_LINES );

    // end program

       // print vertical space
          // function: printEndLines

       // hold screen for user
          // function: system/pause
             system( "PAUSE" );
       // return zero
       return 0;
   }

// Supporting function implementations

    // none



