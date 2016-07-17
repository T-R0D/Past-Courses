// Header Files

   #include "formatted_console_io_v17.h" // for console I/O
    
   using namespace std;

// Global Constant Definitions

   const int PRECISION = 2;

// Function Prototypes

   // none

// Main Program
int main()
   {
    // initialize program

       // initialize variables

          // set initial x and y positions
          int xPos = 5, yPos = 5;

          // set other values
          int intVal = 54321;
          double dblVal = 1234.56;

       // initialize curses
          // function: startCurses
       startCurses();

       // set color for initial screen
          // function: setColor
       setColor( COLOR_RED, COLOR_BLUE, SET_BRIGHT );

       // clear screen
          // function: clearScreen
       clearScreen( SCRN_MIN_X, SCRN_MIN_Y, SCRN_MAX_X, SCRN_MAX_Y );

       // display string
       printStringAt( xPos, yPos, "This is a string", "LEFT" );

       // display integer value
       xPos += 10;
       yPos += 2;
       printIntAt( xPos, yPos, intVal, "LEFT" ); 

       // enter a double value
       xPos += 2;
       yPos += 2;
       dblVal = promptForDoubleAt( xPos, yPos, "Enter a floating point number: " );

       // display double value
       xPos += 2;
       yPos += 10;
       printDoubleAt( xPos, yPos, dblVal, PRECISION, "LEFT" ); 

    // end program

          // hold screen
             // function: waitForInput
          waitForInput( FIXED_WAIT );

       // shut down curses
          // function: endCurses
       endCurses();

       // return success to OS
       return 0;
   }

// Supporting Function Implementation

   // none


