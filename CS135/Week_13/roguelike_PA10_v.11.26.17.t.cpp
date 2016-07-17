// nick_3471@yahoo.com
// thenriod@gmail.com



////////////////////////////////////////////////////////////////////////////////
// Header Files ////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
   #include "formatted_console_io_v18.h"
   #include <fstream>
   using namespace std ;


////////////////////////////////////////////////////////////////////////////////
// Global Constant Definitions /////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// entire screen dimensions
const int uprLeftX  = 0;
const int uprLeftY  = 0;
const int lwrRightX = 0;
const int lwrRightY = 0;

// instruction screen dimensions


// main menu dimensions

// difficulty/name entry screen dimensions 

// game object constants
const char PLAYER    = '*';
const char CLOAK     = char(247);
const char HOLE      = char(178); 
const char TREASURE  = char(232);
const char SWORD     = char(157);
const char MONSTER   = char(206);
const char SPELL     = char(234);
const char INT_WALL  = char(176);
const char OUT_WALL  = char(176);
const char UP_STAIR  = 'U';
const char DWN_STAIR = 'D';

// typical character constants
const char NULL_CHAR = '\0';
const char SPACE     = ' ';
const char NEW_LINE  = '\n';

// game area constants
const int PLAY_AR_H    = 62;
const int PLAY_AR_W    = 32;
const int SCOREBOARD_W = 16;
const int SCOREBOARD_H = 25;
const int MESSAGE_W    = 13;
const int MESSAGE_H    = 13;

// probabilities
const int P_HOLE      = 35;
const int P_CLOAK     = 10; 
const int P_TREASURE  = 5;
const int P_SWORD     = 10;
const int P_MONSTER   = 25;
const int P_SPELL     = 10;

// c-style string lengths
const int NAME_LEN = 15;

// file names
const char FLOOR1[ NAME_LEN ] = "lowlevel.txt";
const char FLOOR2[ NAME_LEN ] = "midlevel.txt";
const char FLOOR3[ NAME_LEN ] = "highlevel.txt";
const char SCORES[ NAME_LEN ] = "scores.txt";

////////////////////////////////////////////////////////////////////////////////
// Global Function Prototypes //////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

    // none

/* 
Name: 
Process: 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
*/


////////////////////////////////////////////////////////////////////////////////
// Main Program Definition /////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int main()
   { 
// initialize variables
   
   // screen position variables
   int xLoc = 25, yLoc = 5;

   // arrays and matrices
   char playerName[ NAME_LEN ];
 
   // user input variables
   int userInput;

   // score related variables 
   int score = 0;

   // flags
   bool XCmode = false;
   bool keepGoing = true;

   // random seed
   srand( time( NULL ) );  


// initialize program

   // initialize curses
      // functions: startCurses, setColor
      startCurses();
      setColor( COLOR_WHITE, COLOR_BLACK, SET_BRIGHT );

// display splash screen
   // function: dispSplash
   userInput = dispSplash();

   // if 'E' was pressed, activate extra credit mode
      // operation: if
      if( userInput == 'E' || userInput == 'e' )
        {
        // set extra credit mode flag
        XCmode = true;
        }

// display instructions screen (the plan is to run the game in extra credit mode by default)
   // function: dispInstructions
   dispInstructions( XCmode );

// implement menu (loop until valid option is chosen)
   // operation: while
   while( keepGoing )

     // display menu
       // clear screen
          // function: setColor, clearScreen

          setColor( COLOR_WHITE, COLOR_BLACK, SET_BRIGHT);
          clearScreen( )

     // display options
        // function: printStringAt

     // prompt for user input
       // function: promptForCharAt

          // operation: switch

          // case: set difficulty
             // function: setDiff

          // case: play the game

             // get player name
                // function: promptForPlayer

             // play the game!
                // function: executeGame

          // case: show high scores
             // function: showHiScores

          // case: quit program  
             // set flag such that menu loop ends

// end the program
   
   // shut down curses
      // function: endCurses

   // return 0           
   return 0;
   }


////////////////////////////////////////////////////////////////////////////////
// Supporting function implementations /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

    // none



