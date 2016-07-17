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
const int UPRLEFTX  = 0;
const int UPRLEFTY  = 0;
const int ENT_SCR_W = 80;
const int ENT_SCR_H = 28;

// instruction screen dimensions/reference points
const int INSTR_UPX = 12;         ////////////////////////////////////////////////
const int INSTR_UPY = 2;
const int INSTR_W   = 53;         ////////////////////////////////////////////////
const int INSTR_H   = 20;

// main menu dimensions/reference points
const int MENU_UPX    = 0;
const int MENU_UPY    = 0;
const int MENU_W      = 30;      //////////////////////////////////////////////////
const int MENU_H      = 16;
const int MENU_CURSEX = ( MENU_UPX + 8 );       ///////////////////////////////////////////////////
const int MENU_CURSEY = ( MENU_UPY + 3 );       

// difficulty/name entry screen reference points 

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
const int PLAY_AR_W    = 32;              ///////////////////////////////
const int PLAY_AR_H    = 62;              ///////////////////////////////
const int SCOREBOARD_W = 16;
const int SCOREBOARD_H = 25;
const int SCORE_X      = 14;
const int MESSAGE_W    = 13;
const int MESSAGE_H    = 13;
const int GAMESTART_X  = 32;     // in gameMatrix coordinates                  
const int GAMESTART_Y  = 1 ;     // in gameMatrix coordinates                  

// probabilities
const int P_HOLE      = 35;
const int P_CLOAK     = 10; 
const int P_TREASURE  = 5 ;
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
   char playerName[NAME_LEN];
   char names[NAME_LEN];
   int scores[NAME_LEN];
 
   // user input variables
   int userInput;

   // game related variables 
   int score = 0;
   int difficulty = 1;

   // flags
   bool XCmode = false;
   bool keepRunning = true;

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

// read in high scores, if file is present, ensure arrays are clean 
   // function: readScores
   readScores( names, scores );

// implement menu (loop until valid option is chosen)
   // operation: while
   while( keepRunning )

     // display menu
       // clear screen
          // function: setColor, clearScreen
          setColor( COLOR_WHITE, COLOR_BLACK, SET_BRIGHT);
          clearScreen( UPRLEFTX, UPRLEFTY, (UPRLEFTX + ENT_SCR_W), (UPRLEFTY + ENT_SCR_H) );
          setColor( COLOR_WHITE, COLOR_BLUE, SET_BRIGHT);
          clearScreen( MENU_UPX, MENU_UPY, (MENU_UPX + MENU_W), (MENU_UPY + MENU_H) );                                                ////////////////////////////////////////////////////////////////

     // display menu text

        // set cursor position
        xLoc = MENU_CURSEX;
        yLoc = MENU_CURSEY;

        // menu title
           // function: printStringAt
           printStringAt( xLoc, yLoc, "MENU SELECTIONS", "LEFT" );
           xLoc = MENU_CURSEX;
           yLoc ++;
           printStringAt( xLoc, yLoc, "---------------", "LEFT" );           

        // option 1
           // function: printStringAt
           xLoc = MENU_CURSEX;
           yLoc += 2;
           printStringAt( xLoc, yLoc, "1. Set <D>ifficulty", "LEFT" );

        // option 2
           // function: printStringAt
           xLoc = MENU_CURSEX;
           yLoc ++;
           printStringAt( xLoc, yLoc, "2. <P>lay Game", "LEFT" );

        // option 3
           // function: printStringAt
           xLoc = MENU_CURSEX;
           yLoc ++;
           printStringAt( xLoc, yLoc, "3. <S>how Top Scores", "LEFT" );

        // option 4
           // function: printStringAt
           xLoc = MENU_CURSEX;
           yLoc ++;
           printStringAt( xLoc, yLoc, "4. <Q>uit program", "LEFT" );

        // prompt for user input
          // function: promptForCharAt
           xLoc = MENU_CURSEX;
           yLoc += 2;
           promptForCharAt( xLoc, yLoc, "Enter Selection: " );

          // operation: switch
          switch( userInput )
             
             // case: set difficulty
             case '1':
             case 'D':
             case 'd':
                {
                // function: setDiff
                difficulty = setDiff();
                }
                break;

             // case: play the game
             case '2':
             case 'P':
             case 'p':
                {
                // get player name
                   // function: promptForPlayer
                   promptForPlayer( playerName );

                // play the game!
                  // function: executeGame
                  executeGame( XCmode, score, playerName );

                // write high score file
                  // function: writeScores
                  writeScores( names, scores );
                }
                break;

             // case: show high scores
             case '3':
             case 'S':
             case 's':
                {
                // function: showHiScores
                showHiScores( names, scores );
                }
                break;

             // case: quit program  
             case '4':
             case 'Q':
             case 'q':
                {
                // set flag such that menu loop ends
                keepRunning = false;
                }
                break;

// end the program
   
   // shut down curses
      // function: endCurses
      endCurses();

   // return 0           
   return 0;
   }


////////////////////////////////////////////////////////////////////////////////
// Supporting function implementations /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

    // none



