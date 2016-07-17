
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
const int MENU_UPX    = 15;      //////////////////////////////////////////////////
const int MENU_UPY    = 5;       //////////////////////////////////////////////////
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
const int PLAY_AR_W    = 23;              ///////////////////////////////
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

/*
  Name: dispSplash
  Process: Displays the splash screen, holds until button is pressed
  Function Input/Parameters: none
  Function Output/Parameters:None 
  Function Output/Returned: None
  Device Input: none
  Device Output/Monitor: Displays Title, and developer's names
  Dependencies: formatted_console_io_v18
  Developer: Nicholas Smith
*/
char dispSplash ();


/* 
Name: 
Process: 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer:
*/
void printASCIIart( char letter, int &currX, int &currY);


/* 
Name: 
Process: 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer:
*/
void dispInstructions( bool xMode );


/*
  Name: readHiScores
  Process: If file excits it reads the file in to be used later
  Function Input/Parameters: none
  Function Output/Parameters: none
  Function Output/Returned: none
  Device Input: none
  Device Output: none
  Dependencies: none
  Developer: Nicholas Smith
*/
void readHiScores ( char names[ NAME_LEN ], int scores[] );


/* 
Name: 
Process: 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer:
*/
void printMenuText();


/* 
Name: 
Process: 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer:
*/
int setDiff();


/* 
Name: 
Process: 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer:
*/
void promptForPlayer( char name[ NAME_LEN ] );


/* 
Name: 
Process: 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer:
*/
int executeGame( bool xcMode, int &score, char name[] );


/*
  Name: isFileThere
  Process: Checks to see if level file is there
  Function Input/Parameters: file name, ifstream object
  Function Output/Parameters: none
  Function Output/Returned: Bool, whethier or not file is there
  Device Input: none
  Device Output: none
  Dependencies: none
  Developer: Nicholas Smith
*/
bool isFileThere ( const string &fileName );\


/* 
Name: 
Process: 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer:
*/
void fileMissingMessage();


/* 
Name: 
Process: 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer:
*/
void makeScoreBoard( bool xcMode );


/*
  Name: makeOutWal
  Process: Creates the outside wall of the game
  Function Input/Parameters: Width of the wall, heigh of the wal
  Function Output/Parameters: none
  Function Output/Returned: non
  Device Input: none
  Device Output: The wall around the board
  Dependencies: none  
  Developer: Nicholas Smith 
*/ 
void makeOutWall( int width, int height );


/* 
Name: 
Process: 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer:
*/
void clearMatrix( short matrix[PLAY_AR_W][PLAY_AR_H] );


/* 
Name: 
Process: 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer:
*/
void setFloor( int floorNumber, short playMatrix[PLAY_AR_W][PLAY_AR_H] );



/* 
Name: 
Process: 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer:
*/
int readFloorData( char fileName[], int Xcoordinates[], int Ycoordinates[] );


/* 
Name: 
Process: 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer:
*/
void genObjectAt( int Xcoord, int Ycoord, short object, short playMatrix[PLAY_AR_W][PLAY_AR_H] );


/* 
Name: 
Process: 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer:
*/
void matrixToScreen(int matrixX, int matrixY, short playMatrix[PLAY_AR_W][PLAY_AR_H] );


/* 
Name: 
Process: 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer:
*/
void keepScore( bool xcMode, int cloaks, int swords, int spells, int treasure, int &score );


/*
  Name: movePlayer
  Process: Decided were to move
  Function Input/Parameters: time it takes to run loop
  Function Output/Parameters: none
  Function Output/Returned: none
  Device Input: none
  Device Output: none
  Dependencies: function: move
  Developer: Nicholas Smith
*/
void movePlayer ( int waitTime, bool &continueProgram );


/* 
Name: 
Process: 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer:
*/
bool collisionTest( int &cloakTally,  int &swordTally,  int &spellTally, int &treasureTally,
                    int intendedX, int intendedY, short playMatrix[PLAY_AR_W][PLAY_AR_H], bool &continueGame );


/*
  Name: move
  Process: Moves the player inside the game area
  Function Input/Parameters: initial x & y positions, and the way the peice is moving
  Function Output/Parameters: none
  Function Output/Returned: none
  Device Input: none
  Device Output: none
  Dependencies: none
  Developer: Nicholas Smith
*/
bool move( int &xPos, int &yPos, int xVect, int yVect );


/* 
Name: 
Process: 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer:
*/
void screenToMatrix( int screenX, int screenY, char object, char playMatrix[PLAY_AR_W][PLAY_AR_H] );


/* 
Name: 
Process: 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer:
*/
void genObjects( bool xcMode, short gameBoard[ PLAY_AR_W ][ PLAY_AR_H ] );


/*
  Name: XYPos
  Process: Generate x and y position to be used for item placement
  Function Input/Parameters: x and y positions as a reference
  Function Output/Parameters: x y postions
  Function Output/Returned: none
  Device Input: none
  Device Output: none
  Dependencies: spaceOccupied, rand()
  Developer: Nicholas Smith
*/
void XYPos ( int &xPos, int &yPos );


/*
  Name: spaceOccupied
  Process: takes a x & y corridnate and runs them through the matrix 
	   checking to see if that spot is occupied
  Function Input/Parameters: x & y position, and matrix
  Function Output/Parameters: none
  Function Output/Returned:Bool, true if nothing is there, false it there is something in that spot
  Device Input: none
  Device Output: none
  Dependencies: none
  Developer: Nicholas Smith
*/
bool spaceOccupied ( int XPos, int YPos, short playMatrix[PLAY_AR_W][PLAY_AR_H]);


/* 
Name: 
Process: 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer:
*/
bool doGenObject( int chancein100 );


/* 
Name: 
Process: 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer:
*/
int get_Random_Number(int low_Value, int high_Value);


/* 
Name: 
Process: 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer:
*/
void gameOverMessage();


/*
  Name: writeScoreFile
  Process: writes scores into file after game has been played
  Function Input/Parameters: name of players and scores (array)
  Function Output/Parameters: none
  Function Output/Returned: none
  Device Input: none
  Device Output: none
  Dependencies: sortScores
  Developer: Nicholas Smith
*/
void writeScoreFile( char names[ NAME_LEN ], int scores[], int candidate );


/*
  Name: sortScores
  Process: sorts the scores from highest to lowest, only has ten
  Function Input/Parameters: name of players, scores for players, new score to be sorted
  Function Output/Parameters: none
  Function Output/Returned:none
  Device Input: none
  Device Output: none
  Dependencies: noen
  Developer: Nicholas Smith
*/
void sortScores( char names[ NAME_LEN], int scores[], int candidate);


/*
  Name: showHiScores
  Process: When selected, the scores will be shone from the file
  Function Input/Parameters: 
  Function Output/Parameters: none
  Function Output/Returned: 
  Device Input: none
  Device Output: Top ten hi scores in order
  Dependencies: readScores, sortScores, writeScoreFile
  Developer: Nicholas Smith
*/
void showHiScores ( char names[], int scores[] );


////////////////////////////////////////////////////////////////////////////////
// Main Program Definition /////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int main()
   { 
// initialize variables

   // arrays and matrices
   char playerName[NAME_LEN];
   char names[NAME_LEN];
   int scores[NAME_LEN];
 
   // user input variables
   int userInput;

   // game related variables 
   int score = 0;
   int difficulty = 5;     // easy by default

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
   readHiScores( names, scores );

// implement menu (loop until valid option is chosen)
   // operation: while
   while( keepRunning )

     // display menu
        // clear screen
           // function: setColor, clearScreen
           setColor( COLOR_WHITE, COLOR_BLACK, SET_BRIGHT);
           clearScreen( UPRLEFTX, UPRLEFTY, (UPRLEFTX + ENT_SCR_W), (UPRLEFTY + ENT_SCR_H) );
           setColor( COLOR_WHITE, COLOR_BLUE, SET_BRIGHT);
           clearScreen( MENU_UPX, MENU_UPY, (MENU_UPX + MENU_W), (MENU_UPY + MENU_H) );         

     // display menu text
        // function: printMenuText
        printMenuText();

          // operation: switch
          switch( userInput )
             {
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
                  writeScoreFile( names, scores, score );
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
             }
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

char dispSplash ()
    {
    // Initialize Variables
       // Set initial x & y positions

       // Set other Variables


    // Set color for Title and Background
       // Funtion: setColor


    // Clear screen
       // Function: clearScreen


    // Display Title
       // printStringAt || printCharAt

    // Display Developer's Names
       // Change x & y positions

    // Hold Screen For User
       // Change x & y postions

       // print string to tell user what to do
	  // printStringAt
       
       // Hold screen by prompting for char
          // promptForCharAt

    // return input
    return 0;     // temporary stub return
    }


void printASCIIart( char letter, int &currX, int &currY)
   {
   // variables
     

   // decide which ASCII art character to display 
    switch( letter )
      {
      // print art E
       case 'E':
       case 'e':
            {
            // first line of letter
               // function:printStringAt
               printStringAt( currX, currY, " EEEEEEEE ", "LEFT" );
             
            // second line of letter
               // function:printStringAt
               currX -= 10;
               currY ++;
               printStringAt( currX, currY, " EE     E ", "LEFT" );

            // third line of letter
               // function:printStringAt
               currX -= 10;
               currY ++;
               printStringAt( currX, currY, " EEEE     ", "LEFT" );

            // fourth line of letter
               // function:printStringAt
               currX -= 10;
               currY ++;
               printStringAt( currX, currY, " EEEE     ", "LEFT" );

            // fifth line of letter
               // function:printStringAt
               currX -= 10;
               currY ++;
               printStringAt( currX, currY, " EE     E ", "LEFT" );

            // sixth line of letter
               // function:printStringAt
               currX -= 10;
               currY ++;
               printStringAt( currX, currY, " EEEEEEEE ", "LEFT" );

            // position cursor for next letter
               currY -= 5;
            }
            break;

      // print art G
       case 'G':
       case 'g':
            {
            // first line of letter
               // function:printStringAt
               printStringAt( currX, currY, "  GGGGGGG  ", "LEFT" );             
             
            // second line of letter
               // function:printStringAt
               currX -= 11;
               currY ++;
               printStringAt( currX, currY, " GGG    GG ", "LEFT" );

            // third line of letter
               // function:printStringAt
               currX -= 11;
               currY ++;
               printStringAt( currX, currY, " GG        ", "LEFT" );

            // fourth line of letter
               // function:printStringAt
               currX -= 11;
               currY ++;
               printStringAt( currX, currY, " GG   GGGG ", "LEFT" );

            // fifth line of letter
               // function:printStringAt
               currX -=11;
               currY ++;
               printStringAt( currX, currY, " GGG    GG ", "LEFT" );

            // sixth line of letter
               // function:printStringAt
               currX -= 11;
               currY ++;
               printStringAt( currX, currY, "  GGGGGGG  ", "LEFT" );

            // position cursor for next letter
               currY -= 5;
            }
            break;

      // print art I
       case 'I':
       case 'i':
            {
            // first line of letter
               // function:printStringAt
               printStringAt( currX, currY, " IIII ", "LEFT" ); 
          
             // second line of letter
               // function:printStringAt
               currX -= 6;
               currY ++;
               printStringAt( currX, currY, "  II  ", "LEFT" );

            // third line of letter
               // function:printStringAt
               currX -= 6;
               currY ++;
               printStringAt( currX, currY, "  II  ", "LEFT" );

            // fourth line of letter
               // function:printStringAt
               currX -= 6;
               currY ++;
               printStringAt( currX, currY, "  II  ", "LEFT" );

            // fifth line of letter
               // function:printStringAt
               currX -= 6;
               currY ++;
               printStringAt( currX, currY, "  II  ", "LEFT" );

            // sixth line of letter
               // function:printStringAt
               currX -= 6;
               currY ++;
               printStringAt( currX, currY, " IIII ", "LEFT" );

            // position cursor for next letter
               currY -= 5;
            }
            break;

      // print art K
       case 'K':
       case 'k':
            {
            // first line of letter
               // function:printStringAt
               printStringAt( currX, currY, " KK  KK  ", "LEFT" );
             
            // second line of letter
               // function:printStringAt
               currX -= 9;
               currY ++;
               printStringAt( currX, currY, " KK KK   ", "LEFT" );

            // third line of letter
               // function:printStringAt
               currX -= 9;
               currY ++;
               printStringAt( currX, currY, " KK K    ", "LEFT" );

            // fourth line of letter
               // function:printStringAt
               currX -= 9;
               currY ++;
               printStringAt( currX, currY, " KKKKK   ", "LEFT" );

            // fifth line of letter
               // function:printStringAt
               currX -= 9;
               currY ++;
               printStringAt( currX, currY, " KK  KK  ", "LEFT" );

            // sixth line of letter
               // function:printStringAt
               currX -= 9;
               currY ++;
               printStringAt( currX, currY, " KK   KK ", "LEFT" );

            // position cursor for next letter
               currY -= 5;
            }
            break;

      // print art L
       case 'L':
       case 'l':
            {
            // first line of letter
               // function:printStringAt
               printStringAt( currX, currY, " LL      ", "LEFT" );
             
            // second line of letter
               // function:printStringAt
               currX -= 9;
               currY ++;
               printStringAt( currX, currY, " LL      ", "LEFT" );

            // third line of letter
               // function:printStringAt
               currX -= 9;
               currY ++;
               printStringAt( currX, currY, " LL      ", "LEFT" );

            // fourth line of letter
               // function:printStringAt
               currX -= 9;
               currY ++;
               printStringAt( currX, currY, " LL      ", "LEFT" );

            // fifth line of letter
               // function:printStringAt
               currX -= 9;
               currY ++;
               printStringAt( currX, currY, " LL      ", "LEFT" );

            // sixth line of letter
               // function:printStringAt
               currX -= 9;
               currY ++;
               printStringAt( currX, currY, " LLLLLLL ", "LEFT" );

            // position cursor for next letter
               currY -= 5;
            }
            break;

      // print art O
       case 'O':
       case 'o':
            {
            // first line of letter
               // function:printStringAt
               printStringAt( currX, currY, "  OOOOOO  ", "LEFT" );
            
            // second line of letter
               // function:printStringAt
               currX -= 10;
               currY ++;
               printStringAt( currX, currY, " OOO  OOO ", "LEFT" );

            // third line of letter
               // function:printStringAt
               currX -= 10;
               currY ++;
               printStringAt( currX, currY, " OO    OO ", "LEFT" );

            // fourth line of letter
               // function:printStringAt
               currX -= 10;
               currY ++;
               printStringAt( currX, currY, " OO    OO ", "LEFT" );

            // fifth line of letter
               // function:printStringAt
               currX -= 10;
               currY ++;
               printStringAt( currX, currY, " OOO  OOO ", "LEFT" );

            // sixth line of letter
               // function:printStringAt
               currX -= 10;
               currY ++;
               printStringAt( currX, currY, "  OOOOOO  ", "LEFT" );

            // position cursor for next letter
               currY -= 5;
            }
            break;

      // print art R
       case 'R':
       case 'r':
            {
            // first line of letter
               // function:printStringAt
               printStringAt( currX, currY, " RRRRRR  ", "LEFT" );
             
            // second line of letter
               // function:printStringAt
               currX -= 9;
               currY ++;
               printStringAt( currX, currY, " RR   RR ", "LEFT" );

            // third line of letter
               // function:printStringAt
               currX -= 9;
               currY ++;
               printStringAt( currX, currY, " RR   RR ", "LEFT" );

            // fourth line of letter
               // function:printStringAt
               currX -= 9;
               currY ++;
               printStringAt( currX, currY, " RRRRRR  ", "LEFT" );

            // fifth line of letter
               // function:printStringAt
               currX -= 9;
               currY ++;
               printStringAt( currX, currY, " RR RR   ", "LEFT" );

            // sixth line of letter
               // function:printStringAt
               currX -= 9;
               currY ++;
               printStringAt( currX, currY, " RR  RR  ", "LEFT" );

            // position cursor for next letter
               currY -= 5;
            }
            break;

      // print art U
       case 'U':
       case 'u':
            {
            // first line of letter
               // function:printStringAt
               printStringAt( currX, currY, " UU    UU ", "LEFT" ); 
            
            // second line of letter
               // function:printStringAt
               currX -= 10;
               currY ++;
               printStringAt( currX, currY, " UU    UU ", "LEFT" );

            // third line of letter
               // function:printStringAt
               currX -= 10;
               currY ++;
               printStringAt( currX, currY, " UU    UU ", "LEFT" );

            // fourth line of letter
               // function:printStringAt
               currX -= 10;
               currY ++;
               printStringAt( currX, currY, " UU    UU ", "LEFT" );

            // fifth line of letter
               // function:printStringAt
               currX -= 10;
               currY ++;
               printStringAt( currX, currY, " UUU  UUU ", "LEFT" );

            // sixth line of letter
               // function:printStringAt
               currX -= 10;
               currY ++;
               printStringAt( currX, currY, "  UUUUUU  ", "LEFT" );

            // position cursor for next letter
               currY -= 5;
            }
            break;

      // move to a new line
       case NEW_LINE:
            {
            // increment the xy and y cursor positions accordingly 
               currX =   5;
               currY -= 40;
            }
            break;
      }

   // no return - void
   }


void dispInstructions( bool xMode )
   {
   // variables

   // paint gray background
      // function: clearScreen
      
      // paint a bigger background if extra credit mode is enabled
         // operation: if

           //function: clearScreen
 
   // write title line w/ underline
      // functions: printStringAt

   // write cloak/hole info
      // functions: printStringAt, setColor, printSpecialCharAt

   // if game is in extra credit mode, display extra credit info
      // operation: if

        // write sword/monster info
           // functions: printStringAt, setColor, printSpecialCharAt

        // write spell/wall info
           // functions: printStringAt, setColor, printSpecialCharAt

   // write treasure info
      // functions: printStringAt, setColor, printSpecialCharAt

   // write stair info
      // functions: printStringAt, setColor, printCharAt

   // write condition(s) for ending the game
      // functions: printStringAt

   // write object of game message
      // functions: printStringAt

   // hold the system for the user
      // function: waitForInput

   }


void readHiScores ( char names[ NAME_LEN ], int scores[] )
   {
    // Initalize Variables
    
    // Loop to read in files, go until index < 11 || fin !.good
   
       // read in name into array
      
       // read in score into array

       // update index 

   }


void printMenuText()
   {
   // set cursor position
        int xLoc = MENU_CURSEX;
        int yLoc = MENU_CURSEY;

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

   // no return - void
   }


int setDiff()
   {
   // variables

   // clear screen to black, then a small area to blue
      // functions: setColor, clearScreen

   // display "screen title"
      // function: printStringAt

   // display prompt
      // function: printStringAt

   // loop until proper entry is received
      // operation: while

         // prompt user for dificulty selection
           // function: promptForIntAt

   // convert user selection to meaningful program terms 
      // operation: switch
        
        // case: easy ( 1 -> 5 )

        // case: medium ( 2 -> 3 )

        // case: hard ( 3 -> 1 )
 

   // return difficulty in meaningful program terms
   return 0;     // temporary return stub
   }


void promptForPlayer( char name[ NAME_LEN ] )
   {
   // variables

   // clear screen to black, then a smaller box to blue
      // functions: setColor, clearScreen

   // write prompt text for player name
      // function: printStringAt

   // prompt for user name, storing the name in a C-style string
      // function: getLineAt (the void one)

   // no return - void
   }


int executeGame( bool xcMode, int &score, char name[] )
   {
   // variables
   
      // cursor positions
          
      // game board matrix/floor coordinates
 
      // current floor and counters

      // floor flag

   // check for floor files
      // function: isFileThere

      // display error message and return user to menu if any files are missing
         // operation: if 

            // display the error message
               // function: fileMissingMessage

            // end gameplay by returning 0 

   // initialize score board
      // function: makeScoreBoard
  
   // place outer wall
      // function: makeOutWall 

   // initialize game board
      // function: clearMatrix

   // begin game loop
      // operation: while
      
        // set/reset floor if necessary
           // operation: if

              // function: setFloor
                                                // also resets player position

        // manage the scoreboard
           // function: keepScore              

        // game object generation
           // function: genObjects

        // manage player movement 
           // function: movePlayer

        // loop breaks when condition to continue game becomes false

   // display game over message
      // function: gameOverMessage

   // return zero
   return 0;          
   }


bool isFileThere ( const string &fileName )
     {
      // Initalize variables
      ifstream fin;	
      fin.clear();
      fin.open( fileName.c_str() );
      if ( !fin.good() )
         {
         printStringAt ( UPRLEFTX, UPRLEFTY, "<<You have fallen into a pit of spikes. YOU ARE DEAD>>", "LEFT" );
	 waitForInput(FIXED_WAIT);
         return false;
         }

       // close file if it was there 
          // function: .close

       // return true to indicate file was there
       return 0;     // temporary stub function
      }

void fileMissingMessage()
   {
   // variables

   // screen does not need to be cleared, this
   // was previously done by the name capturing function

   // print the error message on the screen
 
      // title line

      // main message

   // move the cursor to the position where system hold message will be displayed
  
   // hold the screen for the user

   // no return - void
   }


void makeScoreBoard( bool xcMode )
   {
   //variables

   // paint scoreboard background
      // functions: setColor, clearScreen

   // create tool item tallies

      // cloak tally
         // functions: printStringAt, printIntAt

      // extra credit item tallies if extra credit mode is active
         // operation: if 
         
            // sword tally
               // functions: printStringAt, printIntAt          
   
            // spell tally
               // functions: printStringAt, printIntAt

   // create treasure tally
      // functions: printStringAt, printIntAt

   // create score tally
      // functions: printStringAt, printIntAt

   // write game directions
   
      // movement directions
         // function: printStringAt

      // stair instructions
         // function: printStringAt

      // how the game ends
         // function: printStringAt

   // no return - void   
   }


void makeOutWall( int width, int height )
   {
    // initialize function/variables
    int counter;
    int leftUpperX = SCRN_MAX_X / 2 - width / 2;
    int leftUpperY = SCRN_MAX_Y / 2 - height / 2;
    int rightLowerX = leftUpperX + width - 1;
    int rightLowerY = leftUpperY + height - 1;

    // clear screen in shape of box
    clearScreen( leftUpperX, leftUpperY, rightLowerX, rightLowerY );

    // iterate from left to right across box
    for( counter = leftUpperX; counter < leftUpperX + width; counter++ )
       {
        // print top frame character
        printSpecialCharAt( counter, leftUpperY, OUT_WALL );

        // print bottom frame character
        printSpecialCharAt( counter, rightLowerY, OUT_WALL );
       }
    
    for( counter = leftUpperY; counter < leftUpperY + height; counter++ )
       {
        // print left frame character
        printSpecialCharAt( leftUpperX, counter, OUT_WALL );

        // print right frame character
        printSpecialCharAt( rightLowerX, counter, OUT_WALL );
       }
    }


void clearMatrix( short matrix[PLAY_AR_W][PLAY_AR_H] )
   {
   // variables

   // set the background color
      // function: setColor

   // use loop to fill all positions in matrix with spaces
   // pick a row and fill it, then move on
      // operation: for
 
         // iterate through the row
            // operation: for

               // store a space to the current matrix position

               // loop ends when every position in row is filled

         // loop ends when all columns have been visited 

   // no return - void
   }


void setFloor( int floorNumber, short playMatrix[PLAY_AR_W][PLAY_AR_H] )
   {
   // variables

   // clear the play matrix
      // function: clearMatrix

   // load coordinate arrays depending on which floor is selected
      // operation: switch

         // case: floor 1
      
                // function: readFloorData
    
         // case: floor 2
  
                // function: readFloorData

         // case: floor 3

                // function: readFloorData

   // load walls into game matrix by looping through coordinates until the walls have been "used up"
      // operation: for

   // reload player into matrix

   // reload stairs, depending on level
      // operation: switch

         // case: setting floor 1 (only up stairs)
 
                // function: genObjectAt

         // case: setting floor 2 (both stairs)

                // function: genObjectAt

         // case: setting floor 3 (only down stairs)

                // function: genObjectAt

   // output walls to display, again by looping
      // operation: for

         // use coordinates stored in X and Y arrays to output walls stored in matrix to display
            // setColor, matrixToScreen  
   
   // output player to screen
      // functions: setColor, matrixToScreen

   // output stairs to screen
      // operation: switch

         // case: setting floor 1 (only up stairs)
 
                // function: setColor, matrixToScreen

         // case: setting floor 2 (both stairs)

                // function: setColor, matrixToScreen

         // case: setting floor 3 (only down stairs)

                // function: setColor, matrixToScreen

   // no return - void
   } 


int readFloorData( char fileName[], int Xcoordinates[], int Ycoordinates[] )
   {
   // variables

   // clear fstream object, open the floor file

   // use loop to read the floor pattern data into the coordinate arrays
      // operation: for

         // read an x coordinate, then a y coordinate

   // when the loop breaks, the index will indicate the number of walls read in, store it

   // close the file

   // return the number of walls
   return 0;     // temporary stub function
   }


void genObjectAt( int Xcoord, int Ycoord, short object, short playMatrix[PLAY_AR_W][PLAY_AR_H] )
   {
   // place object in matrix

   // animate object on screen
      // function: matrixToScreen

   // no return - void
   } 


void matrixToScreen(int matrixX, int matrixY, short playMatrix[PLAY_AR_W][PLAY_AR_H] )
   {
   // variables

   // convert matrix x coordinate to screen x coordinate

   // convert matrix y coordinate to screen y coordinate

   // copy the object from the matrix

   // print the object in the matrix to the screen
      // function: printSpecialCharAt

   // no return - void
   }


void keepScore( bool xcMode, int cloaks, int swords, int spells, int treasure, int &score )
   {
   // update item tallies

      // cloak tally
         // function: printIntAt

      // extra credit items if extra credit mode is active
         // operation: if

            // sword tally
               // function: printIntAt 

            // spell tally
               // function: printIntAt
 
      // treasure tally
         // function: printIntAt

   // re-calculate score, if extra credit mode is inactive, extra credit item tallies will be zero
      // operation: math

   // update score
      // function: printIntAt

   // no return - void 
   }


void movePlayer ( int waitTime, bool &continueProgram )
    { 
    // Initialize variables
       int xLoc, yLoc, xVector, yVector;
      

       // initialize locations and movement vectors
       xLoc = PLAY_AR_W / 2;
       yLoc = PLAY_AR_H / 2;
       xVector = 0; yVector = 0;
    do
       {
        // if user input, set direction, otherwise ignore
        switch( waitTime )
           {
            case KB_RIGHT_ARROW:
               xVector = 1;
               yVector = 0;
               break;

            case KB_LEFT_ARROW:
               xVector = -1;
               yVector = 0;
               break;
 
            // set up a case for moving up with the up arrow key
        
            case KB_UP_ARROW:
               xVector = 0;
               yVector = -1;
               break;    

            // set up a case for moving down with the down arrow key
            case KB_DOWN_ARROW:
               xVector = 0;
               yVector = 1;
               break;

            // set up a case for moving down with the down arrow key
            case KB_ESCAPE:
               continueProgram = false;
               break;
            }

        // decide if player can move into next spot
           // operation: if

              // if collisionTest allows, move the character
              move( xLoc, yLoc, xVector, yVector );

              // otherwise, player is not allowed to move
       }
        while( continueProgram );

// no return - void
}


bool collisionTest( int &cloakTally,  int &swordTally,  int &spellTally, int &treasureTally,
                    int intendedX, int intendedY, short playMatrix[PLAY_AR_W][PLAY_AR_H], bool &continueGame )
   {
   // test new player position, if space, allow player to move
      // operation: if; function: spaceOccupied 

   // else, program responds accordingly
      // operations: else, switch

          // case: Hole
             
             // test to see if requisite tool is present
                // operation: if, else

                   // if not enough of tool, return signal to stop movement and end game
                

                // otherwise, decrement item tally

          // case: Monster
             
             // test to see if requisite tool is present
                // operation: if, else

                   // if not enough of tool, return signal to stop movement and end game
                

                // otherwise, decrement item tally       

          // case: Wall
             
             // test to see if requisite tool is present
                // operation: if, else

                   // if not enough of tool, return signal to stop movement
                

                // otherwise, decrement item tally

          // case: cloak
             
             // set flag to allow movement, increment item tally

          // case: sword
             
             // set flag to allow movement, increment item tally

          // case: spell
             
             // set flag to allow movement, increment item tally

          // case: treasure
             
             // set flag to allow movement, increment item tally

   // return movement decision if it hasn't been already
   return 0;     // temporary stub return 
   }


bool move( int &xPos, int &yPos, int xVect, int yVect )
   {
    // initialize function/variables
    bool moved = false;
    int oldX = xPos, oldY = yPos;

    // if x vector is live, test for acceptable movement limits
    if( ( xVect != 0 ) // is meant to move
          && ( xPos + xVect >= SCRN_MIN_X ) // won't go off left side of screen 
               && ( xPos + xVect <= SCRN_MAX_X )  ) // won't go off right side of screen
       {
        // cover up the old marker
	setColor( COLOR_BLACK, COLOR_WHITE, SET_BRIGHT );
        printCharAt( oldX, oldY, SPACE );

        // reset the x position
        xPos = xPos + xVect;

        // set moved flag to indicate success
        moved = true;
       }

    // if y vector is live, test for acceptable movement limits
    else if( ( yVect != 0 ) // is meant to move
               && ( yPos + yVect >= SCRN_MIN_Y ) // won't go off top of screen
                    && ( yPos + yVect <= SCRN_MAX_Y ) ) // won't go off bottom of screen
       {
        // cover up the old marker
	setColor( COLOR_BLACK, COLOR_WHITE, SET_BRIGHT );
        printCharAt( oldX, oldY, SPACE );

        // reset the y position
        yPos = yPos + yVect;

        // set the moved flag to indicate success
        moved = true;
       }

    // print the marker at the specified location
    setColor( COLOR_WHITE, COLOR_BLUE, SET_BRIGHT );
    printCharAt( xPos, yPos, PLAYER );

    // return successful move
    return moved;
   }


void screenToMatrix( int screenX, int screenY, char object, char playMatrix[PLAY_AR_W][PLAY_AR_H] )
   {
   // variables

   // convert screen x coordinate to matrix x coordinate

   // convert screen y coordinate to matrix y coordinate

   // store the object in the matrix

   // no return - void
   }


void genObjects( bool xcMode, short gameBoard[ PLAY_AR_W ][ PLAY_AR_H ] )
   {
   // hole generation

   // cloak generation

   // treasure generation

      // extra credit items if extra credit mode is enabled
         // operation: if
              
         // sword generation

         // monster generation

         // spell generation

   // no return - void
   }


void XYPos ( int &xPos, int &yPos )
   {
    // Initalize variables

    // Loop to run until variable = false


    // generate X variable
       // random number between 0 and width of board
	  // Function: rand()

    // generate y variable
       // random number between 0 and height of board
	  // Function: rand()

    // Check if occupied
       // Function: spaceOccupied

    // End loop
       // if spaceOccupied = ture, varibael = false
   }


bool spaceOccupied ( int XPos, int YPos, short playMatrix[PLAY_AR_W][PLAY_AR_H])
   {
    // Initalize Variables

    // takes XPos and runs it through playMatrix[PLAY_AR_W]

    // take YPos and run through playMatrix[PLAY_AR_H]

    // return false if x and y are occupied

    return true;
   }


bool doGenObject( int chancein100 )
   {
    // Initialize Variables

    // Get random number between 1 and 100
       // Function get_Random_Number

    // Return true if generated number is less or equal to than input number
    return 0;     // temporary stub return
   }


int get_Random_Number(int low_Value, int high_Value)
    {
    // Initialize Variables

    // Initialize Range Value to Inculde Both High & Low Values

    // Generate Random Value Between Low and High; Start at Low Value
       // Function: rand()    

    // return the randomly generated number
    return 0;     // temporary stub return
    }


void gameOverMessage()
   {
   // variables

   // clear the message box
      // functions: setColor, clearScreen

   // print message text
      // function: printStringAt

      // update cursor position

   // wait for user input to terminate game
      // function: waitForInput

   // no return - void
   }


void writeScoreFile( char names[ NAME_LEN ], int scores[], int candidate )
   {
    // initalize variables

    // Open file to sort
       // .open    

    // Sort scores from highest to lowest
       // sortScores

    // Close file
       // .close

   }


void sortScores( char names[ NAME_LEN], int scores[], int candidate)
   {
    // Initilize Variables

    // Takes candidate and bubble sort it through existing scores
    // Places candidate at apropiat index
       // start master loop
       
            // iterate through array and compare values
            
                 // swap values if not in order
          
  }


void showHiScores ( char names[], int scores[] )
   {
    // Initilize Variables

    // Open scoreFile
       // Function .open

    // Set Screen color and clear screen
       // Funtion: setColor, clearScreen

    // Print title
       // Function: printStringAt
   
    // Print "Names" & "Scores" at certian locations
       // Function: printStringAt

    // Display the name and scores from the file

    // Wait for player to press a button to move back to previous screen
       // function: waitForImput

    // Close File
       // .close

   }
