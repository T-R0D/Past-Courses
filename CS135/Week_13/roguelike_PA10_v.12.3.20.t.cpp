
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
const int UPRLEFTX   = 0;
const int UPRLEFTY   = 0;
const int SCR_W      = 80;
const int SCR_H      = 28;
const int SCR_CENTER = 39;

// instruction screen dimensions/reference points
const int INSTR_UPX = 12;         ////////////////////////////////////////////////
const int INSTR_UPY = 2;
const int INSTR_W   = 53;         ////////////////////////////////////////////////
const int INSTR_H   = 16;

// main menu dimensions/reference points
const int MENU_UPX    = 15;      //////////////////////////////////////////////////
const int MENU_UPY    = 5;       //////////////////////////////////////////////////
const int MENU_W      = 30;      //////////////////////////////////////////////////
const int MENU_H      = 16;
const int MENU_CURSEX = ( MENU_UPX + 8 );       ///////////////////////////////////////////////////
const int MENU_CURSEY = ( MENU_UPY + 3 );       

// difficulty/name entry screen reference points 
const int NAME_SCR_UPX = 0;
const int NAME_SCR_UPY = 0;
const int NAME_SCR_W   = 10;
const int NAME_SCR_H   = 20;

// game object and related constants
const char PLAYER      = '*';
const char CLOAK       = char(247);
const char HOLE        = char(178); 
const char TREASURE    = char(232);
const char SWORD       = char(157);
const char MONSTER     = char(206);
const char SPELL       = char(234);
const char IN_WALL     = char(176);
const char OUT_WALL    = char(176);
const char UP_STAIR    = 'U';
const char DWN_STAIR   = 'D';
const int EASY_DIFF    = 5;
const int MED_DIFF     = 3;
const int HARD_DIFF    = 1;
const int DEFAULT_DIFF = EASY_DIFF;
const int MAX_WALLS    = 75;

// typical character constants
const char NULL_CHAR = '\0';
const char SPACE     = ' ';
const char NEW_LINE  = '\n';

// game area constants
const int PLAY_AR_W    = 62;              ///////////////////////////////
const int PLAY_AR_H    = 23;              ///////////////////////////////
const int SCOREBOARD_W = 16;
const int SCOREBOARD_H = 25;
const int SCORE_X      = 14;
const int MESS_UPX     = 2;               
const int MESS_UPY     = 20;              ///////////////////////////////
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
  Device Output/Monitor: Displays Title, and developers' names
  Dependencies: formatted_console_io_v18, curses
  Developer: Nicholas Smith
*/
char dispSplash ();


/* 
Name: printASCIIart
Process: prints one character as ASCII art at the entered position, or moves the
         cursor position depending on the parameter value entered
Function Input/Parameters: the character to be printed (char), the current or desired
                           cursor coordinates (int)
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: displays an ASCII art character on the screen
Dependencies: formatted_console_io_v18, curses
Developer: Terence Henriod
*/
void printASCIIart( char letter, int &currX, int &currY);


/* 
Name: dispInstructions
Process: displays an instruction screen for the game, varies depending on which
         game mode is active
Function Input/Parameters: a flage indicating whether or not extra credit mode has 
                           been activated (bool)
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: displays the appropriate instruction screen
Dependencies: formatted_console_io_v18, curses
Developer: Terence Henriod
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
Name: printMenuText
Process: displays the text of the menu, captures the menu option that the user 
         chooses 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: returns the user's menu selection (char)
Device Input: user's menu selection (char)
Device Output: displays the menu text
Dependencies: formatted_console_io_v18, curses
Developer: Terence Henriod
*/
char printMenuText();


/* 
Name: setDiff
Process: displays the difficulty setting screen, collects the user's choice, converts
         that choice into a meaningful term for the program
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: a wait time corresponding to the difficulty chosen by the 
                          user 
Device Input: the user's difficulty selection (int)
Device Output: displays the set difficulty screen
Dependencies: formatted_console_io_v18
Developer: Terence Henriod
*/
int setDiff();


/* 
Name: promptForPlayer
Process: displays the player name entry screen and collects the player name
Function Input/Parameters: (by reference) a c-style string to contain the entered
                           name
Function Output/Parameters: (by reference) a c-style string containing the entered player name
Function Output/Returned: none
Device Input: prompts for the player name
Device Output: displays the player name entry screen
Dependencies: formatted)console_io_v18, curses
Developer: Terence Henriod
*/
void promptForPlayer( char name[ NAME_LEN ] );


/* 
Name: executeGame
Process: carries out all processes associated with game play
Function Input/Parameters: the extra credit mode flag (bool), (by reference) the
                           score currently acheived by the player (int), (by reference)
                           the player's name (c-style string)
Function Output/Parameters: (by reference) the score achieved by the player
Function Output/Returned: function ends by returning zero
Device Input: gameplay actions (managed by movePlayer)
Device Output: outputs the gameplay to the screen
Dependencies: formatted_console_io_v18, curses
Developer: Terence Henriod
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
Name: fileMissingMessage
Process: if level files are missing before gameplay starts, a message is displayed
         notifying the user why the program will not continue
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: the "level file missing" message, written on the previously created
               screen promptForPlayer screen 
Dependencies: none
Developer: Terence Henriod
*/
void fileMissingMessage();


/* 
Name: makeScoreBoard
Process: initialized the scoreboard portion of the gameplay screen, differs slightly
         depending on game mode
Function Input/Parameters: the extra credit flag (bool)
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: displays the initial scoreboard
Dependencies: formatted_console_io_v18
Developer: Terence Henriod
*/
void makeScoreBoard( bool xcMode );


/*
  Name: makeOutWall
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
Name: clearMatrix
Process: loads the entire gameplay array with only spaces
Function Input/Parameters: (by reference) the gameplay matrix (short)
Function Output/Parameters: (by reference) the gameplay matrix (short)
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer: Terence Henriod
*/
void clearMatrix( short matrix[PLAY_AR_W][PLAY_AR_H] );


/* 
Name: setFloor
Process: loads the game matrix with walls, stairs and empty space; sets the player 
         at the start position
Function Input/Parameters: the floor number the player is heading to (int), (by reference)
                           the gameplay matrix (short)
Function Output/Parameters: (by reference) the gameplay matrix (short)
Function Output/Returned: none
Device Input: none
Device Output: displays a refreshed gameplay area on the screen
Dependencies: formatted_console_io_v18, curses
Developer: Terence Henriod
*/
void setFloor( int floorNumber, short playMatrix[PLAY_AR_W][PLAY_AR_H] );



/* 
Name: readFloorData
Process: reads the interior wall layout into the program
Function Input/Parameters: the name of the file level to be read in (c-style string),
                           the two arrays that will hold the coordinates of the 
                           interior wall placements (int[]
Function Output/Parameters: (by reference) the two arrays that will hold the coordinates 
                            of the interior wall placements (int[]
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: <fstream>
Developer: Terence Henriod
*/
int readFloorData( char fileName[], int Xcoordinates[], int Ycoordinates[] );


/* 
Name: genObjectAt
Process: generates a specified object at the specified location in the game matrix,
         then prints that object on the screen 
Function Input/Parameters: the coordinates where the object should be generated in 
                           the gameplay matrix (int), the character object to be 
                           generated (short), (by reference)the gameplay matrix (short)
Function Output/Parameters: (by reference)the gameplay matrix (short)
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer: Terence Henriod
*/
void genObjectAt( int Xcoord, int Ycoord, short object, short playMatrix[PLAY_AR_W][PLAY_AR_H] );


/* 
Name: matrixToScreen
Process: takes a set of gameplay matrix coordinates, and takes the object stored in that 
         location, and outputs that object in the appropriate location on the screen
Function Input/Parameters: game matrix coordinates (int), the gameplay matrix (short)
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: displays the item in the gamematrix to the screen
Dependencies: formatted_console_io_v18, curses
Developer: Terence Henriod
*/
void matrixToScreen(int matrixX, int matrixY, short playMatrix[PLAY_AR_W][PLAY_AR_H] );


/* 
Name: keepScore
Process: recalculates the score and updates the score board appropriately for 
         the current game mode
Function Input/Parameters: the extra credit mode flag (bool), the item tallies (int),
                           (by reference) the current player score (int)
Function Output/Parameters: (by reference) the current player score (int)
Function Output/Returned: none
Device Input: none
Device Output: updates item tallies and the score on the scoreboard
Dependencies: formatted_console_io_v18, curses
Developer: Terence Henriod
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
Name: collisionTest
Process: if a "collision" between the player an a game item is about to occur,
         the function lets the movement function know if the player is allowed to move
         and if the game should continue, nad updates item tallies as appropriate
Function Input/Parameters: item tallies (int), the game matrix coordinates the player
                           is about to move into (int) the gameplaymatrix (short[][]),
                           and the flag indicating if gameplay should be allowed to
                           continue 
Function Output/Parameters: (by reference) the item tallies (int), the "continue game"
                            flag (bool)
Function Output/Returned: the decision of whether or not the player should be allowed
                          to move (bool)
Device Input: none
Device Output: none
Dependencies: none
Developer: Nicholas Smith & Terence Henriod
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
Name: screenToMatrix
Process: if something changes on the screen (i.e. the player moves), updates the 
         game matrix accordingly
Function Input/Parameters: the screen coordinates the object was displayed at (int),
                           the object that was put on the screen (short), (by reference)
                           the gameplay matrix (short[][])
Function Output/Parameters: (by reference) the gameplay matrix (short[][])
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
Developer: Terence Henriod
*/
void screenToMatrix( int screenX, int screenY, short object, short playMatrix[PLAY_AR_W][PLAY_AR_H] );


/* 
Name: genObjects
Process: decides whether or not to generate activated game objects (i.e. items, dangers),
         and then generates them in free spaces both on the game board and the screen
Function Input/Parameters: the extra credit mode flag (bool), (by reference) the 
                           gameplay matrix (short[][])
Function Output/Parameters: by reference) the gameplay matrix (short[][])
Function Output/Returned: none
Device Input: none
Device Output: displays the generated objects on the screen
Dependencies: none
Developer: Terence Henriod
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
Name: doGenObject
Process: makes a decision whether or not to generate an object based of an integer
         number chance in 100
Function Input/Parameters: the chance in 100 an item should be generated (int)
Function Output/Parameters: none
Function Output/Returned: the decision whether or not to generate that item (bool)
Device Input: none
Device Output: none
Dependencies: none
Developer: Nicholas Smith
*/
bool doGenObject( int chancein100 );


/* 
Name: get_Random_Number
Process: produces a random number in the input range
Function Input/Parameters: the low and high ends of the desired range (int)
Function Output/Parameters: none
Function Output/Returned: a random number in the desred range
Device Input: none
Device Output: none
Dependencies: random number generator
Developer: Nicholas Smith
*/
int get_Random_Number(int low_Value, int high_Value);


/* 
Name: gameOverMessage
Process: displays the game over message on the scoreboard
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: displays the game over message on the scoreboard
Dependencies: formatted_console_io_v18, curses
Developer: Terence Henriod
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
   int difficulty = DEFAULT_DIFF;

   // flags
   bool XCmode = true;
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
           clearScreen( UPRLEFTX, UPRLEFTY, (UPRLEFTX + SCR_W), (UPRLEFTY + SCR_H) );
           setColor( COLOR_WHITE, COLOR_BLUE, SET_BRIGHT);
           clearScreen( MENU_UPX, MENU_UPY, (MENU_UPX + MENU_W), (MENU_UPY + MENU_H) );         

     // display menu text
        // function: printMenuText
        userInput = printMenuText();

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
   int xPos = SCR_CENTER;
   int yPos = (INSTR_UPY + 1);

   // paint gray background
      // functions: setColor, clearScreen
      setColor( COLOR_BLUE, COLOR_WHITE, SET_BRIGHT );
      clearScreen( INSTR_UPX, INSTR_UPY, (INSTR_UPX + INSTR_W), (INSTR_UPY + INSTR_H) );

      // paint a bigger background if extra credit mode is enabled
         // operation: if
         if( xMode )
           {
           //function: clearScreen
            clearScreen( INSTR_UPX, (INSTR_UPY + INSTR_H), (INSTR_UPX + INSTR_W), (INSTR_UPY + INSTR_H + 4) );
           }

   // write title line w/ underline
      // functions: printStringAt
      printStringAt( xPos, yPos, "Rogue-Like Instructions", "CENTER" );
      xPos = SCR_CENTER, yPos ++;
      printStringAt( xPos, yPos, "=======================", "CENTER" );
      yPos += 2;

   // write cloak/hole info
      // functions: printStringAt, setColor, printSpecialCharAt
      xPos = (INSTR_UPX + 2); 
      printStringAt( xPos, yPos, "Green", "LEFT" );
      xPos += 6;
      setColor( COLOR_GREEN, COLOR_WHITE, SET_BRIGHT );
      printSpecialCharAt( xPos, yPos, CLOAK );
      xPos ++;
      setColor( COLOR_BLUE, COLOR_WHITE, SET_BRIGHT );
      printStringAt( xPos, yPos, " are magic cloaks that can cover holes", "LEFT" );
      xPos += 39;
      setColor( COLOR_RED, COLOR_WHITE, SET_BRIGHT );
      printSpecialCharAt( xPos, yPos, HOLE );
      setColor( COLOR_BLUE, COLOR_WHITE, SET_BRIGHT );
      yPos += 2;

   // if game is in extra credit mode, display extra credit info
      // operation: if
      if( xMode )
        {
        // write sword/monster info
           // functions: printStringAt, setColor, printSpecialCharAt
           xPos = (INSTR_UPX + 6); 
           printStringAt( xPos, yPos, "Green", "LEFT" );
           xPos += 6;
           setColor( COLOR_GREEN, COLOR_WHITE, SET_BRIGHT );
           printSpecialCharAt( xPos, yPos, SWORD );
           xPos ++;
           setColor( COLOR_BLUE, COLOR_WHITE, SET_BRIGHT );
           printStringAt( xPos, yPos, " are swords that kill monsters", "LEFT" );
           xPos += 31;
           setColor( COLOR_RED, COLOR_WHITE, SET_BRIGHT );
           printSpecialCharAt( xPos, yPos, MONSTER );
           setColor( COLOR_BLUE, COLOR_WHITE, SET_BRIGHT );
           yPos += 2;

        // write spell/wall info
           // functions: printStringAt, setColor, printSpecialCharAt
           xPos = (INSTR_UPX + 5); 
           printStringAt( xPos, yPos, "Green ", "LEFT" );
           xPos += 6;
           setColor( COLOR_GREEN, COLOR_WHITE, SET_BRIGHT );
           printSpecialCharAt( xPos, yPos, SPELL );
           xPos ++;
           setColor( COLOR_BLUE, COLOR_WHITE, SET_BRIGHT );
           printStringAt( xPos, yPos, " are spells that go through walls", "LEFT" );
           xPos += 34;
           setColor( COLOR_BLUE, COLOR_WHITE, SET_BRIGHT );
           printSpecialCharAt( xPos, yPos, IN_WALL );
           setColor( COLOR_BLUE, COLOR_WHITE, SET_BRIGHT );
           yPos += 2;
        }

   // write treasure info
      // functions: printStringAt, setColor, printSpecialCharAt
      xPos = (INSTR_UPX + 12); 
      printStringAt( xPos, yPos, "Yellow", "LEFT" );
      xPos += 7;
      setColor( COLOR_YELLOW, COLOR_WHITE, SET_BRIGHT );
      printSpecialCharAt( xPos, yPos, TREASURE );
      xPos ++;
      setColor( COLOR_BLUE, COLOR_WHITE, SET_BRIGHT );
      printStringAt( xPos, yPos, " are golden treasure", "LEFT" );
      setColor( COLOR_BLUE, COLOR_WHITE, SET_BRIGHT );
      yPos += 2;

   // write stair info
      // functions: setColor, printCharAt, printStringAt
      xPos = (INSTR_UPX + 8); 
      setColor( COLOR_WHITE, COLOR_BLUE, SET_BRIGHT );
      printCharAt( xPos, yPos, DWN_STAIR );
      xPos ++;  
      setColor( COLOR_BLUE, COLOR_WHITE, SET_BRIGHT );
      printStringAt( xPos, yPos, " and", "LEFT" );
      xPos += 5;
      setColor( COLOR_WHITE, COLOR_BLUE, SET_BRIGHT );
      printSpecialCharAt( xPos, yPos, UP_STAIR );
      xPos ++;
      setColor( COLOR_BLUE, COLOR_WHITE, SET_BRIGHT );
      printStringAt( xPos, yPos, " lead to Upper and Lower floors", "LEFT" );
      setColor( COLOR_BLUE, COLOR_WHITE, SET_BRIGHT );
      yPos += 2;      

   // write condition(s) for ending the game
      // functions: printStringAt
      xPos = (INSTR_UPX + 20);
      printStringAt( xPos, yPos, "Hitting a hole", "RIGHT" );
      xPos ++;
   
      // if extra credit mode is active
         // operation: if
         if( xMode )
           {
           // print any extra conditions
              // function: printStringAt
              printStringAt( xPos, yPos, " or a monster", "LEFT" );
              xPos += 14;          
           }

    // finish the game end condition statement
       // function: printString
       printStringAt( xPos, yPos, "ends the game", "LEFT" );
       yPos += 2;       

   // write object of game message
      // functions: printStringAt
      xPos = (INSTR_UPX + 6);
      printStringAt( xPos, yPos, "Try to keep moving for as long as possible", "LEFT" );
      yPos += 2;
      xPos = (INSTR_UPX + 5);
      printStringAt( xPos, yPos, "More treasure and weapons get a better score", "LEFT" );

   // hold the system for the user
      // function: waitForInput
      waitForInput( FIXED_WAIT );     

   // no return - void 
   }


void readHiScores ( char names[ NAME_LEN ], int scores[] )
   {
    // Initalize Variables
    
    // Loop to read in files, go until index < 11 || fin !.good
   
       // read in name into array
      
       // read in score into array

       // update index 

   }


char printMenuText()
   {
   // set cursor position/variables
   int xLoc = MENU_CURSEX;
   int yLoc = MENU_CURSEY;
   char input = '+';

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
      input = promptForCharAt( xLoc, yLoc, "Enter Selection: " );

   // return the input entered by the user
   return 0;     // temporary stub return
   }


int setDiff()
   {
   // variables
   int input = DEFAULT_DIFF;
   int xPos = SCR_CENTER;
   int yPos = (NAME_SCR_UPY + 1);

   // clear screen to black, then a small area to blue
      // functions: setColor, clearScreen
      setColor( COLOR_WHITE, COLOR_BLACK, SET_BRIGHT );
      clearScreen( UPRLEFTX, UPRLEFTY, (UPRLEFTX + SCR_W), (UPRLEFTY + SCR_H) );
      setColor( COLOR_WHITE, COLOR_BLUE, SET_BRIGHT );
      clearScreen( UPRLEFTX, UPRLEFTY, (UPRLEFTX + SCR_W), (UPRLEFTY + SCR_H) );

   // display "screen title"
      // function: printStringAt
      printStringAt( xPos, yPos, "Game Data Entry Module", "CENTER" );

   // loop until proper entry is received
      // operation: while 
      while( input < 1 || input > 3 )
        {
      // display prompt
         // function: promptForIntAt
         yPos = (NAME_SCR_UPY + 4);
         xPos = (NAME_SCR_UPX + 4);
         input = promptForIntAt( xPos, yPos, "Enter Game Difficulty (1-3): " );
        }
   // convert user selection to meaningful program terms 
      // operation: switch
      switch( input )
        {  
        // case: easy ( 1 -> 5 )
        case 1:
             input = EASY_DIFF;
            break;

        // case: medium ( 2 -> 3 )
        case 2: 
             input = MED_DIFF;
            break;

        // case: hard ( 3 -> 1 )
        case 3:
             input = HARD_DIFF;
            break;
        }

   // return difficulty in meaningful program terms
   return input; 
   }


void promptForPlayer( char name[ NAME_LEN ] )
   {
   // variables
   int xPos = (NAME_SCR_UPX + 5);
   int yPos = (NAME_SCR_UPY + (NAME_SCR_H / 2));

   // clear screen to black, then a small area to blue
      // functions: setColor, clearScreen
      setColor( COLOR_WHITE, COLOR_BLACK, SET_BRIGHT );
      clearScreen( UPRLEFTX, UPRLEFTY, (UPRLEFTX + SCR_W), (UPRLEFTY + SCR_H) );
      setColor( COLOR_WHITE, COLOR_BLUE, SET_BRIGHT );
      clearScreen( UPRLEFTX, UPRLEFTY, (UPRLEFTX + SCR_W), (UPRLEFTY + SCR_H) ); 
       
   // prompt for user name, storing the name in a C-style string
      // function: promptForStringAt (the void one)
      promptForStringAt( xPos, yPos, name );

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
          fin.close();

       // return true to indicate file was there
       return true;     
      }

void fileMissingMessage()
   {
   // variables
   int xPos = (NAME_SCR_UPX + 4);
   int yPos = (NAME_SCR_UPY + 2);

   // clear screen to black, then clear a message box; uses promptForPlayer dimensions
      // functions: setColor, clearScreen
      setColor( COLOR_WHITE, COLOR_BLACK, SET_BRIGHT );
      clearScreen( UPRLEFTX, UPRLEFTY, (UPRLEFTX + SCR_W), (UPRLEFTY + SCR_H) );
      setColor( COLOR_WHITE, COLOR_BLUE, SET_BRIGHT );
      clearScreen( UPRLEFTX, UPRLEFTY, (UPRLEFTX + SCR_W), (UPRLEFTY + SCR_H) );

   // print the error message on the screen
      // function: printStringAt
 
      // title line
      printStringAt( xPos, yPos, "ERROR: Level File(s) Missing", "LEFT" );
      xPos = (NAME_SCR_UPX + 2);
      yPos += 2;

      // main message
      printStringAt( xPos, yPos, "One or more of the level files is missing;", "LEFT" );
      xPos = (NAME_SCR_UPX + 2);
      yPos ++;
      printStringAt( xPos, yPos, "Please ensure that the level files are in place.", "LEFT" );

   // move the cursor to the position where system hold message will be displayed
   xPos = (NAME_SCR_UPX + 4);
   yPos += 2;

   // print the holding message and hold the screen for the user
      // functions: printStringAt, waitForInput
      printStringAt( xPos, yPos, "Press any key to continue . . .", "LEFT" );
      waitForInput( FIXED_WAIT );      

   // no return - void
   }


void makeScoreBoard( bool xcMode )
   {
   //variables
   int xPos = (UPRLEFTX + 2);
   int yPos = (UPRLEFTY + 1);

   // paint scoreboard background
      // functions: setColor, clearScreen
      setColor( COLOR_WHITE, COLOR_BLUE, SET_BRIGHT );
      clearScreen( UPRLEFTX, UPRLEFTY, SCOREBOARD_W, SCR_H );

   // print the title
     // function: printStringAt
     printStringAt( xPos, yPos, "ROGUE - LIKE", "LEFT" );

   // create tool item tallies, update cursor position
   xPos = (UPRLEFTX + 1);
   yPos += 2;              

      // cloak tally
         // functions: printStringAt, printIntAt
         printStringAt( xPos, yPos, "Cloak:", "LEFT" );
         printIntAt( (SCOREBOARD_W - 2), yPos, 0, "LEFT" );
         xPos = (UPRLEFTX + 1);
         yPos ++;      

      // extra credit item tallies if extra credit mode is active
         // operation: if
         if( xcMode ) 
           {
            // sword tally
               // functions: printStringAt, printIntAt          
               printStringAt( xPos, yPos, "Sword:", "LEFT" );
               printIntAt( (SCOREBOARD_W - 2), yPos, 0, "LEFT" );
               xPos = (UPRLEFTX + 1);
               yPos ++;
 
            // spell tally
               // functions: printStringAt, printIntAt
               printStringAt( xPos, yPos, "Spell:", "LEFT" );
               printIntAt( (SCOREBOARD_W - 2), yPos, 0, "LEFT" );
               xPos = (UPRLEFTX + 1);
               yPos ++;
           }

   // create treasure tally
      // functions: printStringAt, printIntAt
      xPos = (UPRLEFTX + 1);
      yPos ++;
      printStringAt( xPos, yPos, "Treasure:", "LEFT" );
      printIntAt( (SCOREBOARD_W - 2), yPos, 0, "LEFT" );
      xPos = (UPRLEFTX + 1);
      yPos += 2;

   // create score tally
      // functions: printStringAt, printIntAt
      xPos = (UPRLEFTX + 1);
      yPos ++;
      printStringAt( xPos, yPos, "Score:", "LEFT" );
      printIntAt( (SCOREBOARD_W - 2), yPos, 0, "LEFT" );

   // write game directions
   
      // movement directions
         // function: printStringAt
         xPos = (UPRLEFTX + 2);
         yPos += 2;
         printStringAt( xPos, yPos, "Arrows set", "LEFT" );
         xPos = (UPRLEFTX + 2);
         yPos ++;
         printStringAt( xPos, yPos, "direction", "LEFT" );

      // stair instructions
         // function: printStringAt
         xPos = (UPRLEFTX + 2);
         yPos += 2;
         printStringAt( xPos, yPos, "Follow U/D", "LEFT" );
         xPos = (UPRLEFTX + 2);
         yPos ++;
         printStringAt( xPos, yPos, "to change", "LEFT" );
         xPos = (UPRLEFTX + 2);
         yPos ++;
         printStringAt( xPos, yPos, "level", "LEFT" );

      // how the game ends
         // function: printStringAt
         xPos = (UPRLEFTX + 2);
         yPos += 2;
         printStringAt( xPos, yPos, "Falling into", "LEFT" );
         xPos = (UPRLEFTX + 3);
         yPos ++;
         printStringAt( xPos, yPos, "a hole", "LEFT" );
         xPos = (UPRLEFTX + 2);
         yPos ++;
         printStringAt( xPos, yPos, "or pressing ESC", "LEFT" );
         xPos = (UPRLEFTX + 3);
         yPos ++;
         printStringAt( xPos, yPos, "ends game", "LEFT" );

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
   int wIndex = 0;
   int hIndex = 0;

   // use loop to fill all positions in matrix with spaces
   // pick a row and fill it, then move on
      // operation: for
      for( hIndex = 0; hIndex < PLAY_AR_H; hIndex ++ )
         {
         // iterate through the row
            // operation: for
            for( wIndex = 0; wIndex < PLAY_AR_W; wIndex ++ )
               {
               // store a space to the current matrix position
               matrix[wIndex][hIndex] = SPACE;
               // loop ends when every position in row is filled
               }

         // loop ends when all columns have been visited 
         }

   // no return - void
   }


void setFloor( int floorNumber, short playMatrix[PLAY_AR_W][PLAY_AR_H] )
   {
   // variables
   int AxPos = 0;
   int AyPos = 0;
   int BxPos = 0;
   int ByPos = 0;
   int index  = 0;
   int xCoords[MAX_WALLS];
   int yCoords[MAX_WALLS];
   int numWalls = 0;

   // clear the play matrix
      // function: clearMatrix
      clearMatrix( playMatrix );

   // load coordinate arrays depending on which floor is selected
      // operation: switch
      switch( floorNumber )
         {
         // case: floor 1
         case 1:
                // function: readFloorData
                numWalls = readFloorData( FLOOR1, xCoords, yCoords );
                break;

         // case: floor 2
         case 2: 
                // function: readFloorData
                numWalls = readFloorData( FLOOR2, xCoords, yCoords );
                break;

         // case: floor 3
         case 3:
                // function: readFloorData
                numWalls = readFloorData( FLOOR3, xCoords, yCoords );
                break;

         }

   // load walls into game matrix by looping through coordinates until the walls have been "used up"
      // operation: for
      for( index = 0; index < numWalls; index ++ )
         {
         playMatrix[ xCoords[index] ][ yCoords[index] ] = INT_WALL;
         }

   // reload player into matrix
   playMatrix[GAMESTART_X][GAMESTART_Y] = PLAYER; 

   // reload stairs, depending on level
      // operation: switch
      switch( floorNumber )
         {
         // case: setting floor 1 (only up stairs)
         case 1:
               // function: genObjectAt
              genObjectAt();
              break;
  
         // case: setting floor 2 (both stairs)
         case 2:
               // function: genObjectAt
               
               break;
         // case: setting floor 3 (only down stairs)
         case 3:
               // function: genObjectAt
 
               break;
         }

   // output walls to display, again by looping
      // operation: for
      for( index = 0; index < numWalls; index ++ )
         {      
         // use coordinates stored in X and Y arrays to output walls stored in matrix to display
            // functions: setColor, matrixToScreen  
            setColor( COLOR_BLUE, COLOR_WHITE, SET_BRIGHT );
            matrixToScreen( xCoords[index], yCoords[index], playMatrix );
         }

   // output player to screen
      // functions: setColor, matrixToScreen
      setColor( COLOR_WHITE, COLOR_BLUE, SET_BRIGHT );
      matrixToScreen( GAMESTART_X, GAMESTART_Y, playMatrix );

   // output stairs to screen
      // operation: switch
      switch( floorNumber )
         {
         // case: setting floor 1 (only up stairs)
 
                // function: setColor, matrixToScreen

         // case: setting floor 2 (both stairs)

                // function: setColor, matrixToScreen

         // case: setting floor 3 (only down stairs)

                // function: setColor, matrixToScreen
         }
   // no return - void
   } fdsakadfsjdafshja
 

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
   int xPos;
   int yPos;
 
   // convert matrix x and y coordinates to screen coordinates
   xPos = (matrixX + SCOREBOARD_W + 1);
   yPos = (matrixY + 1);

   // print the object in the matrix to the screen
      // function: printSpecialCharAt
      printSpecialCharAt( xPos, yPos, playMatrix[matrixX][matrixY] );

   // no return - void
   }


void keepScore( bool xcMode, int cloaks, int swords, int spells, int treasure, int &score )
   {
   // variables
   int xPos = (SCOREBOARD_W - 2);
   int yPos = 5;

   // update item tallies

      // cloak tally
         // function: printIntAt
         printIntAt( xPos, yPos, cloaks, "LEFT" );
         yPos ++;

      // extra credit items if extra credit mode is active
         // operation: if
         if( xcMode )
           {
            // sword tally
               // function: printIntAt 
               printIntAt( xPos, yPos, swords, "LEFT" );
               yPos ++;
             
            // spell tally
               // function: printIntAt
               printIntAt( xPos, yPos, spells, "LEFT" );
               yPos ++;
           }

      // treasure tally
         // function: printIntAt
         yPos ++;
         printIntAt( xPos, yPos, treasure, "LEFT" );
         yPos ++;


   // re-calculate score, if extra credit mode is inactive, extra credit item tallies will be zero
      // operation: math
      score = ( cloaks + swords + spells + (5 * treasure) );

   // update score
      // function: printIntAt
         yPos ++;
         printIntAt( xPos, yPos, score, "LEFT" );
         yPos ++;

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


void screenToMatrix( int screenX, int screenY, short object, short playMatrix[PLAY_AR_W][PLAY_AR_H] )
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
       // if spaceOccupied = ture, variable = false

    // no return - void
   }


bool spaceOccupied ( int XPos, int YPos, short playMatrix[PLAY_AR_W][PLAY_AR_H])
   {
    // Initalize Variables
    char object;
    bool response = false;

    // takes XPos and runs it through playMatrix
    object = playMatrix[XPos][YPos];

    // return false if x and y are occupied
       // operation: if
       if( object != SPACE )
         {
          response = true;
         }

    return response;
   }


bool doGenObject( int chancein100 )
   {
    // Initialize Variables
    bool response;
    int number;

    // Get random number between 1 and 100
       // Function get_Random_Number
       number = get_Random_Number(1, 100);       

    // Return true if generated number is less or equal to than input number
       // operation: if
       if( number <= chancein100 )
         {
          response = true;
         }
       else
         {
          response = false;
         }

    return response;
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
   int xPos = MESS_UPX + (MESSAGE_W / 2);
   int yPos = NAME_SCR_UPY + 3;

   // clear the message box
      // functions: setColor, clearScreen
      setColor( COLOR_YELLOW, COLOR_RED, SET_BRIGHT );
      clearScreen( MESS_UPX , MESS_UPY, (MESS_UPX + MESSAGE_W), (MESS_UPY + MESSAGE_H) );

   // print message text
      // function: printStringAt
      xPos = 
      yPos = 
      printStringAt( xPos, yPos, "GAME", "CENTER");
      xPos = 
      yPos += 2;
      printStringAt( xPos, yPos, "ENDS", "CENTER");

   // wait for user input to terminate game
      // function: waitForInput
      waitForInput( FIXED_WAIT );

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
