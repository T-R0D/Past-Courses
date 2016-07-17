
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
const int INSTR_UPX = 12;         
const int INSTR_UPY = 2;
const int INSTR_W   = 53;         
const int INSTR_H   = 16;

// main menu dimensions/reference points
const int MENU_UPX    = 15;      
const int MENU_UPY    = 5;       
const int MENU_W      = 30;      
const int MENU_H      = 16;
const int MENU_CURSEX = ( MENU_UPX + 8 );       
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

// typical character constants
const char NULL_CHAR = '\0';
const char SPACE     = ' ';
const char NEW_LINE  = '\n';

// game area constants
const int PLAY_AR_W    = 23;              
const int PLAY_AR_H    = 62;             
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
 
   // user input variables

   // game related variables 

   // flags

   // random seed

// initialize program

   // initialize curses
      // functions: startCurses, setColor

// display splash screen
   // function: dispSplash

   // if 'E' was pressed, activate extra credit mode
      // operation: if

        // set extra credit mode flag

// display instructions screen (the plan is to run the game in extra credit mode by default)
   // function: dispInstructions

// read in high scores, if file is present, ensure arrays are clean 
   // function: readScores

// implement menu (loop until valid option is chosen)
   // operation: while

     // display menu
        // clear screen
           // function: setColor, clearScreen

        // print menu text, operate menu options
           // function: printMenuText        

     // operate menu options

          // operation: switch

             // case: set difficulty

                // function: setDiff

             // case: play the game

                // get player name
                   // function: promptForPlayer

                // play the game!
                  // function: executeGame

                // write high score file
                  // function: writeScores

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

char dispSplash ()
    {
    return 0;     // temporary stub return
    }


void printASCIIart( char letter, int &currX, int &currY)
   {
   // no return - void
   }


void dispInstructions( bool xMode )
   {
   // no return - void 
   }


void readHiScores ( char names[ NAME_LEN ], int scores[] )
   {
    // no return - void
   }


char printMenuText()
   {
   return 0;     // temporary stub return
   }


int setDiff()
   {
   return 0;     // temporary stub return
   }


void promptForPlayer( char name[ NAME_LEN ] )
   {
   // no return - void
   }


int executeGame( bool xcMode, int &score, char name[] )
   {
   return 0;     // temporary stub return          
   }


bool isFileThere ( const string &fileName )
     {
     return 0;     // temporary stub function
     }

void fileMissingMessage()
   {
   // no return - void
   }


void makeScoreBoard( bool xcMode )
   {
   // no return - void   
   }


void makeOutWall( int width, int height )
   {
     // no return - void
   }

void clearMatrix( short matrix[PLAY_AR_W][PLAY_AR_H] )
   {
   // no return - void
   }


void setFloor( int floorNumber, short playMatrix[PLAY_AR_W][PLAY_AR_H] )
   {
   // no return - void
   } 


int readFloorData( char fileName[], int Xcoordinates[], int Ycoordinates[] )
   {
   return 0;     // temporary stub function
   }


void genObjectAt( int Xcoord, int Ycoord, short object, short playMatrix[PLAY_AR_W][PLAY_AR_H] )
   {
   // no return - void
   } 


void matrixToScreen(int matrixX, int matrixY, short playMatrix[PLAY_AR_W][PLAY_AR_H] )
   {
   // no return - void
   }


void keepScore( bool xcMode, int cloaks, int swords, int spells, int treasure, int &score )
   {
   // no return - void 
   }


void movePlayer ( int waitTime, bool &continueProgram )
    {
    // no return - void
    }


bool collisionTest( int &cloakTally,  int &swordTally,  int &spellTally, int &treasureTally,
                    int intendedX, int intendedY, short playMatrix[PLAY_AR_W][PLAY_AR_H], bool &continueGame )
   {
   return 0;     // temporary stub return 
   }


bool move( int &xPos, int &yPos, int xVect, int yVect )
   {
    return 0;     // temporary stub return
   }


void screenToMatrix( int screenX, int screenY, short object, short playMatrix[PLAY_AR_W][PLAY_AR_H] )
   {
   // no return - void
   }


void genObjects( bool xcMode, short gameBoard[ PLAY_AR_W ][ PLAY_AR_H ] )
   {
   // no return - void
   }


void XYPos ( int &xPos, int &yPos )
   {
    // no return - void    
   }


bool spaceOccupied ( int XPos, int YPos, short playMatrix[PLAY_AR_W][PLAY_AR_H])
   {
    return 0;     // temporary stub return
   }


bool doGenObject( int chancein100 )
   {
    return 0;     // temporary stub return
   }


int get_Random_Number(int low_Value, int high_Value)
    {
    return 0;     // temporary stub return
    }


void gameOverMessage()
   {
   // no return - void
   }


void writeScoreFile( char names[ NAME_LEN ], int scores[], int candidate )
   {
    // no return - void
   }


void sortScores( char names[ NAME_LEN], int scores[], int candidate)
   {
   // no return - void       
  }


void showHiScores ( char names[], int scores[] )
   {
    // no return - void 
   }
