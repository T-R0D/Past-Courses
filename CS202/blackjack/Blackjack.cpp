////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      Final Project: "21" 
//  Created By: Terence Henriod
//  Course:     CS202
//
//  Summary:    
// 
//  Last Modified: 
//
////////////////////////////////////////////////////////////////////////////////

//============================================================================//
//= Header Files =============================================================//
//============================================================================//

// standard header files / namespace
#include <iostream>
#include <fstream>
#include <conio.h>
#include <stdlib.h>
#include <assert.h>
#include <exception>
using namespace std;

// other headers
#include "Ttools.h"

// class headers
#include "gameClass.h"

//============================================================================//
//= Macros ===================================================================//
//============================================================================//

#pragma warning (disable : 4996)   // disables strcpy, strcat warnings in Visual


//============================================================================//
//= Global Constant Definitions ==============================================//
//============================================================================//
const char* HELP = "/?";
const char* TXT = ".txt";
enum E_CODE{
  HELP_REQ = 1,
  NUM_ARGS, 
  MEM_ERROR,
  FILE_FAIL
}; 

//============================================================================//
//= Global Function Prototypes ===============================================//
//============================================================================//

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: processArgs
// Summary:       Processes all data that is passed into the program via the 
//                command line. Invokes help if necessary.
//
// Parameters:    int argc                The count of the command line
//                                        arguments entered. Includes the name
//                                        of the program.
//                char** argv             A double pointer to the array of the
//                                        command line arguments as strings.
//                unsigned int &rngseed   The parameter to be used to seed the 
//                                        random number generator.
//                char* &fileName         A pointer to be used for the string
//                                        containing the output file name.
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
void processArgs( int argc, char** argv, 
                  unsigned int &rngseed, char* &fileName );

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: dispHelp
// Summary:       Displays the help screen for the user. Then exits the program
//                with an error specific non-zero exit code.
//
// Parameters:    int code   The error code corresponding to the specific 
//                           error.
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
void dispHelp( const int code );

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: dispMainMenu
// Summary:       Displays the main program menu. Offers options to either play
//                "21" or exit the program. Accepts the user's input. The
//                calling function is responsible for handling the input
//
// Parameters:    char* fileName   The name of the output file to be used.
// Returns:       char             The users input.
//
////////////////////////////////////////////////////////////////////////////////
char dispMainMenu( const char* fileName );

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: play21
// Summary:       Initiates gameplay by instantiating a gameClass object. Uses
//                the game object to execute game play, then write the results
//                to the output file.
//
// Parameters:    char* fileName   The name of the output file to be written 
//                                 to.
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
void play21( const char* fileName );

//============================================================================//
//= Main Program Definition ==================================================//
//============================================================================//

int main( int argc, char** argv )
   {
   // vars
   unsigned int rngseed;
   char* recordFile;
   bool menuCont = true;
   char selection = '\0';

   // proccess cmd line args
   try{
     processArgs(argc, argv, rngseed, recordFile);
   }
   catch(bad_alloc&){
     cout << "   Unexpected memory error occurred - "
          << " program must now terminate." << endl
          << " If you suspect this error occurred due to the file name" << endl
          << " or the save file being unable to be opened or created," << endl
          << " please adjust your file name input accordingly";
     holdProg();
     exit(MEM_ERROR);
   }

   // seed the rng
   if(rngseed < 0){
     srand(static_cast<unsigned int>(time(NULL)));
   }
   else{
     srand(rngseed);
   } 

   // display game play menu and handle input
   while(menuCont){
     selection = dispMainMenu(recordFile);
     
     switch(selection){
       case '1':
         try{
           play21(recordFile);
         }
         catch(bad_alloc&){
           cout << "   Unexpected memory error occurred - "
                << " program must now terminate." << endl
                << " If you suspect this error occurred due to the file name" << endl
                << " or the save file being unable to be opened or created," << endl
                << " please adjust your file name input accordingly";
           holdProg();
           exit(MEM_ERROR);
         }
         break;

       case '2':
         menuCont = false;
         break;

       default:
         // do nothing, the loop will redisplay/prompt
         break;
     } 
   }

   // return dynamic memory
   delete [] recordFile;

   // return 0 upon successful completion
   return 0;
   }


//============================================================================//
//= Supporting function implementations ======================================//
//============================================================================//

void processArgs( int argc, char** argv, 
                  unsigned int &rngseed, char* &fileName )
   {
   // vars
   char* tempStr = new char [1000];
   int seed;

   // check to see if help was requested
   if(strcmp(*(argv + 1), HELP) == 0){
     dispHelp(HELP_REQ);
   }

   // check to see that argc is sufficient
   if(argc < 2 || argc > 3){
     // display help, which will end program
     dispHelp(NUM_ARGS);
   }

   // otherwise, store information as appropriate 
     // rngseed
     if(argc == 3){
       seed = atoi(*(argv + 2));
       if( seed < 0){
         rngseed = -1;
       }
       else{
         rngseed = unsigned int(seed);
       }
     }
 
     // output file name
     strcpy( tempStr, *(argv + 1));
     if( !checkExt(tempStr, TXT)){
       strcat(tempStr, TXT);
     }
   
     // copy the file name into its place, correctly sized
     fileName = new char [strlen(tempStr) + 1];  // +1 for '\0'
     strcpy(fileName, tempStr);
 
   // return dynamic memory
   delete [] tempStr;
    
   // void - no return
   }   

void dispHelp(const int code)
   {
   // display header
   cout << "      HELP MENU" << endl << endl;

   // display error specific message
   switch(code){
     case HELP_REQ:
       cout << "  Here is the help you requested:" << endl << endl;
       break;

     case NUM_ARGS:
       cout << "  Please read the following information to determine" << endl
            << "  the correct number of arguments to use." << endl << endl;
       break;

     default: 
       // currently no default message
       cout << endl << endl;
       break;
   }
        
   // display main help message
   cout << "This program is a simplified version of blackjack, aka " << endl
        << "\"21.\" This program implements simple dealing and" << endl
        << "scoring capabilities using the techniques covered in our" << endl
        << "CS 202 class. It is a command line driven program, and" << endl
        << "accepts the following arguments:" << endl << endl
        << "[0] The name of the program (blackjack.exe)" << endl
        << "[1] The name of the game results file; this should" << endl
        << "    be a .txt file." << endl
        << "[2] (optional) The random number generator seed. This" << endl
        << "    should be a positive integer." << endl
        << " Use /? as the first cmd line argument to display the help anytime" 
        << endl;

   // hold program
   holdProg();

   // no return - void
   // but do exit with non-zero exit code
   exit(code);
   }

char dispMainMenu(const char* fileName)
   {  
   // vars
   char selection = '\0';

   // clear the screen
   clrScr();

   // display menu text
   cout << "                     BLACKJACK " << endl << endl << endl
        << "     <1> Play a game of 21"
        << "     <2> Exit this program" << endl << endl << endl
        << "Note: Game results will be recorded in " << fileName << '.' << endl
        << "      When viewing the output file, the file should use a" << endl
        << "      font that will support the suit charachters (like" << endl
        << "      Terminal)." << endl
        << "      Games will be numbered according to every instance" << endl
        << "      that the program is called; timestamps should be" << endl
        << "      used to determine absolute order of games played." << endl
        << endl << endl << endl << endl << endl
        << "  Please enter the number corresponding to your selection: ";
        selection = getch();

   cout << endl << endl;
        
   // return the selection
   return selection;
   }

void play21(const char* fileName)
   {
   // instantiate a game object that will go out of scope
   // when the function ends
   gameClass* game = new gameClass;

   // execute game play
   game->play();

   // record data
   try{ // "expects" file failure exeption as stack "unwinds"
     game->recordGameData(fileName);
   }
   catch(char* ex){
     cout << ex << endl
          << "A file operation failed. Program will now terminate," << endl 
          << "please try to prevent this error";
     holdProg();
     exit(FILE_FAIL);
   }

   // return dynamic memory
   delete game;

   // no return - zero
   }