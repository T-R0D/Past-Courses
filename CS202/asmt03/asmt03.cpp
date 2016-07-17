////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      asmt03: Text Processing: Find and Replace
//  Created By: Terence Henriod
//  Course:     CS202
//
//  
//
////////////////////////////////////////////////////////////////////////////////

//============================================================================//
//= Header Files =============================================================//
//============================================================================//

// standard headers
#include <iostream>
#include <fstream>
#include <conio.h>
#include <stdlib.h>
#include "Ttools.h"
using namespace std;

//============================================================================//
//= Macros ===================================================================//
//============================================================================//

#pragma warning (disable : 4996)   // disables strcpy, strcat warnings
#pragma warning (disable : 4244)   // disables time(NULL) warnings

#define TEXT_USER_ACTION 1         // switch for user interaction elements

//============================================================================//
//= Global Constant Definitions ==============================================//
//============================================================================//

// characters
const char D_SMILE = char(1);   // silly initialization values that are not 
const char BELL = char(7);      // expected to be found in the file to help 
                                // in debugging
// integers

  // help (exit) codes
  const int HELP_REQ     = 1;
  const int HELP_NUMARGS = 2;
  const int HELP_ASCIICODE = 3;
  const int HELP_INPUT_FORMAT = 4;
  const int NOT_ENOUGH_MEM = 5;

  // string/array/pointer capacities
  const int NAME_L = 50;
  const int MAX_LINE_L = 257;   // 256 + 1 to include '\0'
  const int MAX_LINES = 1000;  // max file length

  // others
  const int ASCII_LIMIT = 127; // highest ASCII character value being tracked
                               // 1 must be added when we need the COUNT of
                               // how many characters we'll be tracking
 
// strings
const char TXTEXT[NAME_L]  = ".txt";
const char HELP_PROMPT[NAME_L] = "/?";

//============================================================================//
//= Global Function Prototypes ===============================================//
//============================================================================//

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: checkArgs
// Summary:       checks the count and values of the cmd line arguments to
//                ensure that they will be valid for the program to work with
// Parameters:    int argc      the number of cmd line arguments entered by the 
//                              user
//                char* argv[]  an array of char* that point to the values of
//                              the arguments stored as C-strings
// Returns: void       
//
//////////////////////////////////////////////////////////////////////////////// 
void checkArgs( int argc, char* argv[] );

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: dispHelp 
// Summary:       displays the help screen, displays error appropriate message
//                based on error code passed, exits program afterward
// Parameters:    int code   a non-zero code corresponding to the reason the
//                           help screen is being displayed 
// Returns:       void
//
////////////////////////////////////////////////////////////////////////////////
void dispHelp( int code );

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: processArgs
// Summary:       assigns the values from the cmd line arguments to meaningful
//                variables for program function
// Parameters:    char* argv[]        the array of char* that point to the cmd
//                                    line arguments in C-string form
//                char* textF         a pointer to the name of the text file to
//                                    be worked on
//                char* summaryF      a pointer to the name of the summary file
//                                    to be written
//                char &targetCh      the character that will be replaced in
//                                    the text file
//                char &replacerCh    the character that will be used as the
//                                    replacement in the test file
// Returns:       void
//
////////////////////////////////////////////////////////////////////////////////
void processArgs( char* argv[], char* &textF, char* &summaryF, char &targetCh, char &replacerCh );

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: findLength
// Summary:       finds the number of lines in the text file by counting '\n' 
//                characters
// Parameters:    char* textF   a pointer to the name of the text file to be
//                              worked on
// Returns:       int   the length of the file
//
////////////////////////////////////////////////////////////////////////////////
int findLength( char* textF );

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: sizeUpFile
// Summary:       retrieves the counts of ASCII values of the characters of
//                interest, stores them in an array, and outputs the results to
//                a summary file
// Parameters:    char* textF      a pointer to the name of the text file to be
//                                 worked on
//                char* summaryF   a pointer to the name of the summary file to
//                                 be written
//                int fileLen      the length of the text file
// Returns:       void
//
////////////////////////////////////////////////////////////////////////////////
void sizeUpFile( char* textF, char* summaryF, int fileLen );

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: replaceChar
// Summary:       reads in the contents of a text file to a char** array, one 
//                char at a time, replaces the target character with the 
//                replacement one, appends the results to the text file, and 
//                outputs the counts of the character ASCII vales
//                (replacements included) to a summary file
// Parameters:    char* textF         a pointer to the name of the text file to
//                                    be worked on
//                char* summaryF      a pointer to the name of the summary file
//                                    to be written
//                int fileLen         the length of the supplied text file
//                char targetChar     the character to be replaced in the text
//                                    file
//                char replacerChar   the character that will replace
//                                    targetChar
// Returns:       void
//
////////////////////////////////////////////////////////////////////////////////
void replaceChar( char* textF, char* summaryF, int fileLen, char targetChar, char replacerChar );

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: summarize
// Summary:       outputs the counts of the ASCII values of characters
//                contained in the text file that was processed, capaple of 
//                generating an original file or appending to an existing one
// Parameters:    char* summaryF    a pointer to the name of the summary file
//                                  to be written
//                int* counts       a pointer designating an array of the
//                                  counts of the ASCII values
//                bool appendMode   indicates whether to create an original
//                                  file or append to an existing one
// Returns:       void
//
////////////////////////////////////////////////////////////////////////////////
void summarize( char* summaryF, int* counts, bool appendMode );


//============================================================================//
//= Main Program Definition ==================================================//
//============================================================================//
int main( int argc, char* argv[] )
   { 
   // vars
     // prog arguments
     char* textF = 0;
     char* summaryF = 0;
     char targetCh = D_SMILE;
     char replacerCh = BELL;
 
     // integers
     int fileLen = 0;

   // process cmdline args
     // check the number and value of args
     checkArgs(argc, argv);     

     // process the arguments
     processArgs(argv, textF, summaryF, targetCh, replacerCh);

   // determine file length, gather summary data
   fileLen = findLength(textF);
   sizeUpFile(textF, summaryF, fileLen);   

   // perform replacements, gather new summary data
   replaceChar(textF, summaryF, fileLen, targetCh, replacerCh); 

   // end prog
   return 0;
   }

//============================================================================//
//= Supporting function implementations ======================================//
//============================================================================//

void checkArgs( int argc, char* argv[] )
   {
   // vars
   int targetCh = int('!');
   int replacerCh = int('!');

   // ensure that the number of arguments is appropriate
   if((argc < 2) || (argc > 5))
     {
     dispHelp(HELP_NUMARGS);
     }

   // see if the user wants help
   else if(strcmp(argv[1], HELP_PROMPT) == 0)
     {
     dispHelp(HELP_REQ);
     }

   // now check that all "4" arguments are present
   if(argc != 5)
     {
     dispHelp(HELP_NUMARGS);
     }

   // ensure args 4 and 5 are in the standard ASCII range (0-ASCII_LIMIT) but not 
   // the NULL or endline chars (0, 10) to not disrupt program function
   targetCh = atoi(argv[3]);
   replacerCh = atoi(argv[4]); 
   if( ((targetCh < 0)   || (targetCh > ASCII_LIMIT)) ||
       ((replacerCh < 0) || (replacerCh > ASCII_LIMIT)) )
     {
     dispHelp(HELP_ASCIICODE);
     }
   else if( ((targetCh == 0)   || (targetCh == 10)) ||
            ((replacerCh == 0) || (replacerCh == 10)) )
     {
     dispHelp(HELP_ASCIICODE);
     }

   // if all if statements evaluate to false, arguments are acceptable
   // and program proceeds
   
   // no return - void
   }

void dispHelp( int code )
   {
 
   #if TEXT_USER_ACTION
   clrScr();

   cout << "               WELCOME TO THE HELP SCREEN" << endl
        << endl << endl;

   // display error appropriate message
   switch( code )
     {
     case HELP_REQ:
       {
       cout << "Here is the information you requested:"
            << endl << endl;
       }
       break;
     case HELP_NUMARGS:
       {
       cout << "You have entered an invalid number of command line arguments."
            << endl
            << "Please read the information below and try again."
            << endl << endl;
       }
       break;
     case HELP_ASCIICODE:
       {
       cout << "You have entered at least one invalid ASCII number code."
            << endl
            << "Codes must be within the acceptable range, and not invalid"
            << endl
            << "options."
            << endl << endl;
       }
       break;
     case HELP_INPUT_FORMAT:
       {
       cout << "The text file you have chosen does not have valid dimensions."
            << endl
            << "Please use one with the proper format."
            << endl << endl;
       }
       break;
     case NOT_ENOUGH_MEM:
       {
       cout << "The program could not access to enough memory to"
            << endl
            << "perform its function. Try a smaller file, or"
            << endl
            << "split this file into segments."
            << endl << endl;
       }
       break;
     }

   // display help information
   cout << "This program gathers counts of the ASCII characters contained"
        << endl
        << "in the provided text I/O file, and outputs these to a summary"
        << endl
        << "file. The program then replaces the specified character with" 
        << endl
        << "the specified replacement character, and appends the new counts"
        << endl
        << "to the summary file."
        << endl << endl
        << "- The I/O file must be less than " << MAX_LINES << " lines long"
        << endl
        << "  and no line may be longer than " << (MAX_LINE_L) << " characters."
        << endl << endl
        << "- ASCII values must be between 0 and " << ASCII_LIMIT << "."
        << endl
        << "  ASCII codes may not be 0 or 10 to prevent operation errors."
        << endl << endl 
        << "- The program requires 4 (5 including program name) cmd line" 
        << endl
        << "  arguments.";

   // notify user program will close
   cout << "Program will now end.";

   // hold the screen
   holdProg();
   #endif

   // no return, but exit prog
   exit(code);
   }

void processArgs( char* argv[], char* &textF, char* &summaryF, char &targetCh, char &replacerCh )
   {

   // store text file name (this is an improvement yet to be made)
   textF = argv[1];
     
   // store summary file name (this is an improvement yet to be made)
   summaryF = argv[2];

   // store target character
   targetCh = char(atoi(argv[3]));

   // store the replacement charachter to be used
   replacerCh = char(atoi(argv[4]));

   // no return, void
   }

int findLength( char* textF )
   {
   // vars
   ifstream fin;
   int fileLen = 0;
   int numOnLine = 0;
   char dummy = 'a';

   // clear and open
   fin.clear();
   fin.open(textF);
  
   // grab each char one at a time to find the number of lines in the file
   dummy = fin.get();  // prime

   while(fin.good())
     {
     numOnLine ++;
     if(numOnLine == (MAX_LINE_L - 1))
       {
       dispHelp(HELP_INPUT_FORMAT);
       }

     if(dummy == '\n')
       {
       fileLen ++;
       numOnLine = 0;
       if(fileLen >= MAX_LINES)
         {
         dispHelp(HELP_INPUT_FORMAT);
         }
      
       numOnLine = 0;
       }

     dummy = fin.get();
     }

   // return file length
   return fileLen;
   }

void sizeUpFile( char* textF, char* summaryF, int fileLen )
   {
   // vars
   ifstream fin;
   char dummy = 'a';
   int code = 0;       
   int lineCount = 0;     
   bool appendMode = false;

     // dynamic memory 
     int* counts = new int [ASCII_LIMIT + 1];
       if(counts == 0)
         {
         dispHelp(NOT_ENOUGH_MEM);
         }
       memset(counts, 0, ((ASCII_LIMIT + 1) * sizeof(int)));      

   // clear and open
   fin.clear();
   fin.open(textF);

   // read in file contents, track the counts of each char 
   dummy = fin.get();   // prime

   while(fin.good() && lineCount < fileLen)
     {
     code = int(dummy);
     (*(counts + code)) ++;

     // grab next char
     dummy = fin.get();

     // if necessary, increment line count
     if(dummy == '\n')
       {
       lineCount ++;
       }
     }

   // close
   fin.close();

   // write original summary
   summarize(summaryF, counts, appendMode);

   // return dynamic memory
   delete [] counts;

   // void - no return
   }

void replaceChar( char* textF, char* summaryF, int fileLen, char targetChar, char replacerChar )
   {
   // vars
   fstream fstreem;
   int rndx = 0;
   int cndx = 0;
   int code = 0;
   char dummy = 'a';
   bool appendMode = true;
     // dynamic memory 
     int* counts = new int [ASCII_LIMIT + 1];
       if(counts == 0)
         {
         dispHelp(NOT_ENOUGH_MEM);
         }
       memset(counts, 0, ((ASCII_LIMIT + 1) * sizeof(int)));
     char** contents = new char* [fileLen];
       for(rndx = 0; rndx < fileLen; rndx ++)
         {
         char* p = new char [MAX_LINE_L];   // temporary dummyPtr for mem alloc
           if(p == 0)
             {
             dispHelp(NOT_ENOUGH_MEM);
             }
           memset(p, '\0', (MAX_LINE_L * sizeof(char)));

         *(contents + rndx) = p;
         }         

   // clear and open
   fstreem.clear();
   fstreem.open(textF, fstream::in | fstream::out);

   // read in file contents, replacing the appropriate chars and tracking counts
   rndx = 0, cndx = 0;
   dummy = fstreem.get();   // prime

   while(fstreem.good())
     {
     // check the char for replacement case     
     if(dummy == targetChar)
       {
       dummy = replacerChar;
       }

     // store the char and track counts 
     (*(*(contents + rndx) + cndx)) = dummy;

     code = int(dummy);
     (*(counts + code)) ++;     

     // update memory position now that char has been stored
     cndx ++;

     if(dummy == '\n')
       {
       // null char need not be manually added, already there by memset
       rndx ++;
       cndx = 0;
       }
     
     dummy = fstreem.get();
     }

   // ouptut text with replacements
   fstreem.clear();   // clear the stream before switch to write mode

   fstreem << endl << endl;   // make space

   for(rndx = 0; rndx < fileLen; rndx ++)
     {
     fstreem << *(contents + rndx);
     }

   fstreem << endl;   // bump EOF down a little to prevent errors

   // close
   fstreem.close(); 
 
   // create and append the summary   
   summarize(summaryF, counts, appendMode);

   // return dynamic memory
   delete [] counts;

   for(rndx = 0; rndx < fileLen; rndx ++)
     {
     delete [] *(contents + rndx);
     } 
   delete [] contents;
 
   // void - no return
   }

void summarize( char* summaryF, int* counts, bool appendMode )
   {
   // vars
   ofstream fout;
   int code = BELL;
   int numDigits = 0;
   int numLower  = 0;
   int numNonPrint = 0;
   int numPrintable = 0;
   int numPunct = 0;
   int numUpper  = 0;

   // count up the tallies of interest
   for(code = 0; code <= ASCII_LIMIT; code ++)
     {
     // printable chars
     if( isprint(code) )
       {
       // capture this tally, then the others
       numPrintable += (*(counts + code));

       // letters
       if( isalpha(code) )
         {
         // lower
         if( islower(code) )
           {
           numLower += (*(counts + code));    
           }

         // upper
         else if( isupper(code) )
           {
           numUpper += (*(counts + code));
           }
         }

       //digits
       else if( isdigit(code) )
         {
         numDigits += (*(counts + code));
         }
 
       // punctuation
       else if( ispunct(code) )
         {
         numPunct += (*(counts + code));
         }
       }
     else // char is non printable
       {
       numNonPrint += (*(counts + code));
       } 
     }

   // clear and open
   fout.clear();

   if(appendMode)
     {
     fout.open(summaryF, fstream::app);   // sets cursor to the end of the file
     }
   else
     {
     fout.open(summaryF); 
     }

   // write header
   if(appendMode)
     {
     fout << endl << "SUMMARY FOR APPENDED FILE" << endl;     
     }
   else
     {
     fout << "SUMMARY FOR ORIGINAL FILE" << endl << endl;
     }   

   // write file
     // major counts
     fout << "Number of digits:                   " << numDigits << endl
          << "Number of lower case characters:    " << numLower << endl
          << "Number of nonprintable characters:  " << numNonPrint << endl
          << "Number of printable characters:     " << numPrintable << endl
          << "Number of punctuation characters:   " << numPunct << endl
          << "Number of upper case characters:    " << numUpper << endl;

    // minor counts
    for(code = 0; code <= 9; code ++)
       {
       fout << "Count for ASCII code   " 
                            << code << ":            " << *(counts + code) 
            << endl;  
       }   
    for(code = 10; code <=99 ; code ++)
       {
       fout << "Count for ASCII code   " 
                             << code << ":           " << *(counts + code) 
            << endl;  
       }
    for(code = 100; code <= ASCII_LIMIT; code ++)
       {
       fout << "Count for ASCII code   " 
                             << code <<  ":          " << *(counts + code) 
            << endl;  
       }

     // add space
     fout << endl;

   // close
   fout.close();

   // no return - void   
   }

