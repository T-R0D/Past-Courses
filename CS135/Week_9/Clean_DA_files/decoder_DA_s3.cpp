////////////////////////////////////////////////////////////////////////////////
// Header Files ////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
using namespace std;

////////////////////////////////////////////////////////////////////////////////
// Global Constant Definitions /////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// character constants
const char SPACE   = ' '  ;
const char NEWLINE = '\n' ;

// numerical constants
const int END_DOC_SENTINEL = 11 ; // this is the number of an ASCII character I 
                                  // know will not appear in the encoded document 

// string constants
const string TITLE = "     DECODER PROGRAM" ;
const string UNDER = "     ===============" ;

////////////////////////////////////////////////////////////////////////////////
// Global Function Prototypes //////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/* 
Name: printTitle
Process: prints program title with underline
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: displays title
Dependencies: <iostream>
*/
void printTitle();


/* 
Name: getFileNames
Process: prompts user for an input file name and an output file
         name and passes them to the program by reference
Function Input/Parameters: string &inFile     the name of the input file to be used
                           string &outFile    the name of the output file to be created
Function Output/Parameters: none
Function Output/Returned:  (by reference) string &inFile     see above
                           (by reference) string &outFile    see above
Device Input: input file name strings through keyboard
Device Output: none 
Dependencies: <iostream>
*/
void getFileNames(string &inFile, string &outFile );


/* 
Name: isFileThere
Process: checks to see if an input file exists, returns a Boolean 
         value indicating the result
Function Input/Parameters: const string &name     the name of the input file to be used
Function Output/Parameters:  bool    indicates the status of the input name check
Function Output/Returned: true     indicates the input file name is a valid one
                          false    indicates the input file name is invalid
Device Input: none
Device Output: none
Dependencies: <ifstream>
*/
bool isFileThere( const string &name );


/* 
Name: openFiles
Process: opens the input and output files to be used by the program
Function Input/Parameters: const string &inFileNam      the name of the input file
                                                        used by the program
                           ifstream &inFile             the fstream object used to
                                                        carry out actions related
                                                        to input data
                           const string &outFileNam     the name of the output file
                                                        created by the program
                           ofstream &outFile            the fstream object used to
                                                        carry out actions related
                                                        to output data
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none 
Dependencies: <ifstream>
*/
void openFiles( const string &inFileNam, ifstream &inFile, const string &outFileNam, ofstream &outFile );
 

/* 
Name: closeFiles
Process: closes the files that were in use by the program
Function Input/Parameters: const string &inFileNam      the name of the input file
                                                        used by the program
                           ifstream &inFile             the fstream object used to
                                                        carry out actions related
                                                        to input data
                           const string &outFileNam     the name of the output file
                                                        created by the program
                           ofstream &outFile            the fstream object used to
                                                        carry out actions related
                                                        to output data
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: <ifstream>
*/
void closeFiles( const string &inFileNam, ifstream &inFile, const string &outFileNam, ofstream &outFile );


/* 
Name: decodeData
Process: decodes a text file filled with numbers using the agreed upon encoding
         format of odd integers being meaningless and the most significant 2 or 3
         digits of an even integer corresponding to ASCII values, and outputs the 
         decoded text to a new file, with ten words per line
Function Input/Parameters: const string &inFileNam      the name of the input file
                                                        used by the program
                           ifstream &inFile             the fstream object used to
                                                        carry out actions related
                                                        to input data
                           const string &outFileNam     the name of the output file
                                                        created by the program
                           ofstream &outFile            the fstream object used to
                                                        carry out actions related
                                                        to output data
Function Output/Parameters: none
Function Output/Returned: a decoded body of text to an output file
Device Input: none
Device Output: none
Dependencies: <ifstream>, decodeChar, findEven
*/
void decodeData( const string &inFileNam, ifstream &inFile, const string &outFileNm, ofstream &outFile ); 
  

/* 
Name: decodeChar
Process: converts input integer values into ASCII characters
Function Input/Parameters: int value     the value to be converted to an ASCII
                                         character
Function Output/Parameters: char     the character that has been decoded
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: findEven
*/
char decodeChar( int value );


/* 
Name: findEven
Process: skips odd integer values in a body of encoded text, and returns the 
         next even value found, returns an integer corresponding to an unexpected
         ASCII character to indicate the end of the file
Function Input/Parameters: const string &inFileNam      the name of the input file
                                                        used by the program
                           ifstream &inFile             the fstream object used to
                                                        carry out actions related
                                                        to input data     
Function Output/Parameters: an integer that will later be converted to an ASCII 
                            character
                                or
                            an integer corresponding to an unexpected character,
                            indicating the end of a file
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: <ifstream>
*/
int findEven( const string &inFileNam, ifstream &inFile );


////////////////////////////////////////////////////////////////////////////////
// Main Program Definition /////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int main()
   { 

// Initialize program
   
   // initialize variables

   // print title
      // function: printTitle

// Prompt user for input and output file names
   // function: getFileNames

// display error if input is not found
   // operations: if, else, function: isFileThere

      // display error message
         // function: cout

      // hold system for user
         // function: system
 
      // terminate program, return 0

   // otherwise, if file name is valid, continue to process

// process (decode) file if input file is found

   // notify user that file is being processed
      // function: cout

   // process data

      // open input file, create output file
         // function: openFiles

      // collect and output values 
         // function: decodeData

     // close files
         // function: closeFiles

   // notify user when processing is complete
      // function: cout

// end program

   // hold system for user
      // function: system

   // return 0
   return 0;
   }


////////////////////////////////////////////////////////////////////////////////
// Supporting function implementations /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void printTitle()
   {
    // no return - void
   }


void getFileNames(string &inFile, string &outFile )
   {
    // no return - void
   }


bool isFileThere( const string &name )
   {
    return 0;  // temporary stub return
   }


void openFiles( const string &inFileNam, ifstream &inFile, const string &outFileNam, ofstream &outFile )
   {
    // no return - void
   }

 
void closeFiles( const string &inFileNam, ifstream &inFile, const string &outFileNam, ofstream &outFile )
   {
    // no return - void
   }


void decodeData( const string &inFileNam, ifstream &inFile, const string &outFileNm, ofstream &outFile ) 
   {
    // no return - void
   }


char decodeChar( int value )
   {
    return 'a'; // temporary stub return 
   }


int findEven( const string &inFileNm, ifstream &inputFile )
   {
    return 0; //temporary stub return
   }

