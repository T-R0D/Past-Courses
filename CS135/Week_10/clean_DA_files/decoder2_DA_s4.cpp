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

// string constants
const string TITLE = "     DECODER PROGRAM" ;
const string UNDER = "     ===============" ;

// array related constants
const int MAX_SLOTS = 5000 ;


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
void getFileNames( string &inFileNam, string &outFileNam );


/* 
Name: processInputFile
Process: notifies user that "Uploading" has begun,
         clears and opens the input file, tests its validity, reads the values
         the input file contains into an array, and closes the file. displays an
         error message and indicates to the main function that the program has 
         failed if the input file is invalid.
Function Input/Parameters: const string &inFileNam     the given name of the input file

                           ifstream &inFile            the name of the ifstream object
                                                       repsonsible for managing the 
                                                       input file

                           int array[]                 the array that will be responsible
                                                       for managing the encoded values

                           int &itemCount              a count used to indicate how many
                                                       non-garbage items exist in the 
                                                       array
Function Output/Parameters: int array[]        (by reference)
                            int &itemCount     (by reference)
Function Output/Returned: bool true      indicates to main program should be terminated
                               false     indicates to main that the program can continue
Device Input: none
Device Output: Notifies user "Uploading" has begun
Dependencies: <iostream>, <fstream>, isFileThere, printErrorMessage
*/
bool processInputFile( const string &inFileNam, ifstream &inFile, int array[], int &itemCount );


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
Dependencies: <fstream>
*/
bool isFileThere( const string &name );


/* 
Name: printErrorMessage
Process: displays an error message if the input file is invalid and holds the display
         for the user
Function Input/Parameters: const string &inFileNam     the given input file name
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: displays error message and holds screen
Dependencies: <iostream>
*/
void printErrorMessage( const string &inFileNam );


/* 
Name: decode data
Process: notifies user that decryption has begun, accepts an integer matrix, finds the even 
         integers and places them at the beginning of the array while converting them to 
         ASCII values, then uses those ASCII codes to place decoded characters in a 
         character array
Function Input/Parameters: int intArray[]       the array that will contain the encoded values 
                                                read in from the input file

                           char charArray[]     the array that will store the decoded character
                                                values

                           int &codeCount       a count used to indicate how many useful values 
                                                are contained in the integer array (doesn't 
                                                differentiate between even and odd values)

                           int decodeIndex      used to indicate the positions of useful 
                                                values in both the integer and character
                                                arrays (primarily used to indicate the 
                                                last position) 
Function Output/Parameters: none
Function Output/Returned: int intArray[]       (by reference)
                          char charArray[]     (by reference)
Device Input: none
Device Output: "Decrypting" message
Dependencies: <iostream>
*/
void decodeData( int intArray[], char charArray[], int &codeCount, int &decodeIndex );


/* 
Name: findEven
Process: searches an integer array for an even value and returns it using an idex value
         passed back and forth between this function and the calling function
Function Input/Parameters: int array[]            an array containing integer values

                           int &indexPosition     a number indicating where in the array
                                                  to begin reading, as well as where next
                                                  to read
Function Output/Parameters: int an even integer
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
*/
int findEven( int array[], int &indexPosition );


/* 
Name: outputData
Process: notifies user that "Downloading" has begun, clears the ofstream object and opens/creates 
         the output file, writes the decoded data to the file, and closes the output file 
Function Input/Parameters: const string &outFileNam     the given name for the output file
 
                           ofstream &outfile            the ofstream object used to manage the
                                                        output file

                           char array[]                 the array where all of the decoded character
                                                        values are stored
      
                           int decodeIndex             an index used to indicate the last position of 
                                                        a non-garbage character in the array
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: "Decrypting" message
Dependencies: <iostream>, <fstream>, writeData
*/
void outputData( const string &outFileNam, ofstream &outFile, char array[], int decodeIndex );


/* 
Name: writeData
Process: writes the decoded data to the output file, mangaing spaces so that 10 words appear 
         on each line, unless an endline character is encountered
Function Input/Parameters: ofstream &outFile     the ofstream object that manages the output
                                                 file

                           char array[]          the array containing the decoded characters 

                           int decodeIndex       an indicator of the last non-garbage containing
                                                 position in the character array
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: <fstream>
*/
void writeData( ofstream &outFile, char array[], int decodeIndex );


////////////////////////////////////////////////////////////////////////////////
// Main Program Definition /////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int main()
   { 
// initialize Program

   // initialize variables

      // file names
      string inFileName  = "NONE ENTERED" ;
      string outFileName = "NONE ENTERED" ;
      
      // fstream objects
      ifstream fin;
      ofstream fout;

      // arrays and related variables
      int integerArray[ MAX_SLOTS ] ;
      char charArray[ MAX_SLOTS ] ;

      // boolean variables
      bool terminate_program = false ;

      // counters
      int intCount = 0;
      int charCount = 0;

   // print program title
      // function: printTitle
      printTitle();

// prompt for file names
   // function: getFileNames
   getFileNames( inFileName, outFileName );

// process input file
   // function: processInputFile
    terminate_program = processInputFile( inFileName, fin, integerArray, intCount );

   // input file can't be processed, terminate the program
      // operation: if
      if( terminate_program )
        {
         // error message already displayed by processInputFile
         
         // return 0 to terminate program
         return 0;
        }

// decode data
   // function: decodedData
   decodeData( integerArray, charArray, intCount, charCount );

// output data
   // function: outputData
   outputData( outFileName, fout, charArray, charCount );

// end program

   // hold program for user
      // function: system
      system( "PAUSE" );

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


void getFileNames( string &inFileNam, string &outFileNam )
   {
    // no return - void
   }


bool processInputFile( const string &inFileNam, ifstream &inFile, int array[], int &itemCount )
   {
    return 0;   // temporary stub return
   }


bool isFileThere( const string &inFileNam )
   {
    return 0;  // temporary stub return
   }


void printErrorMessage( const string &inFileNam )
   {
    // no return - void
   }


void decodeData( int intArray[], char charArray[], int &codeCount, int &decodeIndex )
   {
    // no return - void
   }


int findEven( int array[], int &indexPosition )
   {
    return 0;   // temporary stub return
   }


void outputData( const string &outFileNam, ofstream &outFile, char array[], int decodeIndex )
   {
    // no return - void
   }

void writeData( ofstream &outFile, char array[], int decodeIndex )
   {
    // no return - void
   } 
