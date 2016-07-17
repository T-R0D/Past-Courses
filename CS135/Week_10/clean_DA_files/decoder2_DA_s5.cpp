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
    // print title

    // underline title

    // no return - void
   }


void getFileNames( string &inFileNam, string &outFileNam )
   {
    // prompt for input file name
       // functions: cout, cin

    // prompt for output file name
       // functions: cout, cin

    // no return - void
   }


bool processInputFile( const string &inFileNam, ifstream &inFile, int array[], int &itemCount )
   {
    // initialize variables

    // notify user data retreival is being attempted
       // function: cout

    // open input file
       // operations: .clear, .open

    // check validity of input file 
       // if input file name is not valid, print error message and terminate program
          // operation: if, function: isFileThere

             // display error message
                // function: printErrorMessage
 
             // terminate program by returning true

        // otherwise, program continues    

    // extract data and place it in the array using loop, once end of file is
    // detected, end extraction
       // operations: for, .good

    // store the count of items in array for later

    // close input file
       // operation: .close
    
    return 0;   // temporary stub return
   }


bool isFileThere( const string &inFileNam )
   {
    // declare variables/objects

    // clear fstram object and open input file
       // operations: .clear, .open

    // if opening was successful
       // operations: if, .good

          // close the file
             // operation: .close

          // file was opened successfully, indicate by returning true
   
    // otherwise, assume file failed to open,return false
    return 0;  // temporary stub return
   }


void printErrorMessage( const string &inFileNam )
   {
    // print message
       // function cout
      
    // hold system for user
       // function: system

    // no return - void
   }


void decodeData( int intArray[], char charArray[], int &codeCount, int &decodeIndex )
   {
    // initialize variables

    // declare to the user decryption is beginning
      // function: cout

    // find even numbers, convert them to ASCII code numbers, leave invalid numbers at end of array

       // find even value, convert it to an ASCII value, and store it in the
       // array, staring in the 0 position
          // operation: while

             // convert even values to 3 digit numbers

             // increment counters

             // loop breaks when array has been read up to last code value
  

    // convert ASCII values into characters
       // operation: for
          
          // loop breaks when index reaches the position with the last entry to be decoded

    // no return - void
   }


int findEven( int array[], int &indexPosition )
   {
    // initialize variables

    // capture first value to test the file and prime the loop

    // initiate loop if the retrieved value is odd, terminate when even value is found
       // operation: while

          // move to next index position

          // retrieve next value

    // return the even value
    return 0;   // temporary stub return
   }


void outputData( const string &outFileNam, ofstream &outFile, char array[], int decodeIndex )
   {
    // notify user data is being outputted
       // function: cout

    // clear ofstream object and open/create output file
       // operations: .clear, .open

    // write data to file, starting a new line after every 10 spaces
       // operation: writeData    

    // close the file
       // operation: .close

    // no return - void
   }

void writeData( ofstream &outFile, char array[], int decodeIndex )
   {
    // initialize variables

    // write data to output file while accounting for words per line
       // operation for

          // retreive first value
          
          // if value found is a space, increment space counter

          // when space counter reaches 10, start new line

             // start new line
     
             // reset counter

          // when new line char is encountered, allow it to print, reset space counter

    // no return - void
   } 
