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
   string inFileName;
   string outFileName;
   ifstream fin;
   ofstream fout;

   // print title
      // function: printTitle
      printTitle();

// Prompt user for input and output file names
   // function: getFileNames
   getFileNames( inFileName, outFileName );

// display error if input is not found
   // operations: if, else, function: isFileThere
   if( !isFileThere( inFileName ) )
     {
      // display error message
         // function: cout
         cout << endl
              << "ERROR:"
              << "File \""
              << inFileName
              << "\" Not Found"
              << " - "
              << "Program Aborted"
              << endl << endl << endl << endl;
      
      // hold system for user
         // function: system
         system( "PAUSE" );
 
      // terminate program, return 0
      return 0;
     }    

   // otherwise, if file name is valid, continue to process

// process (decode) file if input file is found

   // notify user that file is being processed
      // function: cout
      cout << "Processing Data . . ."
           << endl << endl << endl;

   // process data

      // open input file, create output file
         // function: openFiles
         openFiles( inFileName, fin, outFileName, fout);

      // collect and output values 
         // function: decodeData
         decodeData( inFileName, fin, outFileName, fout );

     // close files
         // function: closeFiles
         closeFiles( inFileName, fin, outFileName, fout);

   // notify user when processing is complete
      // function: cout
      cout << " . . . Data processing completed."
           << endl << endl << endl << endl;

// end program

   // hold system for user
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


void getFileNames(string &inFile, string &outFile )
   {
    // prompt for input file name
       // functions: cout, cin

    // prompt for output file name
       // functions: cout, cin

    // generate space for next operation
       // function: cout

    // no return - void
   }


bool isFileThere( const string &name )
   {
    // declare variables

    // clear and open file 

    // test the file to see if it opens 

       // close the file and return true

    // otherwise, return false
    return 0;  // temporary stub return
   }


void openFiles( const string &inFileNam, ifstream &inFile, const string &outFileNam, ofstream &outFile )
   {
    // clear ifstream object and open file

    // clear and open output file 

    // no return - void
   }

 
void closeFiles( const string &inFileNam, ifstream &inFile, const string &outFileNam, ofstream &outFile )
   {
    // close the input file

    // close the output file

    // no return - void
   }


void decodeData( const string &inFileNam, ifstream &inFile, const string &outFileNm, ofstream &outFile ) 
   {
    // initialize variable(s)

    // extract, decode, and export data through use of looping. loop terminates
    // when values can no longer be extracted
       // operation: do...while

          // use a while loop to count spaces to indicate when ten words have been printed
          // as other functions decode and output all characters. the loop will resest 
          // itself at approprate times
             // operation: while
                 // be sure to prime while loop

                // collect even values from text file
                   // function: findEven

                // check for end of file sentinel

                      // break the spaceCount loop
              
                // convert extracted values from text file to characters
                   // finction: decodeChar

                // ignore endline character
                   // operation: if

                      // turn it into a space

                // output characters
                   // operation: ofstream

                // increment counter when space is identified
                   // operation: if

          // reset space counter to begin new line

    // no return - void
   }


char decodeChar( int value )
   {
    // initialize variables
    
    // convert extracted integer to one corresponding to a char
       // operation: math
    
    // cast charNum as a char
       // function: char
 
    // return decoded char
    return 'a'; // temporary stub return 
   }


int findEven( const string &inFileNm, ifstream &inputFile )
   {
    // initialize variables

    // use a while loop to skip odd integers and extract even ones
       // be sure to prime the loop

    // check the extraction 

      // indicate the end of the file to flag that no more outputting need be done

    // use loop to skip odd integers and extract next integer and stop when an even one is found  

       // if the value is odd

          // retrieve next value

          // return the good value
     
    // return the number of the character signifying the end of the document
    return 0; //temporary stub return
   }

