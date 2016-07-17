/*//////////////////////////////////////////////////////////////////////////
  
    Formatted Command-Line I/O header file.

    Copyright (c) 2008 - 2012 by Michael E. Leverington

    Code is free for use in whole or in part and may be modified for other
    use as long as the above copyright statement is included.
    
    Code Written:        08/18/2008
    Most Recent Update:  01/21/2011 - 5:30 p.m.
    Date Due for Review: 01/21/2011

/////////////////////////////////////////////////////////////////////////*/
/*
    INSTRUCTIONS:

    1) This source file contains the functions you may use for input and
       output of assignments when you need to use formatted command-line I/O. 
       You only need to include this file as a header file in your program. 
       As long as you don't change this file, you can keep using it to develop 
       other programs. It should be noted that header files usually only
       contain function headers, and do not normally contain function 
       implementations. However this file is set up to make it easy for
       you to get started learning C++ programming.
       
    2) For any of the "promptFor..." functions, just place some prompting
       text into the function and call it. The user will be prompted for
       the appropriate value, and the user's response will be assigned
       to the variable you use.

       Example:        userAge = promptForInt( "Enter your age: " );
       Result Displayed:       Enter your age: {user answers here}

    3) For any of the "print..." functions, you need to provide the following:
       - value to be output
       - the text block size required
       - the justification
       - in the case of floating point values, the precision

       Example 1:    printString( "This is a string", 20, "CENTER" );
       Result Displayed:          |  This is a string  |
       Explanation: "This is a string" is displayed centered in a block
                    that is 20 characters wide

       Example 2:   printDouble( 25.45678, 2, 10, "RIGHT" );
       Result Displayed:         |     25.46|
       Explanation: The value 25.45678 is displayed right justified within
                    a block of 10 characters, and with 2 digits to the right
                    of the decimal point (called precision for purposes of
                    this function)

    4) You are provided simple information for all of the functions in a
       standardized format. You will be using this format for your own
       functions in the near future.

    END OF INSTRUCTIONS
*/

#ifndef STANDARD_IO_H
#define STANDARD_IO_H

//  Header Files  /////////////////////////////////////////////////////////
//
    #include <iostream>	// for console I/O
    #include <iomanip>	// for output formatting

    using namespace std;
//
//  Global Constant Definitions  //////////////////////////////////////////
//
    // NONE
//
//  Global Function Prototypes  ///////////////////////////////////////////
//
/*
name: promptForChar
process: prompts user for character, then returns it
function input/parameters: text to prompt user (string)
function output/parameters: none
function output/returned: one character (char) is returned to calling function
device input/keyboard: user entry of a character
device output/monitor: prompt string displayed
dependencies: iostream I/O tools
*/
char promptForChar( const string &promptString );

/*
name: promptForInt
process: prompts user for integer, then returns it
function input/parameters: text to prompt user (string)
function output/parameters: none
function output/returned: one integer (int) value 
                          is returned to calling function
device input/keyboard: user entry of an integer value
device output/monitor: prompt string displayed
dependencies: iostream I/O tools
*/
int promptForInt( const string &promptString );

/*
name: promptForDouble
process: prompts user for double, then returns it
function input/parameters: text to prompt user (string)
function output/parameters: none
function output/returned: one floating point (double) value 
                          is returned to calling function
device input/keyboard: user entry of a floating point value
device output/monitor: prompt string displayed
dependencies: iostream I/O tools
*/
double promptForDouble( const string &promptString );

/*  
name: promptForString
process: prompts user for string, then returns it
function input/parameters: text to prompt user (string)
function output/parameters: none
function output/returned: one text (string) value 
                          is returned to calling function
device input/keyboard: user entry of a string
device output/monitor: prompt string displayed

dependencies: iostream I/O tools
*/
string promptForString( const string &promptString );

/*
name: promptForString
process: prompts user for string, then returns it
function input/parameters: text to prompt user (string)
function output/parameters: text c-style string (char [])
function output/returned: none
device input/keyboard: user entry of a string
device output/monitor: prompt string displayed
dependencies: iostream I/O tools
*/
void promptForString( const string &promptString, char resultString[] );

/*
name: printChar
process: the character is output to screen
function input/parameters: character (char) to be printed
function output/parameters: none
function output/returned: none
device input: none
device output/monitor: one character displayed
dependencies: iostream I/O tools
*/
void printChar( char charVal );

/*
name: printInt
process: the pre-spaces and post-spaces are calculated for the requested
         justification, and then the pre-spaces are printed, the integer value
         is printed, and then the post-spaces are printed
function input/parameters: intVal - the integer (int) value to be output
                           blockSize - the width of the block within 
                                       which to print the integer value
                           justify - either "LEFT", "RIGHT", or "CENTER" 
                                     to inform function 
                                     of the correct justification
function output/parameters: none
function output/returned: none
device input: none
device output/monitor: one integer value displayed, justified
dependencies: requires printString function
*/
void printInt( int intVal, int blockSize, const string &justify );

/*
name: printDouble
process: the precision is set for the double value, then the pre-spaces 
         and post-spaces are calculated for the requested justification,
         and then the pre-spaces are printed, the double value is printed, 
         and then the post-spaces are printed
function input/parameters: doubleVal - the floating point (double) value 
                                       to be output
                           blockSize - the width of the block within 
                                       which to print the integer value
                           precision - the number of digits to be printed 
                                       after the decimal point
                           justify - either "LEFT", "RIGHT", or "CENTER" 
                                     to inform function 
                                     of the correct justification
function output/parameters: none
function output/returned: none
device input: none
device output/monitor: one double value displayed, justified, with precision
dependencies: requires intToString and printString functions
*/
void printDouble( double doubleVal, int precision, int blockSize, const string &justify );

/*
name: printString
process: the pre-spaces and post-spaces are calculated for the requested
         justification, and then the pre-spaces are printed, the string value
         is printed, and then the post-spaces are printed
function input/parameters: stringVal - the text (string) value to be output
                           blockSize - the width of the block within 
                                       which to print the integer value
                           precision - the number of digits to be printed 
                                       after the decimal point
                           justify - either "LEFT", "RIGHT", or "CENTER" 
                                     to inform function 
                                     of the correct justification
function output/parameters: none
function output/returned: none
device input: none
device output/monitor: string is displayed justified within the given block
dependencies: requires printChars function
*/
void printString( const string &stringVal, int blockSize, const string &justify );

/*
name: printEndLines
process: repeatedly prints out endline constants as specified by numEndLines
function input/parameters: numEndLines - number (int) of endlines or vertical 
                                         spaces to print
function output/parameters: none
function output/returned: none
device input: none
device output/monitor: the given number of vertical spaces is displayed
dependencies: iostream I/O tools
*/
void printEndLines( int numEndLines );

/*
name: intToString (supporting function)
process: recursively adds individual digits to string (backwards)
function input/parameters: value - integer (int) value to be converted to string
function output/parameters: none
function output/returned: string form of integer value is returned 
                          to calling function
device input: none
device output: none
dependencies: none
*/
string intToString( int value );

/*
name: charInString (supporting function)
process: searches through given string and tests test character
         to see if it is in string; returns true if found, false if not
function input/parameters: testChar - a test character (char), and
                           testString - a string of characters to be tested
function output/parameters: none
function output/returned: returns true if test character is found to be
                          in given string
device input: none
device output: none
dependencies: none
*/
bool charInString( char testChar, const string &testString );

/*
name: printChars (supporting function)
process: repeatedly prints the given character for numCh times
function input/parameters: numCh - number (int) of characters to print
                           chOut - the character (char) to print
function output/parameters: none
function output/returned: none
device input: none
device output/monitor: the chOut character is displayed on the screen
                       for the numCh number of times
dependencies: iostream I/O tools
*/
void printChars( int numCh, char chOut );
//
//  Supporting Function Definition/Implementation /////////////////////////
//
char promptForChar( const string &promptString )
   {
    // initialize function
    char response;

    // output prompt string
    cout << promptString;

    // input user response
    cin >> response;

    // return user response
    return response;
   }

int promptForInt( const string &promptString )
   {
    // initialize function
    int response;

    // output prompt string
    cout << promptString;

    // input user response
    cin >> response;

    // return user response
    return response;
   }

double promptForDouble( const string &promptString )
   {
    // initialize function
    double response;

    // output prompt string
    cout << promptString;

    // input user response
    cin >> response;

    // return user response
    return response;
   }
  
string promptForString( const string &promptString )
   {
    // initialize function
    string response;

    // output prompt string
    cout << promptString;

    // input user response
    cin >> response;

    // return user response
    return response;
   }

void promptForString( const string &promptString, char resultString[] )
   {
    // initialize function

    // output prompt string
    cout << promptString;

    // input user response
    cin >> resultString;
   }

void printChar( char charVal )
   {
    // output character
    cout << charVal;
   }

void printInt( int intVal, int blockSize, const string &justify )
   {
    // initialize function
    const int MIN_STRING_SIZE = 15;
    char intString[ MIN_STRING_SIZE ];

    // convert integer to string format
    sprintf( intString, "%d", intVal );

    // print integer string with justification
    printString( intString, blockSize, justify );
   }

void printDouble( double doubleVal, int precision, int blockSize, const string &justify )
   {
    // initialize function
    const int MIN_STRING_SIZE = 15;
    char tempString[ MIN_STRING_SIZE ];
    string doubleString;

    // create floating point format string with precision
    doubleString = "%0.";
    doubleString += intToString( precision );
    doubleString += 'f';

    // convert floating point value to string (with precision)
    sprintf( tempString, doubleString.c_str(), doubleVal );

    // print floating point value with justification
    printString( tempString, blockSize, justify );
   }

void printString( const string &stringVal, int blockSize, const string &justify )
   {
    // initialize function
    const char SPACE = ' ';
    int preSpace = 0, postSpace = 0;
    int length = stringVal.length();

    // if right justification, add front-end spaces
    if( justify == "RIGHT" )
       {
        preSpace = blockSize - length;
       }

    // if center justification, add spaces on both ends
    else if( justify == "CENTER" )
       {
        preSpace = ( blockSize / 2 ) - ( length / 2 );
        postSpace = blockSize - preSpace - length;
       }

    // if left justification, add back-end spaces
    else // default if not "RIGHT" or "CENTER"
       {
        postSpace = blockSize - length;
       }

    // print front-end spaces, if any
    printChars( preSpace, SPACE );

    // print the string
    cout << stringVal;

    // print back-end spaces, if any
    printChars( postSpace, SPACE );
   }

void printEndLines( int numEndLines )
   {
    // if there are still endlines to print, print one,
    // and then call the function with one less
    if( numEndLines > 0 )
       {
        cout << endl;
        printEndLines( numEndLines - 1 );
       }
   }

string intToString( int value )
   {
    // if there are digits left, reduce the value by a factor of 10
    // and pick off the next digit
    // repeat until all digits are used
    if( value > 0 )
      {
       return intToString( value / 10 ) 
                + char( ( value % 10 ) + '0' );
      }

    return "";
   }

bool charInString( char testChar, const string &testString )
   {
    // initialize function
    int stringLength = testString.length();
    int index = 0;

    // search through string to find test character    
    while( index < stringLength )
       {
        // if character is in string, return true
        if( testChar == testString.at( index ) )
           {
            return true;
           }

        index++;
       }

    // if characater not in string, return false
    return false;
   }

void printChars( int numCh, char chOut )
   {
    // repeatedly output the character the given (numCh) number of times
    while( numCh > 0 )
       {
        cout << chOut;
        numCh--;
       }
   }
//


// End Standard I/O header file
#endif 


