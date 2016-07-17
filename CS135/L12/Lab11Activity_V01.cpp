//  Header Files  //////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>

using namespace std;

//  Global Constants  //////////////////////////////////////////////////////////
//
    const int MAX_STR_LEN = 80;

    const char COLON = ':';
    const char ENDLINE_CHAR = '\n';
    const char NULL_CHAR = '\0';
//
//  Function Prototypes  ///////////////////////////////////////////////////////
//
    void inputGetLine( ifstream &fin, char inStr[ MAX_STR_LEN ], 
                                                    int strLen, char stopChar );
    void inputIgnore( ifstream &fin, int strLen, char stopChar );
//
//  Main Function  /////////////////////////////////////////////////////////////
int main()
   {
    // initialize program / function
    
       // initialize variables
       ifstream inf;
       char fileName[ MAX_STR_LEN ] = "testfile.txt";
       char testStr[ MAX_STR_LEN ];
       int testInt;
       
    // open file
    inf.open( fileName );

    // check for good file
    if( inf.good() )
       {
        // get string up to colon, and display
        inf.getline( testStr, MAX_STR_LEN, COLON );
        cout << "Found string: " << testStr << endl << endl;

        // get integer value, and display
        inf >> testInt;
        cout << "Found int: " << testInt << endl << endl;

        // ignore action taken
inputGetLine( inf, testStr, MAX_STR_LEN, ENDLINE_CHAR );

        // get next line string, and display
        inf.getline( testStr, MAX_STR_LEN, ENDLINE_CHAR );
        cout << "Found string: " << testStr << endl << endl;
       }
    else
       {
        cout << "ERROR: File not found" << endl << endl;
       }

    // close file
    inf.close();

    // shut down program

       // hold program
       system( "pause" );

       // return success
       return 0;
   }

//  Supporting Function Implementation  ////////////////////////////////////////

void inputGetLine( ifstream &fin, char inStr[ MAX_STR_LEN ], 
                                                     int strLen, char stopChar )
   {
    // initialize function/variables
  
       // initialize index to zero
       int index = 0;

       // initialize stop length to one less than the string length parameter
       int stopLen = (strLen - 1);

       // other variable declaration - the input character
       char inputChar;

    // prime loop with first input cast as a character
       // function: .get
       inputChar = char( fin.get() );

    // start loop through num chars or until end character
    // test for file good AND index less than the stop length
    // AND the input character not equal to the stop character parameter
       // function: .good
       while( fin.good() && index < stopLen && inputChar != stopChar )
         {
        // place next character into the string array
          inStr[ index ] = inputChar;

        // update index
          index ++;

        // attempt to acquire next character, again cast as a character
           // function: .get
           inputChar = char( fin.get() );

    // end loop
          }
    // place a null character in the string array after the last input character
      inStr[ index ] = NULL_CHAR;
   }

void inputIgnore( ifstream &fin, int strLen, char stopChar )
   {
    // initialize function/variables
  
       // initialize index to zero
       int index = 0;

       // initialize stop length to one less than the string length parameter
       int stopLen = (strLen - 1);

       // other variable declaration - the input character
       char inputChar = '~';

    // prime loop with first input cast as a character
       // function: .get
       inputChar = char( fin.get() ); 

    // start loop through num chars or until end character
    // test for file good AND index less than the stop length
    // AND the input character not equal to the stop character parameter
       // function: .good
       while(  fin.good() && index < stopLen && inputChar != stopChar )
          {
        // update index
           index ++;

        // attempt to acquire next character, again as a character
           // function: .get
           inputChar = char( fin.get() ); 

    // end loop
           }
   }
