/*/////////////////////////////////////////////////////////////////////////

    Program:             Lab5Activity.cpp
    Author:              Michael Leverington
    Update By:           n/a
    Description:         Lab Activity - Week 4
                         

    Program Written:     01/22/2009
    Most Recent Update:  01/22/2009 - 12:00 a.m.
    Date Due for Review: 01/22/2009 

/////////////////////////////////////////////////////////////////////////*/

/*
INSTRUCTIONS:

    None

END OF INSTRUCTIONS
*/
//
//  Header Files  /////////////////////////////////////////////////////////
//
    #include <iostream>
    
    using namespace std;
//
//  Global Constant Definitions  //////////////////////////////////////////
//
    // none
//
//  Global Function Prototypes  ///////////////////////////////////////////
//
bool isLetter( char testChar );
bool isLowerCaseLetter( char testChar );
bool isUpperCaseLetter( char testChar );

//
//  Main Program Definition  //////////////////////////////////////////////
//
int main()
   {
    // initialize program

       // variable initialization
       char someChar;
 
    // prompt user for character (someChar) using cout, cin
       cout << "GIVE ME SOME CHAR!!! ";
       cin >> someChar;
       cout << endl;

    // is letter?
    if( !isLetter( someChar ) )
      {
       cout << someChar <<" IS NOT A LETTER!!!!!" << char(2) << endl;;
      }
  
else
  {
    // test and report lower case letter
    if( isLowerCaseLetter( someChar ) )
       {
        // report that it is a lower case letter
        cout << endl << endl;
        cout << "Letter " << someChar 
             << " is a lower case letter." << endl << endl;
       }
    else
       {
        // report that it is not a lower case letter
        cout << endl << endl;
        cout << "Letter " << someChar 
             << " is not a lower case letter." << endl << endl;
       }

    // test and report lower case letter
    if( isUpperCaseLetter( someChar ) )
       {
        // report that it is an upper case letter
        cout << endl << endl;
        cout << "Letter " << someChar 
             << " is an upper case letter." << endl << endl;
       }
    else
       {
        // report that it is not an upper case letter
        cout << endl << endl;
        cout << "Letter " << someChar 
             << " is not an upper case letter." << endl << endl;
       }
  }
    // Close program
       // hold program
       system( "pause" );

       // return success
       return 0;
   }

//  Supporting Function Implementation
bool isLetter( char testChar )
  {
   if( (testChar >= 'a' && testChar <= 'z') || (testChar >= 'A' && testChar <= 'Z') ) 
     {
     return true;
     }
  }

bool isLowerCaseLetter( char testChar )
  {
   if( testChar >= 'a'  &&  testChar <= 'z' )
     {
     return true;
     }
  }

bool isUpperCaseLetter( char testChar )
  {
   if( testChar >= 'A'  &&  testChar <= 'Z' )
     {
     return true;
     }
  }
