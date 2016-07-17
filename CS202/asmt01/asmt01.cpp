/////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      asmt01: Sequence Processor
//  Created By: Terence Henriod
//  Course:     CS 202
//
//  This program processes a range of numbers using a conditional math sequence
//  and outputs relevant data to .txt and .csv files.
//
/////////////////////////////////////////////////////////////////////////////////


/// Header Files/////////////////////////////////////////////////////////////////
#include <iostream>
#include <conio.h>
#include <fstream>
#include <assert.h>
using namespace std;


/// Global Constant Definitions//////////////////////////////////////////////////

// string lengths
const int NAME_LEN = 20; 

// max sequence length
const int MAXSEQLEN = 1000;   // doubles as the maximum range to prevent a gross amount of memory usage for this project

// the base case
const int BASE = 1;   // 1 is not only the assigned base case, but likely the most appropriate

// data array specific constants
const int MAX_LEN = 2;   // defines the two columns needed for the maximum numbers reached and the sequence lengths
const int MAX_ELE = 0;   // the column containing maximum number reached
const int LENGTH = 1;   // the column containing sequence lengths

// file names
const char CSVOUTP[NAME_LEN] = "asmt01.csv";
const char TXTOUTP[NAME_LEN] = "asmt01.txt";

// exit codes
const int EXIT_BADMENU = 1;
const int EXIT_BADNUM = 2;
const int EXIT_BADORDER = 3;


/// Global Function Prototypes///////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
//
// Function Name: exit_function
// Summary:       Exits the program after displaying an error message appropriate
//                to the situation.
// Parameters:
// Returns:       void
//
///////////////////////////////////////////////////////////////////////////////// 
void exit_function( int code );

/////////////////////////////////////////////////////////////////////////////////
//
// Function Name: menu
// Summary:       Exits the program after displaying an error message appropriate
//                to the situation.
// Parameters:
// Returns:       The number of the selection
// 
///////////////////////////////////////////////////////////////////////////////// 
int menu();

/////////////////////////////////////////////////////////////////////////////////
//
// Function Name: getRange
// Summary:       Prompts the user for a range of positive integers. Numbers must 
//                be in order of least to greatest.
// Parameters:
// Returns:       void
// 
///////////////////////////////////////////////////////////////////////////////// 
void getRange( int &startNum, int &endNum );

/////////////////////////////////////////////////////////////////////////////////
//
// Function Name: processRange
// Summary:       Takes the range and applies the conditional function, gathers
//                relevant data. Stops after a maximum number of iterations to 
//                prevent "infinite" looping, or if the base case of 1 has been
//                reached.
// Parameters:
// Returns:       Returns an integer to end the function.
// 
///////////////////////////////////////////////////////////////////////////////// 
int processRange( int startNum, int endNum, int &minNum, int &minLen, int minSeq[],  
                  int &maxNum, int &maxLen, int maxSeq[], int max_len_data[MAXSEQLEN][MAX_LEN] );

/////////////////////////////////////////////////////////////////////////////////
//
// Function Name: clrArr
// Summary:       Fills all elements of an integer array with 1; an attempt to 
//                model the practicality of a C-string, but with integers
// Parameters:
// Returns:       void
// 
///////////////////////////////////////////////////////////////////////////////// 
void clrArr( int arr[MAXSEQLEN] );

/////////////////////////////////////////////////////////////////////////////////
//
// Function Name: mathFunction
// Summary:       Applies the conditional function:
//                if x is odd  f(x) = 3x + 1
//                if x is even f(x) = x / 2
// Parameters:
// Returns:       The integer outcome of the application of the function.
// 
///////////////////////////////////////////////////////////////////////////////// 
int mathFunction( int number );

/////////////////////////////////////////////////////////////////////////////////
//
// Function Name: copySeq
// Summary:       Copies the contents of one array into another, but the array
//                attempting to mimic the functionality of a C-string, using 1 as
//                the null terminator.
// Parameters:
// Returns:       void
// 
///////////////////////////////////////////////////////////////////////////////// 
void copySeq( int original[], int replacement[] );

/////////////////////////////////////////////////////////////////////////////////
//
// Function Name: writeCSV
// Summary:       Creates a .csv file that contains every range element processed,
//                the maximum number, and length of the respective sequences.
// Parameters:
// Returns:       void
// 
///////////////////////////////////////////////////////////////////////////////// 
void writeCSV( int startNum, int endNum, int max_len_data[MAXSEQLEN][MAX_LEN] );

/////////////////////////////////////////////////////////////////////////////////
//
// Function Name: menu
// Summary:       Writes a text file containing the elements of the range that 
//                produced the shortest and longest sequences, as well as the 
//                entirety of their respective sequences.
// Parameters:
// Returns:       void
// 
///////////////////////////////////////////////////////////////////////////////// 
void writeTXT( int startNum, int endNum,  int minNum, int minLen, int minSeq[],  int maxNum, int maxLen, int maxSeq[] );


/// Main Program Definition//////////////////////////////////////////////////////
int main()
   { 
   // vars
   int menuSel = 0;

   // range values
   int startNum = 0;
   int endNum = 0;

   // important numbers within the range
   int minNum = 0;   // These indicate the numbers that will produce the shortest/longest sequences
   int maxNum = 0;   //

   // the important sequences and their attributes
   int minSeq[MAXSEQLEN];
   int maxSeq[MAXSEQLEN];
   int minLen;
   int maxLen;
   int max_len_data[MAXSEQLEN][MAX_LEN];
   
   // implement menu
     menuSel = menu();

     // end program if selected by user
     if( menuSel == 2 )
       {
       return 0;
       }

     // execute sequence processing, if selected
     if( menuSel == 1 )
       {    
       // collect range definition
       getRange( startNum, endNum );

       // execute processing
       processRange( startNum, endNum, minNum, minLen, minSeq, maxNum, maxLen, maxSeq, max_len_data );

       // create necessary files
       writeCSV( startNum, endNum, max_len_data );
       writeTXT( startNum, endNum,  minNum, minLen, minSeq, maxNum, maxLen, maxSeq );
       }

   // end program
   return 0;
   }


/// Supporting function implementations//////////////////////////////////////////

void exit_function( int code )
   {
   // vars
   int n = 0;

   // clear the screen
   while( n < 100 )
     {
     cout << endl;
     n ++;
     }

   // display approriate message
   switch( code )
     {
     case 1:
       {
       cout << "Invalid menu input entered. Please enter an appropriate option." 
            << endl;
       }
       break;
     case 2:
       {
       cout << "Invalid range entries entered. Please enter only positive integers."
            << endl;
       }
     case 3: 
       {
       cout << "First entry must be less than or equal to the second entry." << endl
            << "Please ensure that positive integers are entered in the appropriate order."
            << endl;
       }
     }
     cout << endl << "Program will now terminate." << endl << endl << endl << endl;

   // hold the program for the user
   cout << "   Press any key to continue...";
   for( n = 0; n < 15; n ++ )
     {
     cout << endl;
     }

   while( ! _kbhit() )
     {
     // do nothing until user hits a key
     }

   // end the program
   exit( code );

   // no return - void  plus exit() should have already terminated the prog
   }


int menu()
   {
   // vars
   int selection = 0;

   // initialize menu
   cout << endl << endl << endl << endl << endl << endl;
   cout << "MAIN MENU" << endl << endl;
   cout << "What would you like to do?" << endl << endl;
   cout << "1. Compute Sequences" << endl
        << "2. Quit" << endl << endl;

   // prompt for selection
   cin >> selection;
   cout << endl << endl;

   // test selection for bad input
   if( (selection != 1) && (selection != 2) )
     {
     exit_function( EXIT_BADMENU );
     }

   // return signal of action to be taken
   return selection;
   }


void getRange( int &startNum, int &endNum )
   {
   // prompt for beginning of a range
   cout << "Enter the starting number: ";
   cin >> startNum;
   cout << endl << endl;

   // prompt for end of range
   cout << "Enter the ending number: ";
   cin >> endNum;
   cout << endl << endl;

   // check for valid input
   if( (startNum <= 0) || (endNum <= 0) )
     {
     exit_function( EXIT_BADNUM );
     }
   if( endNum < startNum )
     {
     exit_function( EXIT_BADORDER);
     }

   // no return - void
   }

int processRange( int startNum, int endNum, int &minNum, int &minLen, int minSeq[],  
                  int &maxNum, int &maxLen, int maxSeq[], int max_len_data[MAXSEQLEN][MAX_LEN] )
   {
   // vars
   int currVal = startNum;
   int element = -888;
   int ndx = 0; 
   int rndx = 0;
   int buffer[MAXSEQLEN];

   // prep reference vars
   clrArr( minSeq );
   clrArr( maxSeq );
   minLen = 1000;
   maxLen = 0;

   // process the sequence for every number in the range
   for( rndx = 0; (rndx < MAXSEQLEN) && (currVal <= endNum ); currVal ++, rndx ++ )
	 {
    ndx = 0;
    clrArr( buffer );

    element = currVal; // get starting number in sequence
	 buffer[ndx] = element;
    max_len_data[rndx][MAX_ELE] = element;
    ndx ++;

    while( (element != BASE) && (ndx < MAXSEQLEN) )   // gather the rest of the sequence
      {
      element = mathFunction( element );
      buffer[ndx] = element;

      if( element > max_len_data[rndx][MAX_ELE] )   // store the maximum number in the sequence
        {
        max_len_data[rndx][MAX_ELE] = element;
        }

		ndx ++;
	   }

      // manage shortest/longest sequence info
      max_len_data[rndx][LENGTH] = ndx;
      if( ndx < minLen )   // these utilize the numbers that begin the respective sequence types
        {
        minLen = ndx;
        minNum = currVal;
        copySeq( minSeq, buffer );
        }
      if( ndx > maxLen )
        {
        maxLen = ndx;
        maxNum = currVal;
        copySeq( maxSeq, buffer );
        }
    }

   // return 0 to end function
   return 0;
   }


void clrArr( int arr[MAXSEQLEN] )
   {
   // vars
   int ndx;

   //clear the array, using 1s to indicate the end of a sequence
   for( ndx = 0; ndx < MAXSEQLEN; ndx ++ )
     {
     arr[ndx] = BASE;
     }

   // no return - void
   }


int mathFunction( int number )
   {
   // vars
   int result;

   // implement f(x)
   if( number == BASE )
     {
     return BASE;
     }
   if( (number % 2) == 1)   // if x is odd, f(x) = 3x + 1
     {
	 result = ((3 * number) + 1);
     }
   if( (number % 2) == 0)   // if x is even, f(x) = x/2
     {
	 result = ( number / 2 );
     }

   return result;	   
   }


void copySeq( int original[], int replacement[] )
   {
   // vars
   int ndx = 0;

   clrArr( original );

   while( replacement[ndx] != 1 )
     {
     original[ndx] = replacement[ndx];
     ndx ++;
     }

   // no return - void
   }


void writeCSV( int startNum, int endNum, int max_len_data[MAXSEQLEN][MAX_LEN] )
   {
   // vars
   int rndx = 0;
   int currVal = startNum;
   char delim = ',';
   char f_header[MAXSEQLEN] = "Starting Number,Maximum Number,Length";
   ofstream fout;
   
   // clear and open
   fout.clear();
   fout.open( CSVOUTP );

   // write file
   fout << f_header << endl;

   while( currVal <= endNum )
     {
     fout << currVal << delim << max_len_data[rndx][MAX_ELE] << delim << max_len_data[rndx][LENGTH]
          << endl;
     
     currVal ++, rndx ++; 
     }

   // no return - void
   }


void writeTXT( int startNum, int endNum,  int minNum, int minLen, int minSeq[],  int maxNum, int maxLen, int maxSeq[] )
   {
   // vars
   int ndx = 0;
   ofstream fout;

   // clear and open
   fout.clear();
   fout.open( TXTOUTP );
   
   // write the file
   fout << "Beginning of the range:  " << startNum
        << endl << endl;

   fout << "End of Range:  " << endNum
        << endl << endl;

    fout << "Starting number for the sequence with minimum length:  " << minNum << endl
         << "The minimum length:                                    " << minLen
         << endl << endl;

    fout << "Starting number for the sequence with maximum length:  " << maxNum << endl
         << "The maximum length:                                    " << maxLen
         << endl << endl;

    fout << "Sequence with minimum length:  ";
    for(ndx = 0; ndx < minLen; ndx ++ )
      {
      fout << minSeq[ndx] << endl
         << "                               ";
      }
     fout << endl << endl;

    fout << "Sequence with maximum length:  ";
    for(ndx = 0; ndx < maxLen; ndx ++ )
      {
      fout << maxSeq[ndx] << endl
         << "                               ";
      }

   // close
   fout.close();

   // no return - void
   }
