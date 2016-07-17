//  Header Files
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

//  Global Constant Definitions

const string FILE_NAME = "testfile.txt";
const int LOW_VAL = 10;
const int HIGH_VAL = 99;
const int ITEMS_PER_LINE = 10;
const char DASH = '-';
const string COMMA_SPACE = ", ";

//  Global Function Prototypes

void generateNumbers( int count );

bool displayNumbers(); 

int getRandBetween( int low, int high );

//  Main Program Definition
int main()
   {
    // initialize program

       // variable initialization
       int numNums;

       // seed random number generator
       srand( time( NULL ) );
 
    // get count of numbers from user
    cout << "Enter number of numbers to generate: ";
    cin >> numNums;

    // generate some random numbers
    generateNumbers( numNums );

    // display numbers, check for failure
    if( !displayNumbers() )
       {
        cout << "ERROR: File access failure, program aborted"
             << endl;
       }

    // Close program

       // create vertical lines
       cout << endl << endl;

       // hold program
       system( "pause" );

       // return success
       return 0;
   }

//  Supporting Function Implementation

void generateNumbers( int count )
   {
    // initialize function/variables
    ofstream outF;
    int outVal, counter;

    // open file
    outF.open( FILE_NAME.c_str() );

    // write numbers to file

       // loop up to count
       for( counter = 0; counter < count; counter++ )
          {
           // get random value
           outVal = getRandBetween( LOW_VAL, HIGH_VAL );

           // check for all items other than first one
           if( counter > 0 )
              {
               // output a comma and a space
               outF << COMMA_SPACE;
              }

           // check for items in line at max
           if( counter % ITEMS_PER_LINE == 0 )
              {
               // output end line
               outF << endl;
              }

           // output the value
           outF << outVal;
          }

    // add end line to last value entered
    outF << endl;

    // close the file
    outF.close();
   }

// loadData
bool displayNumbers()
   {
    // initialize function/variables
    ifstream inf;
    int value, itemCounter = 0;
    char dummy;

    // clear and open file
    inf.clear(); 
    inf.open( FILE_NAME.c_str() );

    // prime loop with first read
    inf >> value ;

    // check for good file
    while( inf.good() )
       {
        // bring in the comma/delimiter
        inf >> dummy;

        // check for max items on line
        if( itemCounter % ITEMS_PER_LINE == 0 )
           {
            // output an end of line
            cout << endl;
           }

        // otherwise, if not at end of line
        else
           {
            // output a dash
            cout << DASH;
           }

        // display value
        cout << value;

        // increment item counter
        itemCounter++;

        // attempt to read in next data
        inf >> value;
       }

    // close file
    inf.close();

    // add end line to last value
    cout << endl;

    // return file access success (or failure)
    return ( itemCounter > 0 );
   }

int getRandBetween( int low, int high )
   {
    // initialize function/variables
    int range = high - low + 1;
    int randVal;

    // generate value
    randVal = rand() % range + low;

    // return value
    return randVal;
   }

