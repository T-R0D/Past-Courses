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
const int MAX_VALUES = 100;
const char DASH = '-';
const string COMMA_SPACE = ", ";

//  Global Function Prototypes

void generateNumbers( int count, int arr[] );

void downloadNumbers( int count, int arr[] );

void findLimitValues( int numNums, int arr[], int &lowVal, int &highVal );

int uploadNumbers( int arr[] );

void displayValues( int count, int arr[] );

int getRandBetween( int low, int high );

//  Main Program Definition
int main()
   {
    // initialize program

       // variable initialization
       int downloadNums, uploadNums, lowest, highest;
       int downloadArray[ MAX_VALUES ];
       int uploadArray[ MAX_VALUES ];

       // seed random number generator
       srand( time( NULL ) );
 
    // get count of numbers from user
    cout << "Enter number of numbers to generate: ";
    cin >> downloadNums;

    // generate some random numbers
       // function: generateNumbers
    generateNumbers( downloadNums, downloadArray );

    // download number list
       // function: downloadNumbers
    downloadNumbers( downloadNums, downloadArray );

    // acquire numbers to new array
       // function: uploadNumbers
    uploadNums = uploadNumbers( uploadArray );

    // check for values found
    if( uploadNums > 0 )
       {
        // find lowest and highest values
           // function: findLimitValues
        findLimitValues( uploadNums, uploadArray, lowest, highest );

        // display values
           // function: displayValues
        displayValues( uploadNums, uploadArray );

        // display limit values
           // function: iostream <<
        cout << "Lowest value is: " << lowest << ", and "
             << "highest value is: " << highest << endl << endl;
       }
    // Close program

       // create vertical lines
           // function: iostream <<
       cout << endl << endl;

       // hold program
           // function: system/pause
       system( "pause" );

       // return success
       return 0;
   }

//  Supporting Function Implementation

void generateNumbers( int count, int arr[] )
   {
    // initialize function/variables
    int index;

    // write numbers to array

       // loop up to count
       for( index = 0; index < count; index++ )
          {
           // get random value
              // function: getRandBetween
           arr[ index ] = getRandBetween( LOW_VAL, HIGH_VAL );
          }
   }

// download numbers
void downloadNumbers( int count, int arr[] )
   {
    // initialize function/variables
    int index;
    ofstream outF;

    // open file for output
       // function: fstream .open
       outF.open( FILE_NAME.c_str() );

    // iterate across array
    for( index = 0; index < count; index++ )
       {
        // check for number of items on a line
        if( index % ITEMS_PER_LINE == 0 && index > 0 )
           {
            // output a comma with an end of line
              // function: fstream <<
            outF << ", " << endl;
           }

        // otherwise, check for index > 0
          if( index > 0)
           {
            // output a comma to delimit between numbers
              // function: fstream <<
            outF << ", ";
           }

        // output the value
           // function: fstream <<
        outF << arr[ index ];
       }

    // output end of lines
       // function: fstream <<
    outF << endl << endl;
   }

void findLimitValues( int numNums, int arr[], int &lowVal, int &highVal )
   {
    // initialize function/variables
    int index;

    // initialize limits
    lowVal = highVal = arr[ 0 ];

    // iterate across array
    for( index = 0; index < numNums; index++ )
       {
        // check for value lower than lowest
        if( arr[ index ] < lowVal )
           {
            // reset lowest value
            lowVal = arr[ index ];
           }

        // otherwise, check for value higher than highest
        if( arr[ index ] > highVal )
           {
            // reset highest value
            highVal = arr[ index ];
           }
       }    
   }

// loadData
int uploadNumbers( int arr[] )
   {
    // initialize function/variables
    ifstream inf;
    int value, index = 0;
    char dummy;

    // clear and open file
       // function: fstream .clear, .open
       inf.clear();
       inf.open( FILE_NAME.c_str() );

    // prime loop with first read
       // function: fstream >>
       inf >> value;

    // check for good file
       // function: fstream .good
       while( inf.good() )
       {
        // bring in the comma/delimiter
           // function: fstream >>
        inf >> dummy;

        // assign value to array
        arr[ index ] = value;

        // increment item counter
        index++;

        // attempt to read in next data
           // function: fstream >>
        inf >> value;
       }

    // close file
       // function: fstream .close
       inf.close();

    // return number of items
    return index;
   }

void displayValues( int count, int arr[] )
   {
    // initialize function/variables
    int index;

    // show presentation
    cout << endl << "Number list: ";

    // loop across array
    for( index = 0; index < count; index ++ )
       {
        // check for after first item
        if( index > 0 )
           {
            // display comma delimiter
                // function: iostream <<
            cout << ", ";
           }

        // output value
           // function: iostream <<
        cout << arr[ index ];
       }

    // output end lines
       // function: iostream <<
    cout << endl << endl;
   }

int getRandBetween( int low, int high )
   {
    // initialize function/variables
    int range = high - low + 1;
    int randVal;

    // generate value
       // function: rand
    randVal = rand() % range + low;

    // return value
    return randVal;
   }

