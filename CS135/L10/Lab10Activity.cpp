//  Header Files
#include <iostream>

using namespace std;

//  Global Constant Definitions

const int MAX_ITEMS = 25;
const char COMMA = ',';

//  Global Function Prototypes

void loadNumbers( int arr[], int numberToLoad );
void displayNumbers( int arr[], int numNums, const string &name );

//  Main Program Definition
int main()
   {
    // initialize program

       // variable initialization
       int fromIndex, toIndex, numEvenNums, numOriginalNums = 25;
       int intArray[ MAX_ITEMS ];

    // load numbers into array
       // function: loadNumbers
       loadNumbers( intArray, numOriginalNums );

    // loop to end of array
    for( toIndex = 0, fromIndex = 0; fromIndex < numOriginalNums; fromIndex++ )
       {
        // check for even number
        if( intArray[ fromIndex ] % 2 == 0 ) 
           {
            // assign even number to next available even number location
            intArray[ toIndex ] = intArray[ fromIndex ];

            // increment the toIndex
            toIndex++;
           }

        // show which number was processed
           // function: iostream <<
        cout << "After processing the number " 
             << intArray[ fromIndex ] << COMMA << endl;

        // display list with original numbers
           // function: displayNumbers
           displayNumbers( intArray, numEvenNums, "Original List");

        // set number of even numbers and display processed list of evens
           // function: displayNumbers
        numEvenNums = toIndex;
        displayNumbers( intArray, numEvenNums, "New Even List" );

        // hold the screen for the user
           // function: system/pause
        system( "pause" );

        // make vertical space
           // function: iostream <<
        cout << endl;
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

void loadNumbers( int arr[], int numberToLoad )
   {
    // initialize function/variables
    int index, number;

    // loop across number to load
    for( index = 0, number = 1; number <= numberToLoad; index ++, number ++ )
       {
        // load number into element at index
        arr[ index ] = number;
       }
   }

void displayNumbers( int arr[], int numNums, const string &name )
   {
    // initialize function/variables
    int index;

    // output list title
       // function: iostream <<
    cout << name << " Number List: ";

    // loop through array
    for( index = 0; index < numNums; index++ )
       {
        // check for item after first
        if( index > 0 )
           {
            // output comma/space
               // function: iostream <<
            cout << ", ";
           }

        // output number
           // function: iostream <<
        cout << arr[ index ];
       }

    // end line
       // function: iostream <<
    cout << endl;
   }


