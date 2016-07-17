// header files
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cstring>

using namespace std;

// global constants

   const int STD_STR_LEN = 45;
   const int MAX_LINE_LENGTH = 70;
   const int OPTIME_MIN = 5;
   const int BASE_TEN = 10;

// function prototypes

void getNewOp( int percent, char opStr[], int maxCyclesPerOp );
int getCycleLimit( int oddPercent, int maxCyclesPerOp );
bool getOdds( int oddPercent );
int getRandBetween( int low, int high );
int checkLineLength( char str[], int startLen, ofstream &outF );

// main program
int main()
   {
    // initialize program

       // initialize variables
       int procPercentage, numOps, numPrograms, maxCycles;
       int progCtr, opCtr, lineLength = 0;
       char fileName[ STD_STR_LEN ];
       ofstream outFileObject;
       char opString[ STD_STR_LEN ];
       char tempString[ STD_STR_LEN ];

       // initialize random generator
       srand( time( NULL ) );

       // show title
       cout << "Program Meta-Data Creation Program" 
            << endl << endl;

    // get file name for meta-data file
    cout << "Enter file name to use: ";
    cin >> fileName;

    // get max cycle number
    cout << "Enter maximum cycles possible per process: ";
    cin >> maxCycles;

    // get weight of processing
    cout << "Enter percentage of processing: ";
    cin >> procPercentage;

    // get number of actions
    cout << "Enter number of operations per program: ";
    cin >> numOps;

    // get number of programs to generate
    cout << "Enter number of programs: ";
    cin >> numPrograms;

    // open file
    outFileObject.open( fileName );

    // output file description header
    outFileObject << "Program Meta-Data Code:" << endl << endl;

    // set temporary string with string literal
    strcpy( tempString, "S(start)0; " );

    // output Operating System start
    outFileObject << tempString;
   
    // update length, output end of line as needed
    lineLength = checkLineLength( tempString, lineLength, outFileObject );

    // loop across number of programs
    for( progCtr = 0; progCtr < numPrograms; progCtr++ )
       {
        // set temporary string with string literal
        strcpy( tempString, "A(start)0; " );
    
        // update length, output end of line as needed
        lineLength = checkLineLength( tempString, lineLength, outFileObject );

        // show begin of program
        outFileObject << tempString;

        // loop across number of operations
        for( opCtr = 0; opCtr < numOps; opCtr++ )
           {
            // get new op
            getNewOp( procPercentage, opString, maxCycles );

            // update length, output end of line as needed
            lineLength = checkLineLength( opString, lineLength, outFileObject );

            // output new op
            outFileObject << opString;
           }
        // end loop across number of operations

        // set temporary string with string literal
        strcpy( tempString, "A(end)0; " );

        // update length, output end of line as needed
        lineLength = checkLineLength( tempString, lineLength, outFileObject );

        // add end of program
        outFileObject << tempString;
       }
    // end number of programs loop

    // set temporary string with string literal
    strcpy( tempString, "S(end)0." );

    // update length, output end of line as needed
    lineLength = checkLineLength( tempString, lineLength, outFileObject );

    // output end of operating system
    outFileObject << tempString;

    // output extra endline        
    outFileObject << endl;

    // close file
    outFileObject.close();

    // shut down program

       // hold program
       //system( "pause" );

       // return success
       return 0; 
   }

// supporting function implementations

int getRandBetween( int low, int high )
   {
    // initialize function/ variables

       // declare and define range
       int range = high - low + 1;

    // return constrained random value
    return rand() % range + low;    
   }

bool getOdds( int oddPercent )
   {
    // initialize function/ variables

       // initialize and generate random value
       int randVal = rand() % 100 + 1;

    // check for random value within odds
    if( randVal <= oddPercent )
       {
        // return true
        return true;
       }

    // assume value outside odds, return false
    return false;
   }

int getCycleLimit( int oddPercent, int maxCyclesPerOp )
   {
    // initialize function/ variables

    // process percent, return
    return int( ( double( oddPercent ) / 100 ) * maxCyclesPerOp );
   }

void getNewOp( int percent, char opStr[], int maxCyclesPerOp )
   {
    static int ioOpVal = 0;
    static bool runLastTime = false;
    int cycleNum;
    int procHigh = getCycleLimit( percent, maxCyclesPerOp );
    int ioHigh = maxCyclesPerOp - procHigh;

    char numStr[ STD_STR_LEN ];

    // check for chance of processing
    if( !runLastTime )
       {
        strcpy( opStr, "P(run)" );

        cycleNum = getRandBetween( OPTIME_MIN, procHigh );

        runLastTime = true;
       }

    else
       {
        if( ioOpVal == 0 ) // input process
           {
            if( getOdds( 50 ) ) // toss coin
               {
                strcpy( opStr, "I(hard drive)" );
               }
            else
               {
                strcpy( opStr, "I(keyboard)" );
               }

            ioOpVal = 1;
           }
        else
           {
            if( getOdds( 50 ) ) // toss coin
               {
                strcpy( opStr, "O(hard drive)" );
               }
            else
               {
                strcpy( opStr, "O(monitor)" );
               }

            ioOpVal = 0;
           }

        cycleNum = getRandBetween( OPTIME_MIN, ioHigh );

        runLastTime = false;
       }

    sprintf( numStr, "%d", cycleNum );

    strcat( opStr, numStr );

    strcat( opStr, "; " );
   }

int checkLineLength( char str[], int startLen, ofstream &outF )
   {
    int newLength = startLen + strlen( str );

    if( newLength > MAX_LINE_LENGTH )
       {
        outF << endl;

        return 0;
       }

    return newLength;
   }


