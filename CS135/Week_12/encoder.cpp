////////////////////////////////////////////////////////////////////////////////
// Header Files ////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
   #include <iostream>
   #include <fstream> 
   using namespace std ;


////////////////////////////////////////////////////////////////////////////////
// Global Constant Definitions /////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// character constants

const char NULL_CHAR = '\0';

// numerical constants

const int MAX_MESSAGE = 1001;     // 1001 to accomodate for the null terminator

const int MAX_ENCODED = 7500;

const int OUT_STRNG_LEN = 8;

const int EVEN_PROB = 25;

////////////////////////////////////////////////////////////////////////////////
// Global Function Prototypes //////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

    // none

/* 
Name: 
Process: 
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: none
*/
void printTitle();


void promptForFileNames( char inName[], char outName[] );


int readData( char inName[], char message[], bool &fileThere, bool &fileTooLarge, int &messSize );


void noFileError( char inName[] );


void oversizeError( char inName[]);








////////////////////////////////////////////////////////////////////////////////
// Main Program Definition /////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int main()
   { 

   // initialize function/variables

      // vars

         // c-strings
         char inputFile[ MAX_MESSAGE ]  = "None entered yet...";
         char outputFile[ MAX_MESSAGE ] = "None entered yet...";
         char message[ MAX_MESSAGE ];
         int encodedMessage[ MAX_ENCODED ];

         // flags
         bool fileThere  = false;
         bool fileTooLarge = true;
         int messageSize = 0;

         // initialize random seed 
         srand( time( NULL ) );

      // display program title
         // function: printTitle
         printTitle();

   // prompt for in/out file names
      // function: promptForFileNames
      promptForFileNames( inputFile, outputFile );

   // read in message
      // function: readData
      readData( inputFile, message, fileThere, fileTooLarge, messageSize );

      // if input file is invalid/reading fails, abort program
         // operation: if
            if( !fileThere )
              {
               // display error message
                  // function: noFileError
                  noFileError( inputFile ); 

               // return 0 to end program
               return 0;          
              }

// if input file is too large, abort program
         // operation: if
            if( fileTooLarge )
              {
               // display error message
                  // function: oversizeError
                  oversizeError( inputFile ); 

               // return 0 to end program
               return 0;          
              }

   // encrypt message
      // function: encodeData

   // write encrypted message
      // function: writeCodedMessage

      // if writing was unsuccessful, notify user and end program
         // operation: if

   // end program

      // hold program for user
         // function: system 
         system ( "PAUSE" );
    
      // return 0
      return 0;
   }


////////////////////////////////////////////////////////////////////////////////
// Supporting function implementations /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void printTitle()
   {
   // display title with underline, create space between title and rest of program
      // function: cout
      cout << "     ENCODER PROGRAM" << endl
           << "     ===============" << endl << endl;

   // no return - void
   }


void promptForFileNames( char inName[], char outName[] )
   {
   // prompt for input file name
      // functions: cout, cin
      cout << "Enter input file name: ";
      cin  >> inName;
      cout << endl;
  
   // prompt for output file name
      // functions: cout, cin
      cout << "Enter output file name: ";
      cin  >> outName;
      cout << endl << endl;

   // no return - void
   }


int readData( char inName[], char message[], bool &fileThere, bool &fileTooLarge, int &messSize )
   {
   // initialize variables and flags
   ifstream fin;
   int index = 0;
   fileThere = false;
   fileTooLarge = true;

   // notify user reading is being attempted
      // function cout
      cout << "Uploading Data . . ." << endl << endl;

   // open the input file
      // functions: .clear, .open
      fin.clear();
      fin.open( inName );

   // read file into program

      // prime the loop
         // function: .get
         message[ index ] = fin.get();

      // use loop to continue reading file, while possible
         // operation: while
         while( fin.good() && index < ( MAX_MESSAGE ))     
           {
            // indicate reading was successful
            fileThere = true;

            // increment index
            index ++;

            // get character ASCII value and store it to array
               // function: .get
               message[ index ] = char( fin.get() );         
           }

   // if file is of acceptable size, set the flag to indicate such
      // operation: if
      if( index >= MAX_MESSAGE )     // indicates that there is no room for the null terminator
        {
         // set flag
         fileTooLarge = true;

         // end function by returning 0
         return 0;
        }

   // store the message size for later reference
   messSize = index;     // index will be at the null terminator position, so it is a count of the chars

   // terminate the c-style string with \0
   message[ index ] = NULL_CHAR;

   // close the input file
      // function: .close
      fin.close();

   // return 0 to end function  
   return 0;
   }


void noFileError( char inName[] )
   {
   // display message
      // function: cout
      cout << endl
           << "ERROR: File \"" << inName 
           << "\" Not Found - Program Aborted"
           << endl << endl << endl;

   // hold system for user
      // function: system
      system( "PAUSE" );

   // no return - void    
   }


void oversizeError()
   {
   // display the message
      // function: cout
      cout << "ERROR: The message is too large -"
           << "Rather than write a truncated message," << endl
           << "program will now terminate..." << endl << endl;

   // hold program for user
      // function: system
      system( "PAUSE" );
   
   // no return - void   
   }


void encodeData( char message[], int encodedMessage[], int messSize )
   {
   // initialize variables
   int messIndex = 0;
   int codeIndex = 0;


   }














bool doLoadEven( int timesIn100 )
   {
   // initialize variables
   bool decision = true;
   int option = 0; 

   // use rand() to generate a number to use to make the decision
      // function: rand
      option = ( ( rand() % 100 ) );

   // decide whether or not to send the signal to output an even value
      // operation: if
      if( option >= 0 && option < timesIn100 )
        {
         // decision will be true to indicate that an even value should be outputted
         decision = true;    
        } 

      // operation: else
      else if( option >= timesIn100 && option <= 99 )
        {
         // decision will be false to indicate an odd value should be outputted
         decision = false;
        }
    
   // return whatever decision was made
   return decision;
   }
