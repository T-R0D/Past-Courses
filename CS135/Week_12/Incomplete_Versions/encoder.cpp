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

const int MAX_CAP = 1002;     // 1002 to allow for a message larger than 1000 
                              // characters to be identifiable 


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


bool readData( char inName[], char message[] );


void readErrorMessage( char inName[] );


void encodeData( char message[] );


bool writeEncodedMessage( char outName[], int encodedMessage[] );


int findMessageSize( int encodedMessage[] );


bool doOutputEven();




////////////////////////////////////////////////////////////////////////////////
// Main Program Definition /////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int main()
   { 

   // initialize function/variables

      // vars

         // c-strings
         char inputFile[ MAX_CAP ]  = "None entered yet...";
         char outputFile[ MAX_CAP ] = "None entered yet...";
         char message[ MAX_CAP ];
         int encodedMessage[ MAX_CAP ];

         // flags
         bool readSuccess  = false;
         bool writeSuccess = false;

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
      readSuccess = readData( inputFile, message );

      // if input file is invalid/reading fails, abort program
         // operation: if
            if( !readSuccess )
              {
               // display error message
                  // function: readErrorMessage
                  readErrorMessage( inputFile ); 

               // return 0 to end program
               return 0;          
              }

   // encrypt message
      // function: encodeData

   // write encrypted message
      // function: writeCodedMessage
      writeSuccess = writeEncodedMessage( outputFile, encodedMessage );

      // if writing was unsuccessful, notify user and end program
         // operation: if
         if( !writeSuccess )
           {
            // display error message
               // function
           }

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


bool readData( char inName[], char message[] )
   {
   // initialize variables
   bool readSuccess = false;
   ifstream fin;
   int index = 0;

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

      // increment index
      index ++;

      // use loop to continue reading file, if possible
         // operation: while
         while( fin.good())
           {
            // indicate reading was successful
            readSuccess = true;

            // get character ASCII value and store it to array
               // function: .get
               message[ index ] = char( fin.get() );         

            // increment index
            index ++;
           }

   // terminate the c-style string with \0
   message[ index ] = NULL_CHAR;

   // close the input file
      // function: .close
      fin.close();

   // return response to indicate if reading was successful   
   return readSuccess;
   }


void readErrorMessage( char inName[] )
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















void encodeData( char message[] )
   {
   // initialize variables

   }


bool writeEncodedMessage( char outName[], int encodedMessage[] )
   {
   // initialize variables
   ofstream fout;
   bool writeSuccess = false;
   int index = 0;
   int valCount = 0;

   // notify user encryption hase begun
      // function: cout
      cout << endl << "Downloading Data . . ." << endl << endl;

   // check to see if message is too large, if so, terminate program
      // operation: if, function: findMessageSize
      if( findMessageSize( encodedMessage ) > 1000 )
        {
         // notify main that message is too large
            // return false
            writeSuccess = false; 
            return writeSuccess; 
        }

   // open output file
      // functions: .clear, .open
      fout.clear();
      fout.open( outName );

   // write data to file, stop when null terminator is reached
      // operation: while 
      while( encodedMessage[ index ] != NULL_CHAR )
        {
         // when outputting, only output 5 values per line, then start a new line
            // operation: for
            for( valCount = 0; valCount < 5; valCount ++ )
              {

               // determine whether of not to write an even (true) value to the file
                  // operation: if, function: doOutputEven 
                  if( doOutputEven() )
                    {
                     // write an encoded true value to the file










                     // increment the index to get the next true value to be written
                     index ++;
                    }

               // otherwise, output an odd (dummy) value
                  // operation: else
                  else
                    {










                    }
              }

            // start the new line
               // function: fout <<
               fout << endl;
        }

   // close the file
      // function: .close
      fout.close();

   // return success status
   return writeSuccess;
   }


int findMessageSize( int encodedMessage[] )
   {
   // initialize variables
   int index = 0;
   int charCount;

   // loop through to find message length, stop when \0 is encountered
      // operation: for
      for( index = 0; encodedMessage[ index ] != NULL_CHAR; index ++ )

   // index will now refer to the position past \0, this will be the count of
   // characters including the null terminator
   charCount = index;

   // return number of characters
   return charCount; 
   } 


void writeErrorMessage()
   {
   // display the message
      // function: cout
      cout << "ERROR: The message is too large -" << endl
           << "Rather than write a truncated message," << endl
           << "program will now terminate..." << endl << endl;

   // hold program for user
      // function: system
      system( "PAUSE" );
   
   // no return - void   
   }


bool doOutputEven()
   {
   // initialize variables
   bool decision = true;
   int option = 0; 

   // use rand() to generate a number to use to make the decision
      // function: rand
      option = ( ( rand() % 100 ) );

   // decide whether or not to send the signal to output an even value
      // operation: if
      if( option <= 0 && option >= 24 )
        {
         // decision will be true to indicate that an even value should be outputted
         decision = true;    
        } 

      // operation: else
      else if( option >= 25 && option <= 99 )
        {
         // decision will be false to indicate an odd value should be outputted
         decision = false;
        }
    
   // return whatever decision was made
   return decision;
   }





























void inputGetLine( ifstream &fin, char inStr[], int strLen, char stopChar )
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
