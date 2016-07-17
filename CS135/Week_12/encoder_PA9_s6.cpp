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
const char NEW_LINE  = '\n';
const char SPACE     = ' ';

// numerical constants

const int MAX_MESSAGE = 1001;     // 1001 to accomodate for the null terminator
const int MAX_ENCODED = 7500;
const int OUT_STR_LEN = 8;
const int EVEN_PROB = 25;
const int ASCII_LOW = 33;
const int ASCII_HI  = 126; 
const int EVEN = 0;
const int ODD  = 1;
const int MAX_VAL_PER_LINE = 5;


////////////////////////////////////////////////////////////////////////////////
// Global Function Prototypes //////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/* 
Name: printTitle
Process: displays the program title on the console
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: displays the program title
Dependencies: <iostream>
*/
void printTitle();


/* 
Name: promptForFileNames
Process: propmts the user for 2 file names: one for a message to read and encode,
         a second to create and write the coded message to
Function Input/Parameters: char inName[]      a c-style string for the name of
                                              message file to be read in 
                           char outName[]     a c-style string representing the 
                                              name of the output file to be created
Function Output/Parameters: (by reference)     char inName[]
                            (by reference)     char outName[]
Function Output/Returned: none
Device Input: the two file names
Device Output: a message prompting for the two file names
Dependencies: <iostream>, <fstream> 
*/
void promptForFileNames( char inName[], char outName[] );


/* 
Name: readData
Process: notifies the user attempted reading has begun, reads the message of the
         input file into a c-style string to store in the program
Function Input/Parameters: char inName[]         a c-style string for the name of
                                                 message file to be read in
                           char message[]        a c-style string that holds the 
                                                 message that is read in
                           bool fileThere        a boolean flag to indicate whether 
                                                 or not the input file name references
                                                 an existing file
                           bool fileTooLarge     a boolean flag to indicate whether
                                                 or not the message being read in is
                                                 too large to be handled by the program
                           int messSize          indicates how large the message
                                                 that was read in is
Function Output/Parameters: (by reference) char message[]
                            (by reference) bool fileThere
                            (by reference) bool fileTooLarge
                            (by reference) int messSize
Function Output/Returned: none
Device Input: none
Device Output: none
Dependencies: <iostream>, <fstream>
*/
int readData( char inName[], char message[], bool &fileThere, bool &fileTooLarge, int &messSize );


/* 
Name: noFileError
Process: displays an error message indicating that the given file name does not 
         reference an existing file  
Function Input/Parameters: char inName[]     a c-style string for the name of
                                             message file to be read in
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: the invalid file name error message
Dependencies: <iostream>
*/
void noFileError( char inName[] );


/* 
Name: oversizeError
Process: displays an error message indicating that the input message is too long
         to be processed by the program
Function Input/Parameters: char inName[]     a c-style string for the name of
                                             message file to be read in
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: the file is too large error message
Dependencies: <iostream>
*/
void oversizeError( char inName[]);


/* 
Name: encodeData
Process: notifies user that encryption has begun, encodes the message one character 
at a time, and places them interspersed between dummy values, even codes being 
valid characters, odd values being dummies
Function Input/Parameters: char message[]           a c-style string that holds the 
                                                    message that is read in
                           int encodedMessage[]     an array to contain both valid
                                                    and dummy values that have been
                                                    coded
                           int messSize             indicates how large the message
                                                    that was read in is
                           int codeSize             indicate how large the coded
                                                    message is
Function Output/Parameters: (by reference) int codeSize
Function Output/Returned: none
Device Input: none
Device Output: notifies user encryption has begun
Dependencies: none
*/
void encodeData( char message[], int encodedMessage[], int messSize, int &codeSize );


/* 
Name: doLoadEven 
Process: returns true, in this case to indicate an even number should be loaded, 
         the specified percentage of times 
Function Input/Parameters: int timesIn100     indicates the expected times out of 100
                                              true (do load an even value) is returned
Function Output/Parameters: none
Function Output/Returned: true      signals an even value should be loaded
                          false     signals an even value should not be loaded
Device Input: none
Device Output: none
Dependencies: rand()
*/
bool doLoadEven( int timesIn100 );


/* 
Name: getRandBetween
Process: generates a random number within the range of the input values
Function Input/Parameters: int low      the lower limit of the range for the 
                                        generated random number
                           int high     the upper limit of the range for the 
                                        generated random number
Function Output/Parameters: none
Function Output/Returned: a random number within the specified range
Device Input: none
Device Output: none
Dependencies: rand()
*/
int getRandBetween( int low, int high );


/* 
Name: encodeVal
Process: encodes a value that is presumably, but not necessarily, the code for an
         ASCII character
Function Input/Parameters: int value        the presumed ASCII code to be encoded
                           int tailType     indicates whether the encoded value
                                            whould end in an even or odd digit
                                            to mark it as a true or dummy value
Function Output/Parameters: none
Function Output/Returned: an encoded integer
Device Input: none
Device Output: none
Dependencies: none
*/
int encodeVal( int value, int tailType );


/* 
Name: genTail
Process: randomly generates a tail value to be used in the encoding process
Function Input/Parameters: int tailType      indicates whether the generated tail 
                                             value should be even or odd 
Function Output/Parameters: none
Function Output/Returned: an appropriate integer tail value
Device Input: none
Device Output: none
Dependencies: none
*/
int genTail( int tailType );


/* 
Name: genDummy
Process: randomly generates a value in the range of valid ASCII code
Function Input/Parameters: none     
Function Output/Parameters: none
Function Output/Returned: a random integer that is within the range of valid
                          ASCII code
Device Input: none
Device Output: none
Dependencies: none
*/
int genDummy();


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
void writeEncodedMessage( char outName[], int encodedMessage[], int codeSize );


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
void stringConversion( char tempString[], int codeVal );


////////////////////////////////////////////////////////////////////////////////
// Main Program Definition /////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int main()
   { 

   // initialize function/variables

      // vars

         // arrays and c-style strings
         char inputFile[ MAX_MESSAGE ]  = "None entered yet...";
         char outputFile[ MAX_MESSAGE ] = "None entered yet...";
         char message[ MAX_MESSAGE ];
         int encodedMessage[ MAX_ENCODED ];

         // flags
         bool fileThere  = false;
         bool fileTooLarge = true;
         int messageSize = 0;
         int codeSize = -999;

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
      encodeData( message, encodedMessage, messageSize, codeSize );

   // write encrypted message
      // function: writeEncodedMessage
      writeEncodedMessage( outputFile, encodedMessage, codeSize );

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
      cout << endl << endl << endl;

   // no return - void
   }


int readData( char inName[], char message[], bool &fileThere, bool &fileTooLarge, int &messSize )
   {
   // initialize variables and flags
   ifstream fin;
   int index = 0;
   fileThere = false;       // assume file is not there
   fileTooLarge = true;     // and it is too large

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
         while( fin.good() && index < MAX_MESSAGE )     // allows for 1001 chars to be read to   
           {                                            // determine if file is too large later on
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
      if( index < MAX_MESSAGE  )     // indicates that if 1000 characters have been read,
        {                            // index will = 1000 (999 + 1), and message[index]
        // set flag                  // will still be empty (garbage)
        fileTooLarge = false;
        }

   // terminate the c-style string with \0
   message[ index ] = NULL_CHAR;

   // store the message size for later reference
   messSize = index;     // index will be at the null terminator position, so it is a count of the chars

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


void oversizeError( char inName[] )
   {
   // display the message
      // function: cout
      cout << endl
           << "ERROR: The message found in " << inName << " is too large."
           << endl << "Rather than write a truncated message, "
           << "program will now terminate..." << endl << endl;

   // hold program for user
      // function: system
      system( "PAUSE" );
   
   // no return - void   
   }


void encodeData( char message[], int encodedMessage[], int messSize, int &codeSize )
   {
   // initialize variables
   int messIndex = 0;
   int codeIndex = 0;
   int tempVal = 999;

   // notify user encryption has begun
      // function: cout
      cout << endl << "Encrypting Data . . ." << endl << endl;

   // load the code array with loop, with both true and false values, to end of message
      // operation: while
      while( messIndex < messSize )
        {
        // decide whether to load an even, in value, or an odd dummy one
           // operation: if
           if( doLoadEven( EVEN_PROB ) )
             {
             // convert the character value into a coded even one

                // cast the character value as an integer
                tempVal = int( message[ messIndex ] );

                // finish the encoding process
                   // function: encodeVal
                   tempVal = encodeVal( message[ messIndex ], EVEN );

             // load the coded value into the code array
             encodedMessage[ codeIndex ] = tempVal;
        
             // increment message index
             messIndex ++;
             }

           // operation: else
           else
             {
             // generate an odd value to load
                // function: genDummy
                tempVal = genDummy();

             // encode the dummy
                // function: encodeVal
                tempVal = encodeVal( tempVal, ODD );

             // load the value
             encodedMessage[ codeIndex ] = tempVal;
             }

        // increment the code index
        codeIndex ++;
        }
   
   // store coded message size for later use
   codeSize = codeIndex;     // codeIndex should refer to the first invalid position in the array

   // no return - void
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


bool doLoadEven( int timesIn100 )
   {
   // initialize variables
   bool decision = true;
   int option = 0; 

   // use rand() to generate a number to use to make the decision
      // function: rand
      option = getRandBetween( 0, 99 );

   // decide whether or not to send the signal to output an even value
      // operation: if
      if( ( option >= 0 ) && ( option < timesIn100 ) )
        {
         // decision will be true to indicate that an even value should be outputted
         decision = true;    
        } 

      // operation: else if
      else if( ( option >= timesIn100 ) && ( option <= 99 ) )
        {
         // decision will be false to indicate an odd value should be outputted
         decision = false;
        }
    
   // return whatever decision was made
   return decision;
   }


int encodeVal( int value, int tailType )
   {
   // multiply value by 1000
   value *=1000;

   // add an appropriate tail
      // function: genTail
      value += genTail( tailType );

   // return the wncoded value
   return value;
   }

int genTail( int tailType )
   {
   // initialize variables
   int tail = 0;

   // generate an initial tail
           // function: getRandBetween
           tail = getRandBetween( 1, 999 );

   // check the tail. if incorrect, loop until desired tail is generated
      // operation: while
      while( ( tail % 2 ) != tailType )  
           {
           // generate a new random tail
              // function: getRandBetween
              tail = getRandBetween( 1, 999);

           // if tail is desired type, loop breaks
           }

   // return the appropriate tail
   return tail;
   }

int genDummy()
   {
   // variables
   int dummy; 

   // generate a random ASCII value
      // function: getRandBetween
      dummy = getRandBetween( ASCII_LOW, ASCII_HI );

   // return that value
   return dummy;
   }


void writeEncodedMessage( char outName[], int encodedMessage[], int codeSize )
   {
   // intitialize variables
   ofstream fout;
   char tempString[ OUT_STR_LEN ];
   int tempStrIndex = 0;
   int codeIndex = 0;
   int valOnLine = 0;  

   // notify user writing has begun
      // function: cout
      cout << endl << "Downloading Data . . ." << endl << endl << endl;

   // open the output file
      // functions: .clear, .open
      fout.clear();
      fout.open( outName );

   // write data to file using 8 character c-strings
      // operation: while
      while( codeIndex < codeSize )
        {
        // only write 5 values per line
           // operation: for
           for( valOnLine = 0; (valOnLine < MAX_VAL_PER_LINE) && (codeIndex < codeSize); 
                valOnLine ++, codeIndex ++ )
             {
             // convert integer values to string values
                // function: stringConversion
                stringConversion( tempString, encodedMessage[ codeIndex ] ); 

             // output the string value
                // operation: for
                for( tempStrIndex = 0; tempString[ tempStrIndex ] != NULL_CHAR; tempStrIndex ++ )
                  {
                  // write one character at a time
                  fout << tempString[ tempStrIndex ];
                     
                  // loop breaks befor null terminator is printed
                  }
             }
        // start new line
           // function: fout <<
           fout << NEW_LINE;

        // writing ends when last coded value is written
        }

   // close the file
      // function: .close
      fout.close();

   // no return - void 
   }


void stringConversion( char tempString[], int codeVal )
   {
   // initialize variables
   int strIndex = 0;
   int tempInt = codeVal;
   int lastDigit = 0;
   int tempChar = 'a';

   // load the temporary string with spaces

      // load the spaces
         // operation: for
         for( strIndex = 0; strIndex < (OUT_STR_LEN - 2); strIndex ++ )
         {
         // load a space
         tempString[ strIndex ] = SPACE;
         }

      // load the null terminator
      tempString[ (OUT_STR_LEN - 1) ] = NULL_CHAR;

   // use loop to input the coded value in reverse order
      // operation: for
      for( strIndex = ( OUT_STR_LEN - 2 ); tempInt > 0; strIndex -- )
        {
        // grab the last digit of the code value
           // operation: %
           lastDigit = ( tempInt % 10 );            

        // convert the digit to its ASCII character value
           // operation: char()
           tempChar = lastDigit + '0';

        // load the now character into the string
        tempString[ strIndex ] = tempChar;

        // drop the last digit for the next pass of the loop
        tempInt /= 10;     // integer math will preserve the leading digits
        }

   // no return - void
   } 
