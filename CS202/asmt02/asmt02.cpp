/////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      asmt02: Database Entry Generator
//  Created By: Terence Henriod
//  Course:     CS 202
//
//  This program generates database entries using random number generation and
//  random entries.
//
/////////////////////////////////////////////////////////////////////////////////


/// Header Files/////////////////////////////////////////////////////////////////

// standard headers
#include <iostream>
#include <fstream>
#include <conio.h>
#include <stdlib.h>
#include "Ttools.h"
#include <conio.h>
using namespace std;

// assignment specific headers
#include "city.hpp"
#include "name_l.hpp"
#include "name_f_f.hpp"
#include "name_f_m.hpp"
#include "title.hpp"
#include "salary.hpp"
#include "street_lv.hpp"
#include "street_r.hpp"


/// Macros //////////////////////////////////////////////////////////////////////
#define MESSAGEFLAG 1


/// Global Constant Definitions//////////////////////////////////////////////////

// string lengths
const int STRING_L    = 50;   // used for short strings
const int HEAD_L      = 80;  

// strings
const char DOTCSV[STRING_L] = ".csv";    // file extension
const char HELP[STRING_L] = "/?";      // help prompt
const char HEADINGS[HEAD_L] = "EmpID,Last,First,MI,Gender,Age,Title,Salary,Address,City";

// chars
const char DELIM = ',';

// help (exit) codes
const int HELP_REQ     = 1;
const int HELP_NUMARGS = 2;
const int HELP_NUMRECS = 3;

/// Global Function Prototypes///////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
//
// Function Name: exit_function
// Summary:       Exits the program after displaying an error message appropriate
//                to the situation.
// Parameters:    int code   The predefined non-zero integer exit code.
// Returns:       void
//
///////////////////////////////////////////////////////////////////////////////// 
void dispHelp( int code );
void createOutput( int numRecs, char outf[STRING_L] );
char getGender();
char getCityLet();
void createAddress( char address[STRING_L], char cityLet, int RstreetS, int LVstreetS );


/// Main Program Definition//////////////////////////////////////////////////////
int main( int argc, char* argv[] )
   { 
   // vars
   char outf[STRING_L];              // argument 1 storage
     // clear the array
     int ndx = 1;
     outf[0] = '\0';
     while(ndx < STRING_L)
       {
       outf[ndx] = ' ';
       ndx ++;
       }
    
	 // assign given value
     strcpy(outf, argv[1]);   

   int numRecs = atoi( argv[2] );              // argument 2 storage
   int seed;                                   // optional argument 3 storage
   int code = 0;                               // help code   

   // manage cmdline arguments
   if( (strcmp( argv[1], HELP ) == 0) )   // display help immediately, it has been requested
     {
     code = HELP_REQ;
     dispHelp(code);
     }
   else if( (argc < 3) || (argc > 4) )    // bad input cases
     {                                    
     code = HELP_NUMARGS;
     }
   else if( numRecs < 1 )
     {
     code = HELP_NUMRECS;
     }

   // if necessary display help and exit
   if( code > 0 )
     {     
     dispHelp( code );
     }    

   // add extenstion to outfile name argument
   strcpy( outf, argv[1] );
   strcat( outf , DOTCSV );

   // if 3rd argument exists, attempt to use it as rngseed
   if( argc == 4 )
     {
     seed = atoi( argv[3] );   // convert entry to int
     if( seed < 0 )            // if negative, use system clock
       {  
       srand( time(NULL) );
       }  
     else                      // otherwise, use given argument
       {
       srand( seed );
       }
     }
   else                        // seed rng normally if 3rd argument is not present
     {
     srand( time(NULL) );
     }
 
   // create output
   createOutput(numRecs, outf);

   // end program
   return 0;
   }


/// Supporting function implementations//////////////////////////////////////////

void dispHelp( int code )
   {
   #if MESSAGEFLAG
   clrScr();

   cout << "               WELCOME TO THE HELP SCREEN" << endl
        << endl << endl;

   // display error appropriate message
   switch( code )
     {
     case HELP_REQ:
       {
       cout << "Here is the information you requested:"
            << endl << endl;
       }
       break;
     case HELP_NUMARGS:
       {
       cout << "You have entered an invalid number of command line arguments."
            << endl
            << "Please read the information below and try again:"
            << endl << endl;
       }
       break;
     case HELP_NUMRECS:
       {
       cout << "You have entered an invalid request for the number of randomly"
            << endl
            << "generated records. A positive integer must be entered."
            << endl << endl;
       }
       break;
     }

   // display help information
   cout << "This program generates imaginary employee records from lists of"
        << endl
        << "various field entries that are randomly selected. The program"
        << endl
        << "accepts the following command line arguments:" << endl
        << "  1. The name of the output file that will contain the records"
        << endl
        << "(the .csv file extension is not necessary, it will be added)."
        << endl
        << "  2. The number of imaginary records to be generated (must be"
        << endl
        << "a positive integer)." << endl
        << "  3. [optional] The number used to seed the random number"
        << endl 
        << "generator (not of particular concern to the user)." << endl;

   holdProg();
   
   #endif

   // no return, but do exit prog
   exit(code);
   }

void createOutput( int numRecs, char outf[STRING_L] )
   {
   // vars
   int counter = 1;
   ofstream fout;
   char gender = 'M';
   char cityLet = 'R';
   int pick = 0;
   /* I know that the compiler may or may not optimize these functions/macros
   out of existence, replacing the function/macro call with the appropriate 
   value at compile time, but I stored the values to variables for the sake
   of good practice */
   int cityNameS = ARR_SIZE( city_names );
   int FFnameS   = ARR_SIZE( names_female );
   int MFnameS   = ARR_SIZE( names_male );
   int LnameS    = ARR_SIZE( names_last );  
   int salaryS   = ARR_SIZE( salary );
   int LVstreetS = ARR_SIZE( street_names_las_vegas );
   int RstreetS  = ARR_SIZE( street_names_reno );
   int titleS    = ARR_SIZE( job_title );

   char address[STRING_L];
     // clear the array
     int ndx = 1;
     address[0] = '\0';
     while(ndx < STRING_L)
       {
       address[ndx] = ' ';
       ndx ++;
       }

   // clear and open
   fout.clear();
   fout.open( outf );

   // write file headings
   fout << HEADINGS << endl;

   // generate the number of records requested
   while( counter <= numRecs )
     {
     // EmpID - Seqential
     fout << counter << DELIM;
     counter ++;

     // decide gender and city first
     gender = getGender();
     cityLet = getCityLet();

     // Last name
     pick = randBetw(0, (LnameS - 1));   // -1 to change a count to an index position
     fout << names_last[pick] << DELIM;

     // First name 
     if( gender == 'M' )
       {
       pick = randBetw(0, (MFnameS - 1));
       fout << names_male[pick] << DELIM;
       }
     else if(gender == 'F')
       {
       pick = randBetw(0, (FFnameS - 1));
       fout << names_female[pick] << DELIM;
       }     

     // MI
     fout << char( randBetw(int('A'), int('Z')) ) << DELIM;

     // gender
     fout << gender << DELIM;

     // age
     fout << randBetw(18, 70) << DELIM;

     // title
     pick = randBetw(0, (titleS - 1));
     fout << job_title[pick] << DELIM;

     // salary
     pick = randBetw(0, (salaryS - 1));
     fout << salary[pick] << DELIM;

     // address (number and street)
     createAddress( address, cityLet, RstreetS, LVstreetS);
     fout << address << DELIM;

     // city
     if( cityLet == 'R' )
       {
       fout << city_names[1];
       }
     else if( cityLet == 'L' )
       {
       fout << city_names[0];
       }     

     fout << endl;
     } 
   fout << endl;

   // close
   fout.close();

   // no return - void
   }

char getGender()
   {
   // vars
   int number;
   char gender = 'M';

   number = randBetw(0, 1);
 
   switch( number )
     {
     case 0:
       {
       gender = 'F';
       }
       break;   
     case 1:
       {
       gender = 'M';
       }
       break;     
     }

   return gender;
   }

char getCityLet()
   {
   // vars
   int number;
   char letter = 'R';

   number = randBetw(0, 1);
 
   switch( number )
     {
     case 0:
       {
       letter = 'R';
       }
       break;   
     case 1:
       {
       letter = 'L';
       }
       break;     
     }

   return letter;
   }

void createAddress( char address[STRING_L], char cityLet, int RstreetS, int LVstreetS )
   {
   // vars
   int addNumL;
   int ndx;
   int pick;

   // clear address string
   address[0] = '\0';
   for(ndx = 1; ndx < (STRING_L - 1); ndx ++ );    
      {
      address[ndx] = ' ';
      }

   // make number
   addNumL = randBetw(3, 5);
   ndx = 0;

   address[ndx] = (char(randBetw(1, 2)) + '0');
   ndx ++;

   while( ndx < addNumL )
     {
     address[ndx] = (char(randBetw(0, 9)) + '0');
     ndx ++;
     }

   // put a space
   address[ndx] = ' ';  
   ndx ++;
   address[ndx] = '\0';

   // add street name
   if( cityLet == 'R' )
     {
     pick = randBetw(0, (RstreetS - 1));
     strcat( address, street_names_reno[pick] );
     }
   else if( cityLet == 'L' )
     {
     pick = randBetw(0, (LVstreetS - 1));
     strcat( address, street_names_las_vegas[pick] );
     }
   
   // no return - void
   }


