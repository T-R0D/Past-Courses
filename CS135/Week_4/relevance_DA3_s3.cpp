// Header Files
   #include "formatted_cmdline_io_v08.h"
   #include <cmath> 
   using namespace std ;

// Global Constant Definitions

    // none

// Global Function Prototypes

/* 
Name: printTitle 
Process: prints the program title and its underline
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: title displayed
Dependencies: formatted command line I/O tools
*/
void printTitle() ;

/* 
Name: calcRelevance
Process: uses frequency, inductance, and percent data to calculate capacitance 
Function Input/Parameters: percent of frequncy of relevance (int) and 
                           frequency of relevance and inductance (double)
Function Output/Parameters: none
Function Output/Returned: capacitance
Device Input: none
Device Output: none
Dependencies: cmath library
*/
double calcRelevance( int pct_relev, double frequency, double inductance ) ;

/* 
Name: displayResultHeader
Process: displays a title and underline to begin the display of results
Function Input/Parameters: none
Function Output/Parameters: none
Function Output/Returned: none 
Device Input: none
Device Output: results title displayed
Dependencies: formatted command line I/O tools
*/
void displayResultHeader() ;

/* 
Name: printResultString
Process: accepts the user input system name and displays the system name line
         of the output screen 
Function Input/Parameters: system_name (string)
Function Output/Parameters: none
Function Output/Returned: none
Device Input: none
Device Output: system name result displayed
Dependencies: formatted command line I/O tools
*/
void printResultString( string &system_name ) ;

/* 
Name: printResultData
Process: prints the results of input data and calculated relevance values
Function Input/Parameters: inductance (double), frequency (double), 
                           relev_low (double), relev_ideal (double), 
                           relev_hi (double)
Function Output/Parameters: none
Function Output/Returned: none 
Device Input: none
Device Output: displays the numerical results of the output
Dependencies: formatted command line I/O tools
*/
void printResultData() ;


// Main Program Definition

int main()
   { 

   // initialize program

      // initialize variables

      // print program title
         // function: printTitle

   // prompt for data from user

      // collect system name
         // function: promptForString

      // collect frequency
         // function: promptForDouble

      // collect inductance
         // function: promptForDouble

      // hold screen before outputting data
         // function: system
 
   // calculate relevance

      // 50% relevance
         // function: calcRelevance

      // 100% relevance
         // function: calcRelevance

      // 150% relevance
         // function: calcRelevance

   // output data

      // print output title
         // function: displayResultHeader

      // print system name
         // function: printResultString

      // print numerical results
         // function: printResultData

   // end program 

      // hold program before terminating
      system ( "PAUSE" );
      // return 0
      return 0;

   }

// Supporting function implementations

void printTitle()
   {



   // no return, void
   }


double calcRelevance( int pct_relev, double frequency, double inductance )
   {



   return 0 ; // temporary stub return
   }


void displayResultHeader()
   {



   // no return, void
   }


void printResultString( string &system_name )
   {



   // no return, void
   }


void printResultData()
   {



   // no return, void
   }



