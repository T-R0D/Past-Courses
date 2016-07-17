// Header Files
   #include "formatted_cmdline_io_v08.h"
   #include <cmath> 
   using namespace std ;

// Global Constant Definitions

   // spacing constants

   // calculation constants

   // constant strings and character
   
   // precision constants


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
Function Input/Parameters: percent of frequency of relevance (int) 
                           frequency of relevance (double)
                           inductance (double)
Function Output/Parameters: none
Function Output/Returned: calculated result for capacitance(double)
Device Input: none
Device Output: none
Dependencies: cmath library
*/
double calcCapacitance( int pct_relev, double frequency, double inductance ) ;

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
void printResultString( string item_name, string &system_name ) ;

/* 
Name: printResultData
Process: prints the results of input data and calculated relevance values
Function Input/Parameters: inductance (double)
                           frequency (double) 
                           capacitance of lower bound of relevance (double) 
                           capacitance of ideal relevance (double) 
                           capacitance of upper bound of relevance (double)
Function Output/Parameters: none
Function Output/Returned: none 
Device Input: none
Device Output: displays the numerical results of the output
Dependencies: formatted command line I/O tools
*/
void printResultData( double frequency, double inductance, double capacitance_lo, 
                     double capacitance_ideal, double capacitance_hi ) ;




// Main Program Definition

int main()
   { 

   // initialize program

      // initialize variables

      // print program title
         // function: printTitle

   // prompt for data from user

      // collect system name
         // function: printString, promptForString, printEndLines

      // collect frequency
         // function: printString, promptForDouble, printEndLines
 
      // collect inductance
         // function: printString, promptForDouble, printEndLines

      // hold screen before outputting data
         // functions: printString, system

   // calculate capacitance at % relevance

      // capacitance at 50% relevance
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
  
      // return 0
      return 0;

   }



// Supporting function implementations

void printTitle()
   {
  

   // no return, void
   }


double calcCapacitance( int pct_relev, double frequency, double inductance )
   {
  
      
   return 1.56 ; // temporary stub return
   }


void displayResultHeader()
   {
 
   // no return, void
   }


void printResultString( string item_name, string &system_name )
   {


   // no return, void
   }


void printResultData( double frequency, double inductance, double capacitance_lo, 
                     double capacitance_ideal, double capacitance_hi )
   {
 

   // no return, void
   }



