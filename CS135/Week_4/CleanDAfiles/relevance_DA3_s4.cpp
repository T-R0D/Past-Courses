// Header Files
   #include "formatted_cmdline_io_v08.h"
   #include <cmath> 
   using namespace std ;

// Global Constant Definitions

   // spacing constants
   const int NO_BLOCK_SIZE = 0  ;
   const int PROMPT_BLOCK  = 43 ;
   const int OUT_BLOCK     = 40 ; 
   const int ONE_LINE      = 1  ;
   const int TWO_LINES     = 2  ;

   // calculation constants
   const double PI       = 3.14159265359 ;
   const int RELEV_LO    = 50  ;
   const int RELEV_IDEAL = 100 ;
   const int RELEV_HI    = 150 ;

   // constant strings and character
   const string COLON = ": " ;  

   // precision constants
   const int FREQ_IND_PREC = 2 ;
   const int CAPACIT_PREC  = 4 ;


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
      string system_name ;
      double frequency ;
      double inductance ;
      int pct_relev ;
      double capacitance_lo ;
      double capacitance_ideal ;
      double capacitance_hi ;

      // print program title
         // function: printTitle
         printTitle() ; 

   // prompt for data from user

      // collect system name
         // function: printString, promptForString, printEndLines
         printString( "Enter system name", PROMPT_BLOCK, "LEFT" ) ;
         system_name = promptForString( COLON ) ;
         printEndLines( TWO_LINES ) ;

      // collect frequency
         // function: printString, promptForDouble, printEndLines
         printString( "Enter frequency of system            (Hertz) ", PROMPT_BLOCK, "LEFT" ) ;
         frequency = promptForDouble( COLON ) ;
         printEndLines( TWO_LINES ) ;
 
      // collect inductance
         // function: printString, promptForDouble, printEndLines
         printString( "Enter inductance of system          (Henrys) ", PROMPT_BLOCK, "LEFT" ) ;
         inductance = promptForDouble( COLON ) ;
         printEndLines( TWO_LINES ) ;

      // hold screen before outputting data
         // functions: printString, system
         printString( "End of data input, ", NO_BLOCK_SIZE, "LEFT" ) ;
         system( "PAUSE" ) ;

   // calculate capacitance at % relevance

      // capacitance at 50% relevance
         // function: calcRelevance
         pct_relev = RELEV_LO ;
         capacitance_lo = calcCapacitance( pct_relev, frequency, inductance ) ; 

      // 100% relevance
         // function: calcRelevance
         pct_relev = RELEV_IDEAL ;
         capacitance_ideal = calcCapacitance( pct_relev, frequency, inductance ) ; 

      // 150% relevance
         // function: calcRelevance
         pct_relev = RELEV_HI ;
         capacitance_hi = calcCapacitance( pct_relev, frequency, inductance ) ; 

   // output data

      // print output title
         // function: displayResultHeader
         displayResultHeader() ;

      // print system name
         // function: printResultString
         printResultString( "System Name", system_name ) ;

      // print numerical results
         // function: printResultData
         printResultData( frequency, inductance, capacitance_lo, capacitance_ideal, capacitance_hi ) ;

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



