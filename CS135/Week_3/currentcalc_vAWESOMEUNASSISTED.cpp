// Header Files

   // initialize formatted_console_io_v17.h header file
   #include "formatted_console_io_v17.h"

// Global Constant Definitions

   // declare character/message string constants
   const char PIPE = '|' ;
   const string COLON = ": " ;
   const string PROG_TITLE   = "Current Calculation Program" ;
   const string TITLE_UNDERL = "===========================" ;
   const string PRESS_ANY1 = "Press any key to continue..." ;
   const string PRESS_ANY2 = "End of data input, press any key to continue . . ." ;
   
   // declare table component string constants
   const string TABLE_TOP   = "|------------------------------------------------------------|" ;
   const string TABLE_HORIZ = "|--------|---------------|------------|------------|---------|" ;
   const string TABLE_BOT   = "|------------------------------------------------------------|" ;
   const string TABLE_TITLE = "|                      DATA PRESENTATION                     |" ;
   const string SUB_TITLES  = "|  Test  |  Experimenter |   Voltage  | Resistance | Current |" ;

   // declare any necessary numerical information constants 
   const int PRECISION = 2 ;  

   // declare spacing contstants
   const int TITLE_BLOCK = 60 ;

   // declare any X- or Y-position constants
      // Note: these values all include a "+5" in order to match them to the default x starting position
   const int COLON_X       = 42 ;
   const int TEST_VAL_X    = 9  ;
   const int PIPE1_X       = 14 ;
   const int NAME_VAL_X    = 16 ;
   const int PIPE2_X       = 30 ;
   const int VOLT_VAL_X    = 41 ;
   const int PIPE3_X       = 43 ;  
   const int RESIST_VAL_X  = 54 ;
   const int PIPE4_X       = 56 ;
   const int CURRENT_VAL_X = 64 ;
   const int PIPE_CLOSE_X  = 66 ;

// Global Function Prototypes

   // none

// Main Program Definition

   int main()
   {

   // initialize program/function
      
      // initialize variables

         // initialize screen display variables
         int xPos = 5 ; 
         int yPos = 5 ;

         // experiment/experimenter variables
         int test_number ;
         string experimenter_name ;

         // measurement variables
         double voltage ;
         double resistance ;
         double current ;

      // initialize display and settings

         // initialize curses
            // function: startCurses   
            startCurses() ;

         // declare display settings
            // function: setColor
            setColor( COLOR_RED, COLOR_BLUE, SET_BRIGHT ) ;

         // clear screen
            // function: clearScreen
            clearScreen( SCRN_MIN_X, SCRN_MIN_Y, SCRN_MAX_X, SCRN_MAX_Y ) ; 

      // print program title

         // print title at desired xy-location
            // function: printStringAt
            xPos = 5 ; 
            yPos = 3 ;
            printStringAt( xPos, yPos, PROG_TITLE, "LEFT" ) ;
            yPos += 1 ;
            printStringAt( xPos, yPos, TITLE_UNDERL, "LEFT" ) ;

   // prompt for input

      /* Note: for whatever reason (aesthetics, to present a challenge), the colons that come before the 
               data prompts are all lined up in a column at the right of the screen. My code will print the
               separately to acheive this.   */        

      // collect test number
         // function: promptForIntAt
         yPos += 2 ;
         printStringAt( xPos, yPos, "Enter test number", "LEFT" ) ;
         test_number = promptForIntAt( COLON_X, yPos, COLON ) ;

      // collect experimenter name
         // function: promptForStringAt
         yPos += 2 ;
         printStringAt( xPos, yPos, "Enter Experimenter Name", "LEFT" ) ;
         experimenter_name = promptForStringAt( COLON_X, yPos, COLON ) ;

      // collect voltage value
         // function: promptForDoubleAt
         yPos += 2 ;
         printStringAt( xPos, yPos, "Enter voltage measured (volts)", "LEFT" ) ;
         voltage = promptForDoubleAt( COLON_X, yPos, COLON ) ;

      // collect resistance value
         // function: promptForDoubleAt
         yPos += 2 ;
         printStringAt( xPos, yPos, "Enter resistance measured (ohms)", "LEFT" ) ;
         resistance = promptForDoubleAt( COLON_X, yPos, COLON ) ;

      // hold before screen change
         // function: printStringAt, waitForInput
         yPos += 2 ;
         printStringAt( xPos, yPos, PRESS_ANY2, "LEFT" ) ;
         waitForInput( FIXED_WAIT ) ;

   // calulate current 

      // (V=IR --> I=V/R)
         // function: math
         current = voltage / resistance ;

   // output results

      // change display settings

         // re-initialize print location x and y variables          
         xPos = 5, yPos = 5 ;
         // redeclare display colors
            // function: setColor
            setColor( COLOR_BLUE, COLOR_WHITE, "FALSE" ) ;

         // clear screen
            // function: clearScreen
            clearScreen( SCRN_MIN_X, SCRN_MIN_Y, SCRN_MAX_X, SCRN_MAX_Y ) ;

      // print table

         // print first top table border
            // function: printStringAt
            printStringAt( xPos, yPos, TABLE_TOP, "LEFT" ) ;

         // print table title box
            // function: printStringAt
            yPos ++ ;
            printStringAt( xPos, yPos, TABLE_TITLE, "LEFT" ) ;
            yPos ++ ;
            printStringAt( xPos, yPos, TABLE_HORIZ, "LEFT" ) ;

         // print subtitles box(es)
            // function: printStringAt
            yPos ++ ;
            printStringAt( xPos, yPos, SUB_TITLES, "LEFT" ) ;
            yPos ++ ; 
            printStringAt( xPos, yPos, TABLE_HORIZ, "LEFT" ) ; 

         // print data row
            // functions: printCharAt, printStringAt
            yPos ++ ;
            printCharAt( xPos, yPos, PIPE) ;
            printCharAt( PIPE1_X, yPos, PIPE ) ;
            printCharAt( PIPE2_X, yPos, PIPE ) ;
            printCharAt( PIPE3_X, yPos, PIPE ) ;
            printCharAt( PIPE4_X, yPos, PIPE ) ; 
            printCharAt( PIPE_CLOSE_X, yPos, PIPE ) ;
            yPos ++ ;
            printStringAt( xPos, yPos, TABLE_BOT, "LEFT" ) ; 

      // print data

      yPos -- ;

         // functions: printIntAt, printStringAt, printDoubleAt, setColor, clearScreen
         
         // change text color

            // redeclare display settings
            setColor (COLOR_RED, COLOR_WHITE, "FALSE" ) ;
               /* Note: There is no need to clear the screen because I am not
                        overwriting the entire screen, therefore I can let 
                        functions print over the old display settings as they
                        output data.   */

         // print test number
         printIntAt( TEST_VAL_X, yPos, test_number, "LEFT" ) ;
  
         // print experimenter name
         printStringAt( NAME_VAL_X, yPos, experimenter_name, "LEFT" ) ;

         // print voltage
         printDoubleAt( VOLT_VAL_X, yPos, voltage, PRECISION, "RIGHT" ) ;
               
         // print resistance
         printDoubleAt( RESIST_VAL_X, yPos, resistance, PRECISION, "RIGHT" ) ;

         // print current
         printDoubleAt( CURRENT_VAL_X, yPos, current, PRECISION, "RIGHT" ) ;

   // end program 

      // hold the screen
         // functions: printStringAt, waitForInput
         xPos += 28, yPos +=4 ;
         printStringAt( xPos, yPos, PRESS_ANY1, "LEFT" ) ;
         waitForInput( FIXED_WAIT ) ;

      // shut down curses
         // function: endCurses
         endCurses() ;

      // return zero        
         return 0;
   
   }

// Supporting function implementations

   // none



