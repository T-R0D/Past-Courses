// Header Files

   #include "formatted_console_io_v17.h" // for console I/O
    
   using namespace std;

// Global Constant Definitions

   // declare character/message string constants
   const char PIPE = '|' ;
   const string COLON = ": " ;
   const string PROG_TITLE   = "Current Calculation Program" ;
   const string TITLE_UNDERL = "===========================" ;
   const string PRESS_ANY1 = "Press any key to continue . . ." ;
   const string PRESS_ANY2 = "End of data input, press any key to continue" ;
   
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

// Function Prototypes

   // none

// Main Program
int main()
   {
    // initialize program

       // initialize variables

          // set initial x and y positions
          int xPos = 5 ; 
          int yPos = 3 ;

          // declare other variables
             // experiment/experimenter variables
             int test_number ;
             string experimenter_name ;

             // measurement variables
             double voltage ;
             double resistance ;
             double current ;


       // initialize curses
          // function: startCurses
          startCurses() ;

       // set color for initial screen
          // function: setColor
          setColor( COLOR_RED, COLOR_BLUE, SET_BRIGHT ) ;

       // clear screen
          // function: clearScreen
          clearScreen( SCRN_MIN_X, SCRN_MIN_Y, SCRN_MAX_X, SCRN_MAX_Y ) ; 

       // show title with underline
          // function: printStringAt
          printStringAt( xPos, yPos, PROG_TITLE, "LEFT" ) ;
          yPos ++ ;
          printStringAt( xPos, yPos, TITLE_UNDERL, "LEFT" ) ;


    // input data

            /* Note: for whatever reason (aesthetics, to present a challenge), the colons that come before the 
               data prompts are all lined up in a column at the right of the screen. My code will print the
               colon separately to acheive this.   */        


        // update y position and get test number from user
           // function: promptForIntAt
           yPos += 2 ;
           printStringAt( xPos, yPos, "Enter test number", "LEFT" ) ;
           test_number = promptForIntAt( COLON_X, yPos, COLON ) ;

        // update y position and get experimenter name
           // function: promptForStringAt
           yPos += 2 ;
           printStringAt( xPos, yPos, "Enter Experimenter Name", "LEFT" ) ;
           experimenter_name = promptForStringAt( COLON_X, yPos, COLON ) ;

        // update y position and get voltage from user
           // function: promptForDoubleAt
           yPos += 2 ;
           printStringAt( xPos, yPos, "Enter voltage measured (volts)", "LEFT" ) ;
           voltage = promptForDoubleAt( COLON_X, yPos, COLON ) ;

        // update y position and get resistance from user
           // function: promptForDoubleAt
           yPos += 2 ;
           printStringAt( xPos, yPos, "Enter resistance measured (ohms)", "LEFT" ) ;
           resistance = promptForDoubleAt( COLON_X, yPos, COLON ) ;

        // update y position for prompt display
         yPos += 2 ;
         
        // display initial screen hold prompt
            // function: printStringAt
            printStringAt( xPos, yPos, PRESS_ANY2, "LEFT" ) ;

        // hold screen
            // function: waitForInput
            waitForInput( FIXED_WAIT ) ;

    // calculate current
       // operation: math
       current = voltage / resistance ;

    // make display box
       
        // set background for box
           // function: setColor
           setColor( COLOR_BLUE, COLOR_WHITE, "FALSE" ) ;

        // clear screen - also sets background
           // function: clearScreen
           clearScreen( SCRN_MIN_X, SCRN_MIN_Y, SCRN_MAX_X, SCRN_MAX_Y ) ;

        // reset x and y positions for table display
        xPos = 5, yPos = 5 ;
 
        // print top line and main title
           // function: printStringAt
           printStringAt( xPos, yPos, TABLE_TOP, "LEFT" ) ;
           yPos ++ ;
           printStringAt( xPos, yPos, TABLE_TITLE, "LEFT" ) ;
           yPos ++ ;
           printStringAt( xPos, yPos, TABLE_HORIZ, "LEFT" ) ;

        // update y position and print sub titles and container box
           // function: printStringAt
           yPos ++ ;
           printStringAt( xPos, yPos, SUB_TITLES, "LEFT" ) ;
           yPos ++ ; 
           printStringAt( xPos, yPos, TABLE_HORIZ, "LEFT" ) ;  

        // update y position and print data/results box and bottom line
           // function: printStringAt
           yPos ++ ;
           printCharAt( xPos, yPos, PIPE) ;
           printCharAt( PIPE1_X, yPos, PIPE ) ;
           printCharAt( PIPE2_X, yPos, PIPE ) ;
           printCharAt( PIPE3_X, yPos, PIPE ) ;
           printCharAt( PIPE4_X, yPos, PIPE ) ; 
           printCharAt( PIPE_CLOSE_X, yPos, PIPE ) ;
           yPos ++ ;
           printStringAt( xPos, yPos, TABLE_HORIZ, "LEFT" ) ; 

    // display verified and calculated data

        // reset x and y positions for data display
        yPos -- ;

        // set display color
           // function: setColor
           setColor (COLOR_RED, COLOR_WHITE, "FALSE" ) ;
               /* Note: There is no need to clear the screen because I am not
                        overwriting the entire screen, therefore I can let 
                        functions print over the old display settings as they
                        output data.   */

        // print first data set

           // print test number
              // function: printIntAt
              xPos += 4 ;
              printIntAt( xPos, yPos, test_number, "LEFT" ) ;

           // update x position and print experimenter name
              // function: printStringAt
              xPos += 7 ;
              printStringAt( xPos, yPos, experimenter_name, "LEFT" ) ;

           // update x position and print voltage
              // function: printDoubleAt
              xPos += 25 ;
              printDoubleAt( xPos, yPos, voltage, PRECISION, "RIGHT" ) ;

           // update x position and print resistance
              // function: printDoubleAt
              xPos += 13 ;
              printDoubleAt( xPos, yPos, resistance, PRECISION, "RIGHT" ) ;             

           // update x position and print current
              // function: printDoubleAt
              xPos += 10 ;
              printDoubleAt( xPos, yPos, current, PRECISION, "RIGHT" ) ;

    // end program

       // prompt user for quit and hold screen

          // update x and y positions for hold screen prompt
          xPos -= 31 , yPos += 4 ;
 
          // set color for prompt
             // function: setColor 
                /* Not sure why, we are already printing in red, but will write
                   the code for it anyway... */
             setColor (COLOR_RED, COLOR_WHITE, "FALSE" ) ;

          // print prompt string
             // function: printStringAt
             printStringAt( xPos, yPos, PRESS_ANY1, "LEFT" ) ;

          // hold screen
             // function: waitForInput
             waitForInput( FIXED_WAIT ) ;

       // shut down curses
          // function: endCurses
          endCurses() ;

       // return success to OS
       return 0;
   }

// Supporting Function Implementation

   // none


