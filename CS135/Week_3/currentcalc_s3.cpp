// Header Files

   // initialize formatted_console_io_v17.h header file
   #include "formatted_console_io_v17.h"

// Global Constant Definitions

   // declare character/message string constants
   
   // declare table component string constants

   // declare any necessary numerical information constants 

   // declare spacing contstants

   // declare any X- or Y-position constants
      // Note: these values all include a "+5" in order to match them to the default x starting position

// Global Function Prototypes

   // none

// Main Program Definition

   int main()
   {

   // initialize program/function
      
      // initialize variables

         // initialize screen display variables

         // experiment/experimenter variables

         // measurement variables

      // initialize display and settings

         // initialize curses
            // function: startCurses  

         // declare display settings
            // function: setColor

         // clear screen
            // function: clearScreen

      // print program title

         // print title at desired xy-location
            // function: printStringAt

   // prompt for input

      /* Note: for whatever reason (aesthetics, to present a challenge), the colons that come before the 
               data prompts are all lined up in a column at the right of the screen. My code will print the
               colons separately from the prompt messages to acheive this.   */        

      // collect test number
         // function: promptForIntAt

      // collect experimenter name
         // function: promptForStringAt

      // collect voltage value
         // function: promptForDoubleAt

      // collect resistance value
         // function: promptForDoubleAt

      // hold before screen change
         // function: printStringAt, waitForInput

   // calulate current 

      // (V=IR --> I=V/R)
         // operation: math

   // output results

      // change display settings

         // re-initialize print location x and y variables          

         // redeclare display colors
            // function: setColor

         // clear screen
            // function: clearScreen

      // print table

         // print first top table border
            // function: printStringAt

         // print table title box
            // function: printStringAt

         // print subtitles box(es)
            // function: printStringAt

         // print data row
            // functions: printCharAt, printStringAt

      // print data
         
         // change text color
            //function: setColor

               /* Note: There is no need to clear the screen because I am not
                        overwriting the entire screen, therefore I can let 
                        functions print over the old display settings as they
                        output data.   */

         // print test number
            // function: printIntAt
  
         // print experimenter name
            // function: printStringAt

         // print voltage
            // function: printDoubleAt
               
         // print resistance
            // function: printDoubleAt

         // print current
            // function: printDoubleAt

   // end program 

      // hold the screen
         // functions: printStringAt, waitForInput

      // shut down curses
         // function: endCurses

      // return zero        
         return 0;
   
   }

// Supporting function implementations

   // none



