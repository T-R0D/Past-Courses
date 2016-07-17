// Header Files

   // none

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

         // declare display settings

         // clear screen

      // print program title

         // print title at desired xy-location
        
   // prompt for input

      /* Note: for whatever reason (aesthetics, to present a challenge), the colons that come before the 
               data prompts are all lined up in a column at the right of the screen. My code will print the
               colons separately from the prompt messages to acheive this.   */        

      // collect test number

      // collect experimenter name

      // collect voltage value

      // collect resistance value

      // hold before screen change

   // calulate current 

      // (V=IR --> I=V/R)

   // output results

      // change display settings

         // re-initialize print location x and y variables          

         // redeclare display colors

         // clear screen

      // print table

         // print first top table border

         // print table title box

         // print subtitles box(es)

         // print data row

      // print data
         
         // change text color

               /* Note: There is no need to clear the screen because I am not
                        overwriting the entire screen, therefore I can let 
                        functions print over the old display settings as they
                        output data.   */

         // print test number
  
         // print experimenter name

         // print voltage
               
         // print resistance

         // print current

   // end program 

      // hold the screen

      // shut down curses

      // return zero        
         return 0;
   
   }

// Supporting function implementations

   // none



