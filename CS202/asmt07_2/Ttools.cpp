#include "Ttools.h"

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//~ "system()" replacers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
void holdProg()
   {
   // vars
   char dummy = ' ';

   // make space
   cout << endl << endl;

   // display message
   cout << "Press any key to continue. . .";

   // wait for key hit
   dummy = getch();

   // make more space
   cout << endl << endl;

   // no return - void
   }


void clrScr()
   {
   // vars
   int n = 0;
 
   // clear screen with endl
   while( n < 50 )
     {
     cout << endl;
     n ++;
     } 

   // no return - void
   }


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//~ Miscelaneous ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

int randBetw( int lo, int hi )
   {
   // vars
   int range = (hi - lo + 1);
 
   // return random val
   return ( (rand() % range) + lo );
   }
