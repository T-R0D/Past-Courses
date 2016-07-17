#include "Ttools.h"

void holdProg()
   {
   // make space
   cout << endl << endl;

   // display message
   cout << "Press any key to continue. . .";

   // wait for key hit
   while( ! _kbhit )
     {
     // do nothing
     }

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


int randBetw( int lo, int hi )
   {
   // vars
   int range = (hi - lo + 1);
 
   // return random val
   return ( (rand() % range) + lo );
   }
