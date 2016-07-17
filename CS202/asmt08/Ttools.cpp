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

bool checkExt( char* fname, const char* extension )
   {
   // vars
   bool isValid = false;
   int fLen = strlen(fname);
   int extLen = strlen(extension);
   char* candidate;
   int i = (fLen - extLen);  
   
   // test similarity in reverse order
   candidate = strstr(fname, extension);

   if(strcmp((fname + i), candidate) == 0)
     {
     isValid = true;   // assumes that the extension would only appear
     }                 // once (at the end) of the file name

   // return test result
   return isValid;
   }

bool openFile( const char* fname, ifstream &fobj )
   {
   // vars 
   bool success = true;

   // clear and open
   fobj.clear();
   fobj.open( fname );

   // check for file opening success
   if( !fobj.good() )
     {
     success = false;
     }   

   // return success state
   return success;
   }

bool openFile( const string fname, ifstream &fobj )
   {
   // vars 
   bool success = true;

   // clear and open
   fobj.clear();
   fobj.open(fname.c_str());

   // check for file opening success
   if( !fobj.good() )
     {
     success = false;
     }   

   // return success state
   return success;
   }

bool openFile( const char* fname, ofstream &fobj )
   {
   // vars 
   bool success = false;

   // clear and open
   fobj.clear();
   fobj.open( fname );

   // check for file opening success
   if( fobj.good() )
     {
     success = true;
     }   

   // return success state
   return success;
   }

bool openFile( const string fname, ofstream &fobj )
   {
   // vars 
   bool success = false;

   // clear and open
   fobj.clear();
   fobj.open(fname.c_str());

   // check for file opening success
   if( fobj.good() )
     {
     success = true;
     }   

   // return success state
   return success;
   }

void closeFile( ifstream &fobj )
   {
   // close the file
   fobj.close();

   // no return - void
   }

void closeFile( ofstream &fobj )
   {
   // close the file
   fobj.close();

   // no return - void
   }