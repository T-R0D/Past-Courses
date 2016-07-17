////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      asmt05: Drink Machine Simulator
//  Created By: Terence Henriod
//  Course:     CS202
//
//  Summary:    This program simulates a typical vending machine. It displays a 
//              menu of beverages, and tracks its inventory. It also has the 
//              added functionality of allowing a "technician" to load
//              inventory data, as well as save it, via binary files.  
//
////////////////////////////////////////////////////////////////////////////////

//============================================================================//
//= Header Files =============================================================//
//============================================================================//

// standard headers
#include <iostream>
#include <fstream>
#include <conio.h>
#include <stdlib.h>
#include <assert.h>
#include "Ttools.h"
using namespace std;


#include "drink_list.hpp"
#include "price_list.hpp"


//============================================================================//
//= Macros ===================================================================//
//============================================================================//

#pragma warning (disable : 4996)   // disables strcpy, strcat warnings in Visual
#pragma warning (disable : 4244)   // disables time(NULL) warnings in Visual


//============================================================================//
//= Global Constant Definitions ==============================================//
//============================================================================//
 
// integers
enum exitC
   {
   REQUEST = 1,
   NUM_ARGS,
   BAD_NAME,
   BAD_FILE
   };

const int SELECTION_SIZE = 8;
const int NAME_LEN = 15;
const int CAPACITY = 2;

// strings
const char* HELP = "/?";
const char* DOTDAT = ".dat";


//============================================================================//
//= Structure Definitions ====================================================//
//============================================================================//
struct beverage
   {
   char name[NAME_LEN];
   double price;
   int onHand;
   };

//============================================================================//
//= Class Definitions ========================================================//
//============================================================================//


//============================================================================//
//= Global Function Prototypes ===============================================//
//============================================================================//

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: 
// Summary:       
// Parameters:    
// Returns: 
//
//////////////////////////////////////////////////////////////////////////////// 
void checkArgs( int argc, char** argv );

bool checkExt( char* fname, const char* extension );

double runMachine( beverage inventory[SELECTION_SIZE], char* outF, char* inF );

void dispMenu( beverage inventory[SELECTION_SIZE] );

void loadMachine( beverage inventory[SELECTION_SIZE] );

void loadData( char* inF, beverage inventory[SELECTION_SIZE] );

void saveData( char* outF, beverage inventory[SELECTION_SIZE] );

bool fileMissing( char* saveF );

double dispenseDrink( beverage* choice );

double getMoney( beverage* choice );

void dispHelp( int code );


//============================================================================//
//= Main Program Definition ==================================================//
//============================================================================//
int main(int argc, char** argv )
   { 
   // vars
   int outLen = 0;
   int inLen = 0;
   char* outF = 0;
   char* inF = 0;

   beverage inventory[SELECTION_SIZE]; 
   
   // check cmdline args, take appropriate action
   checkArgs(argc, argv);

   outLen = strlen( *(argv + 1));
   outF = new char [outLen + 1]; // +1 for '\0'
     assert(outF != 0);
     strcpy(outF, *(argv + 1));
   inLen = strlen( *(argv + 2));
   inF = new char [inLen + 1];
     assert(inF != 0);
     strcpy(inF, *(argv + 2));  

   // run machine
   runMachine(inventory, outF, inF);

   // write save data for next time machine is run
   saveData(outF, inventory); 

   // end program
  delete [] outF;                
  delete [] inF;

   return 0;
   }

//============================================================================//
//= Supporting function implementations ======================================//
//============================================================================//

void checkArgs( int argc, char** argv )
   { 
   // check to see if number of cmd line arguments is in appropriate range
   if(argc > 3)
     {
     dispHelp(NUM_ARGS);
     }

   // if help was requested, display it
   if(strcmp( *(argv + 1), HELP) == 0)
     {
     dispHelp(REQUEST);
     }

   // check the file names
     // output file
     if( !checkExt( *(argv + 1), DOTDAT) )
       {
       dispHelp(BAD_NAME);
       }     

     // input file
     if( !checkExt( *(argv + 1), DOTDAT) )
       {
       dispHelp(BAD_NAME);
       }   

   // no return - void 
   }

bool checkExt( char* fname, const char* extension )
   {
   // vars
   bool isValid = false;
   int fLen = strlen(fname);
   int extLen = strlen(DOTDAT);
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

double runMachine( beverage inventory[SELECTION_SIZE], char* outF, char* inF )
   {
   // vars
   bool contMenu = true;

   char selection = '=';

   beverage* choice;

   int button = 0;
   int i = 0;

   double take = 0.00;

   // load the machine as though full
   loadMachine(inventory);

   // implement menu
   while(contMenu)
     {
     // display menu text
     dispMenu(inventory);
     
     // get selection
     cout << "Please select an option: "; 
     selection = getch();

     // take appropriate action
     switch(selection)
       {
       case '1':
       case '2':
       case '3':
       case '4':
       case '5':
       case '6':
       case '7':
       case '8':
         {
         // convert selection to number
         button = int(selection - '0');
         choice = &(inventory[button - 1]);       
 
         // collect money, serve drink (if in stock)
         take += dispenseDrink(choice);          
         } break;

       case 'L':
       case 'l':
         {
         loadData(inF, inventory);
         } break;

       case 'S':
       case 's':
         {
         saveData(outF, inventory);
         } break;

       case 'T':
       case 't':
         {
         // set flag to stop running
         contMenu = false;

         // display the take
         printf("\n \nThe machine collected $%3.2f today.", take);
         holdProg();
         } break;
       }

     selection = '=';   // refresh buffer (hopefully)
     }
   
   // reurn take
   return take;  
   }

void dispMenu( beverage inventory[SELECTION_SIZE] )
   {
   // vars
   int i = 0;
   beverage* currDrink;

   // clear screen 
   clrScr();

   // write drink menu
   cout << "            *** SODA MACHINE ***" << endl << endl;

   while(i < SELECTION_SIZE)
     {

     currDrink = &(inventory[i]);

     printf("%i", (i+1));
     printf(". ");
     printf("%-15s", currDrink->name);//<-- Find out how to get rid of hard code
     printf("%c%3.2f \n", '$', currDrink->price);

     i ++;
     }

     // load/save options
     cout << endl
          << "<L>oad Inventory Data" << endl
          << "<S>ave Inventory Data" << endl;

     // give option to leave
     cout << "<T>urn Machine Off / Leave" << endl << endl << endl << endl
          << endl << endl;

   // no return - void
   }

void loadMachine( beverage inventory[SELECTION_SIZE] )
   {
   // vars
   int i = 0;
   int inStock = ARR_SIZE(drink_list);
   beverage* currDrink;

   // load the data into the beverage struct array
   while(i < inStock)
     {
     currDrink = &(inventory[i]);
     strcpy(currDrink->name, drink_list[i]);
     currDrink->price = price_list[i];
     currDrink->onHand = CAPACITY;

     i ++; 
     } 

   while(i < SELECTION_SIZE)
     {
     currDrink = &(inventory[i]);
     strcpy(currDrink->name, "OUT OF STOCK");
     currDrink->price = 0;
     currDrink->onHand = 0.00;
 
     i ++;
     }

   // no return - void
   }

void loadData( char* inF, beverage inventory[SELECTION_SIZE] )
   {
   // vars
   ifstream fin;
   int i = 0;

   // check if file is there
   if( fileMissing(inF))
     {
     dispHelp(BAD_FILE);
     } 

   // load the machine with the data in the file
     // clear and open
     fin.clear();
     fin.open(inF, ios::in | ios::binary);

     // read the file contents
     if(fin.good())
       {
       fin.read(reinterpret_cast<char*>(inventory), sizeof(inventory));

       cout << endl << "Contents of " << inF << " have been loaded." << endl;
       holdProg();
       }

     // close
     fin.close();

   // no return - void
   }

void saveData( char* outF, beverage inventory[SELECTION_SIZE] )
   {
   // vars
   int i = 0;
   ofstream fout;
      
   // clear and open
   fout.clear();
   fout.open(outF, ios::out | ios::binary);

   // write contents to file
   fout.write( reinterpret_cast<char*>(inventory), sizeof(inventory));

   // notify that file has been created
   cout << endl << "Data has been saved as " << outF << "." << endl; 
   holdProg();

   // close
   fout.close();
   
   // no return - void
   }

bool fileMissing( char* fname )
   {
   // vars
   ifstream fTest;
   bool result = true;

   // clear and open
   fTest.clear();
   fTest.open(fname, ios::in | ios::binary);

   // check if file opened
   if( fTest.good() )
     {
     // set signal and close file
     result = false;
     fTest.close();
     }

   // return result
   return result;
   }

double dispenseDrink( beverage* choice )
   {
   // vars
   double payment = 0.00;
   double change = 0.00;
   double take = 0.00;

   // check drink availability, if out of stock, 
   // notify customer, return payment
   if(choice->onHand <= 0)
     {
     cout << endl << endl << "Sorry, that one is out of stock..." << endl;
     holdProg();
   
     return 0;
     }

   // collect money
   payment = getMoney(choice);
   change = (payment - choice->price);
   take = choice->price;

   if(payment != 0)
     {     
     // dispense drink and change
     printf("Here is your change: $%3.2f \n", change);
     cout << "Enjoy your " << choice->name << "!";

     choice->onHand --;
     holdProg(); 
     }

   // return take off this transaction
   return take; 
   }

double getMoney( beverage* choice )
   {
   // vars
   double payment = 0.00;  
   int dollar = 0;
   int dime = 0;
   int penny = 0;

   int numTries = 0;
   
   // request money
   while( (payment <= 0.00) || (payment > 1.00) || (payment < choice->price) )
     {
     if(numTries > 0)
       {
       cout << "Please enter an amount $1.00 or less, but that is" << endl
            << "still enough to pay for your beverage." << endl << endl;
       
       payment = 0.00;
       dollar = 0;
       dime = 0;
       penny = 0;

       // if they can't seem to get it right, go back to main menu
       if(numTries > 5)
         {
         // take no money
         cout << "NO SODA FOR YOU!";
         holdProg();
         return 0;
         }
       } 
     
     // prompt for money
     printf("Your drink costs $%3.2f \n", choice->price);
     cout << "Please insert payment: $";
     dollar = (getch() - '0');
     cout << dollar << ".";
     dime = (getch() - '0');
     cout << dime;
     penny = (getch() - '0');
     cout << penny;
     payment = double(dollar) + (double(dime) / 10) + (double(penny) / 100);
     cout << endl;

     numTries ++;
     }

   // return payment
   return payment;
   }

void dispHelp( int code )
   {
   // display title
   clrScr();
   cout << "     *** HELP MENU ***" << endl << endl;

   // display situation specific message
   switch(code)
     {
     case REQUEST:
       {
       cout << "Here is the help menu, as requested." << endl << endl;
       } break;

     case NUM_ARGS:
       {
       cout << "This program accepts 2 or 3 arguments (including" << endl
            << "the name of the program." << endl << endl;
       } break;

     case BAD_NAME:
       {
       cout << "File names must have the \".dat\" extension." << endl << endl; 
       } break;
     case BAD_FILE:
       {
       cout << "One or more of the files entered does not exist."
            << endl << endl;
       }break;
     }

   // display general message
   cout << "This is a vending machine simulation. The simulation" << endl
        << "must be booted from the command line." << endl;

   holdProg();

   // no return, but exit prog
   exit(code);
   }

