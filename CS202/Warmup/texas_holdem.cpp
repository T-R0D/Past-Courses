
// Header Files/////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
using namespace std;

// Global Constant Definitions//////////////////////////////////////////////////

const char NULLCH = '\0';
const char SPACE = ' ';
const char HEART = char(3);
const char DIAMOND = char(4);
const char CLUB = char(5);
const char SPADE = char(6);


const int D_SIZE = 52;
const int C_NAME = 6;
const int FNAME_LEN = 20;

const int NUMSHUFF = 5;

const char DECK_F[FNAME_LEN] = "hold_em_deck.txt";
const char SHUFFLED_F[FNAME_LEN] = "shuffled.txt";
const char HARD_F[FNAME_LEN] = "unshuffled_cards.txt";


// Global Function Prototypes///////////////////////////////////////////////////

    // none

/* 
Name: 
Process: 
Function Input/Parameters: 
Function Output/Parameters: 
Function Output/Returned: 
Device Input: 
Device Output: 
Dependencies: 
*/
//void function() ;

void generateDeck( char deck[D_SIZE][C_NAME] );
void readDeck( char deck[D_SIZE][C_NAME] );
void bubbleShuffle( char deck[D_SIZE][C_NAME] );
void clrCard( char card[C_NAME] );
void strCop( char orig[], char copy[] );
int getRandBetw( int loVal, int hiVal );
void writeDeck( char deck[D_SIZE][C_NAME], const char fname[] );
void playHoldem( char deck[D_SIZE][C_NAME] );


// Main Program Definition//////////////////////////////////////////////////////
int main()
   { 
// vars
char deck[D_SIZE][C_NAME];

   // random seed
   srand( time( NULL ) );  

// gen deck
  generateDeck( deck );

// read deck
  readDeck( deck );
  //writeDeck( deck, SHUFFLED_F ); 

// execute play
  playHoldem( deck );

// end program

  system( "PAUSE" ); 
  return 0;
   }

////////////////////////////////////////////////////////////////////////////////

void generateDeck( char deck[D_SIZE][C_NAME] )
   {
   // vars
   int rndx = 0; //cndx = 0;
   int suitnum = 0;
   char suit;
   int cardnum = 2;
   
   // loop through the deck to create the cards
   while( rndx < D_SIZE )
     {
     switch( suitnum )
       {
       case 0:
         suit = HEART;
         break;
       case 1:
         suit = DIAMOND;
         break;
       case 2: 
         suit = CLUB;
         break;
       case 3:
         suit = SPADE;
         break;
       }

     deck[rndx][0] = 'A';
     deck[rndx][1] = ' ';
     deck[rndx][2] = suit;
     deck[rndx][3] = NULLCH;
     rndx ++;

     for( cardnum = 2; cardnum < 10; cardnum ++, rndx ++)
       {
       deck[rndx][0] = ( char( cardnum ) + '0');
       deck[rndx][1] = ' ';
       deck[rndx][2] = suit;
       deck[rndx][3] = NULLCH;
       }

     deck[rndx][0] = '1';
     deck[rndx][1] = '0';
     deck[rndx][2] = ' ';
     deck[rndx][3] = suit;
     deck[rndx][4] = NULLCH;
     rndx ++;

     deck[rndx][0] = 'J';
     deck[rndx][1] = ' ';
     deck[rndx][2] = suit;
     deck[rndx][3] = NULLCH;
     rndx ++;

     deck[rndx][0] = 'Q';
     deck[rndx][1] = ' ';
     deck[rndx][2] = suit;
     deck[rndx][3] = NULLCH;
     rndx++;

     deck[rndx][0] = 'K';
     deck[rndx][1] = ' ';
     deck[rndx][2] = suit;
     deck[rndx][3] = NULLCH;
     rndx ++;

     suitnum ++;
     }

   writeDeck( deck, DECK_F );
 
   // no return - void
   }


void readDeck( char deck[D_SIZE][C_NAME] )
   {
   // vars
   ifstream fin;
   int rndx = 0, cndx = 0;
   char rank = 'a';
   char suit = HEART;
   char dummy = 'q';

   // clear/open fstream object
   fin.clear();
   fin.open( DECK_F );

   // read in the deck
   fin >> rank;   // prime

   while( fin.good() && (rndx < D_SIZE) )
     {
     deck[rndx][cndx] = rank;

     fin >> dummy;
     deck[rndx][cndx] = dummy;

     fin >> suit;
     deck[rndx][cndx] = suit;

     fin >> dummy;
     if( dummy == '\n' )
       {
       rndx ++;
       }

     fin >> rank;
     }

   // close the file
   fin.close();

   // void - no return
   }


void bubbleShuffle( char deck[D_SIZE][C_NAME] )
   {
   // vars
   int rndx = 0;
   int swapPos = 0;
   int counter = 0;
   char buffer[C_NAME];

   // swap card at current position with one in a random position, 
   // iterating through the deck one at a time, NUMSHUFF times to ensure a good shuffle
   while( counter < NUMSHUFF )
     {
     for( rndx = 0; rndx < D_SIZE; rndx ++ )  
       {
       clrCard( buffer );              // save card at current position
       strCop(deck[rndx], buffer);
       clrCard( deck[rndx] );
 
       swapPos = getRandBetw( 0, (D_SIZE - 1));     // make the swap
       strCop( deck[swapPos], deck[rndx] );
       clrCard( deck[swapPos] );
       strCop( buffer, deck[swapPos] );
       }

     counter ++;
     }

   // no return - void
   }


void clrCard( char card[C_NAME] )
   {
   // vars
   int indx = 0;

   // store spaces to all valid char slots
   while( indx < (C_NAME - 2))
     {
     card[indx] = SPACE;
     indx ++;
     } 
   card[C_NAME - 1] = NULLCH;
   
   // no return - void
   }


void strCop( char orig[], char copy[] )
   {
   // vars
   int ndx = 0;

   // replace the contents of copy with those of orig, stop at \0
   while( orig[ndx] != NULLCH )
     {
     copy[ndx] = orig[ndx];
     ndx ++;
     }
   
   copy[ndx] = NULLCH;
   }


int getRandBetw( int loVal, int hiVal )
   {
   // vars
   int range = (hiVal - loVal);
   int randVal;

   // generate appropriate number
   randVal = ((rand() % range) + loVal);

   // return
   return randVal;   
   }

void writeDeck( char deck[D_SIZE][C_NAME], const char fname[] )
   {
   // vars
   ofstream fout;
   int rndx = 0; //cndx = 0;
   int cardcount = 0;

   // clear and open
   fout.clear();
   fout.open( fname );

   // output deck
   for( rndx = 0; rndx < D_SIZE; )
     {
     for( cardcount = 0; cardcount < 13; cardcount ++)
       {
       fout << deck[rndx] << ' ';
       rndx ++;
       }
     
     fout << '\n';
     }

   // close
   fout.close();

   // no return - void
   }

void playHoldem( char deck[D_SIZE][C_NAME] )
   {
   // vars
   int selection = 9;

   // initialize prog
   cout << "Welcome to Texas Hold 'em!" << endl << endl;













   // shuffle (and write shuffled)
   bubbleShuffle( deck );
   writeDeck( deck, SHUFFLED_F);


   }
