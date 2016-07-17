////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      gameClass.cpp
//  Created By: Terence Henriod
//  Course:     CS202
//
//  Summary:    The implementation of the class defined in gameClass.h
// 
//  Last Modified: 
//
////////////////////////////////////////////////////////////////////////////////


//============================================================================//
//= Header Files =============================================================//
//============================================================================//

// class definition header
#include "gameClass.h"

// other headers
using namespace std;


//============================================================================//
//= Global (Static) Variables ================================================//
//============================================================================//
int gameClass::gameNumber = 0;

//============================================================================//
//= Function Implementation ==================================================//
//============================================================================//


// Constructor(s) //////////////////////////////////////////////////////////////
gameClass::gameClass()
   {
   // vars 
   int i = 0;

   // increment the game number
   gameNumber ++;
 
   // initialize/shuffle the deck
      // done during class data member initialization

   // get number of players ( numP = P1 + optionals)
   numP = randBetw(1, MAX_P);   

   // initialize players
   currPlayer = new player[MAX_P];
     // initialize their names properly
     // i = 1 because player 1's name was already set to player 1
     for(i = 1; i < MAX_P; i++){  
       currPlayer[i].setName(i);
     }
   theHouse = new dealer;

   // get game time of start
   gameStartTime();

   // no return - constructor
   }


// Destructor //////////////////////////////////////////////////////////////////
gameClass::~gameClass()
   {
   // return dynamic memory
   delete [] currPlayer;
   delete theHouse;

   // no return - destructor
   }

// Internal/Maintenance ////////////////////////////////////////////////////////
void gameClass::gameStartTime()
   {
   // vars
   time_t rawtime;

   // get the time, convert it to something useable
   time( &rawtime );
   startTime = localtime ( &rawtime );
   }

void gameClass::startDeal()
   {
   // vars
   int pndx = 0;
   displayHandClass* display;
   bool hideFirst = true;

   // deal to the house 
   theHouse->hit(theDeck);
   theHouse->hit(theDeck);

   // deal to each player
   for(pndx = 0; pndx < numP; pndx ++){
   currPlayer[pndx].hit(theDeck);
   currPlayer[pndx].hit(theDeck);   
   }

   // display the initial hands
   display = new displayHandClass;

     // first the dealer
     display->simpleDisp(theHouse, hideFirst);
     holdProg();

     // then the players
     hideFirst = false;
     for(pndx = 0; pndx < numP; pndx++){
       display->simpleDisp(&currPlayer[pndx], hideFirst);
       holdProg();
     }

   // return the dynamic memory  
   delete display;

   // no return - void
   }

void gameClass::playerTurn( int playerNum )
   {
   // vars
   bool turnDone = false;
   char response = '\0';
   displayHandClass* displayer;

   // notify player of their turn
   clrScr();
   cout << currPlayer[playerNum].name() << " it's your turn." 
        << endl <<    "=========================" << endl;

     // display their hand
     displayer = new displayHandClass;
     displayer->disp(currPlayer[playerNum].vHand());
     delete displayer;
     cout << endl << endl;

   // if the hand is blackjack, end the turn before it starts
   if(currPlayer[playerNum].vHand().score() == 21){
     cout << "     BLACKJACK!!! " << endl 
          << "  You're turn is done." << endl << endl;
     turnDone = true;
     holdProg();
   }

   // implement turn menu
   while(currPlayer[playerNum].score() < 21 &&
         !turnDone){      
     // display
     response = hitMenu();

     // execute decision
     switch(response)
       {
       case '1':
         // give the player a card
         currPlayer[playerNum].hit(theDeck);
         cout << "   YOU HIT!" << endl << endl;

         // if the player busts end their turn
         if(currPlayer[playerNum].score() > 21){
           cout << "   BUST!!!" << endl << endl;
           turnDone = true;
         }
         if(currPlayer[playerNum].score() == 21){
           cout << "   BLACKJACK!!!" << endl << endl;
           turnDone = true;     
         }
         break;

       case '2':
         turnDone = true;
         cout << "   YOU STAND!" << endl << endl;
         break;

       case 'p':
         theDeck.peekAtTop();
         break;
     }

     // display hand
     displayer = new displayHandClass;
     displayer->disp(currPlayer[playerNum].vHand());
     holdProg();
     delete displayer;
   }

   // no return - void
   }

void gameClass::dealerTurn()
   {
   // vars
   displayHandClass* displayer;

   // make dealer hit until score is >= 17
   while(theHouse->score() < 17){
     theHouse->hit(theDeck);
   }

   // display dealer hand
   cout << "   The " << theHouse->name() << "'s final hand:" << endl
        << "=========================" << endl;
   displayer = new displayHandClass;
   displayer->disp(theHouse->vHand());
   delete displayer;
  
   // hold the program so the hand can be used
   holdProg();

   // no return - void
   }

char gameClass::hitMenu()
   {
   // vars
   char response = 'a';

   // display options
   cout << "   What are you going to do?" << endl << endl
        << "     <1> Hit me!" << endl
        << "     <2> Stand." << endl;
   cout << endl << endl;

   // get response
   response = getch();
   cout << endl << endl;

   // return response
   return response;   
   }   

void gameClass::scoreGame()
   {
   // vars
   int i = 0;
   result tempR;

   // load the data for non-playing players
   for(i = 2; i >= numP; i--){ 
     // set the played flag to false, and copy the name over
     tempR.played = false;
     strcpy(tempR.pName, currPlayer[i].name());
     // other data is irrelevant and does not need initialization

     // push the result onto the summary stack
     summary.push(tempR);     
   }

   // load summary data for the players who played
   for( /**/ ; i >= 0; i--) {
     // copy over nonconditional data
     tempR.played = true;
     strcpy(tempR.pName, currPlayer[i].name()); 
     tempR.pHand = currPlayer[i].vHand();
     tempR.score = currPlayer[i].score(); 
     
     // determine the outcome
     // the player busts
     if(currPlayer[i].score() > 21){
       tempR.outcome = 'L';
     }
     // the player has the low score and dealer doesn't bust
     else if(currPlayer[i].score() < theHouse->score() &&
             theHouse->score() < 21){
       tempR.outcome = 'L';
     }
     // "push"
     else if(currPlayer[i].score() == theHouse->score() &&
             theHouse->score() < 21){
       tempR.outcome = 'P';
     }
     // dealer busts, player does not
     else if(currPlayer[i].score() <= 21 &&
             theHouse->score() > 21){
       tempR.outcome = 'W';
     }
     // player wins, straight up
     // by now we have already checked for a bust in either case
     else if(currPlayer[i].score() > theHouse->score()){
       tempR.outcome = 'W';
     }

     // push the data onto the stack
     summary.push(tempR);
   }
     
   // store the dealer data
   tempR.played = true;
   strcpy(tempR.pName, theHouse->name());
   tempR.pHand = theHouse->vHand();
   tempR.score = theHouse->score();
   tempR.outcome = 'D';
   summary.push(tempR);

   // no return - void 
   }

void gameClass::showOutcomes()
   {
   // vars
   result temp;
   displayHandClass* display = new displayHandClass;
   stack<result> holder;

   // indicate that these are the results
   cout << "     GAME RESULTS" << endl
        << "     ~~~~~~~~~~~~" << endl << endl;

   // output data from the stack to screen
   // hold the data for reloading
   while(!summary.empty()){
     // copy data, display it, hold it in the queue
     temp = summary.top();
     summary.pop();

     // if the player played, display the result
     if(temp.played == true){
       // the player's name
       cout << "  " << temp.pName << endl
       // the hand
            << "     Hand: " << temp.pHand << endl
       // the score
            << "     Score: " << temp.score << endl
       // and the outcome
            << "     Result: ";
       switch(temp.outcome){
         case 'D':
           cout << "DEALER";
           break;

         case 'L':
           cout << "LOSS"; 
           break;
 
         case 'P':
           cout << "PUSH";
           break;

         case 'W':
           cout << "WIN";
           break;
       }
       cout << endl << endl;
     }
  
     // hold the result
     holder.push(temp);
   }

   // hold the program for the user to view the result
   holdProg();
  
   // put the data back in the original stack
   while(!holder.empty()){
     summary.push(holder.top());
     holder.pop();
   }

   // return dynamic memory
   delete display;

   // no return - void
   }

// Accessors ///////////////////////////////////////////////////////////////////
void gameClass::play()
   {
   // vars
   int i = 0;

   // perform initial deal
   startDeal();
   
   // give player's their turns
   for(i = 0; i < numP; i ++){
     playerTurn(i);
   }

   // dealer turn 
   dealerTurn();

   // score the game for each player
   scoreGame();

   // put the outcomes on screen
   showOutcomes();

   // no return - void
   }
  
void gameClass::recordGameData(const char* outFile)
   {
   // vars 
   ofstream fout;
   result tempR;
   char* fileFailEx = "A file operation failed unexpectedly...";


   // clear/open
   fout.clear();      
   fout.open(outFile, ios::app);  // open in append mode
     
   // write data by popping it off the stack one at a time
     // write the game number and timestamp (requires asctime(struct tm* obj))
     fout << "Game " << gameNumber << ": " << endl
          << "    " << asctime(startTime) << endl;

     // write the player data
     while(!summary.empty()){
      // get the result
      tempR = summary.top();
        summary.pop();

      // make space and write the player's name
      fout << endl << tempR.pName << endl;

      // check to see if the player played
      if(!tempR.played){
      fout << " DID NOT PARTICIPATE IN THIS GAME" << endl << endl;
      }
      else{
        // write the number of cards
        fout << "Number of cards: " << tempR.pHand.numC() << endl;

        // the cards in the hand
        fout << "The hand: " << tempR.pHand << endl;

        // the score
        fout << "The score: " << tempR.score << endl;

        // the outcome, if necessary
        switch(tempR.outcome){
          case 'D':
            // no outcome necessary
            break;

          case 'L':
            fout << "Outcome: LOSS" << endl;   
            break;

          case 'P':
            fout << "Outcome: PUSH" << endl;
            break;

          case 'W':
            fout << "Outcome: WIN" << endl;
            break;
        }
     }  
   } 

     // if any file operation failed, throw exception
     // this will be caught in main to demonstrate
     // "unwinding the stack" (play21 function)
     if(!fout.good()){
       throw fileFailEx;
     }
   

   // make some space between records
   fout << endl << endl;

   // close
   fout.close();

   // no return - void
   }

// Mutators ////////////////////////////////////////////////////////////////////

// Overloaded Operators ////////////////////////////////////////////////////////

