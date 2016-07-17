////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      asmt07_driver.cpp  
//  Created By: Terence Henriod
//  Course:     CS202
//
//  Summary:    The driver program to display the capabilities of the myIntList
//              doubly linked list class.
// 
//  Last Modified: 4/16/2013 19:30
//
////////////////////////////////////////////////////////////////////////////////



//============================================================================//
//= Header Files =============================================================//
//============================================================================//

// header files / namespace
#include <iostream>
#include <fstream>
#include <conio.h>
#include <stdlib.h>
#include <assert.h>
#include "Ttools.h"
using namespace std;

// supplementary headers

// class headers
#include "myIntListClass.h"


//============================================================================//
//= Macros ===================================================================//
//============================================================================//

#pragma warning (disable : 4996)   // disables strcpy, strcat warnings in Visual
#pragma warning (disable : 4244)   // disables time(NULL) warnings in Visual
#pragma warning (disable : C4996)   // disables getch warning in Visual



//============================================================================//
//= Global Constant Definitions ==============================================//
//============================================================================//



//============================================================================//
//= Global Function Prototypes ===============================================//
//============================================================================//



//============================================================================//
//= Main Program Definition ==================================================//
//============================================================================//

int main()
   {
   // vars
   myIntListClass list1;    // uses default constructor
     list1.add(1);
   myIntListClass list2(2);  // uses 1 of the overloaded constructors
     int set[] = {4, 8, 6, 8, 6, 4, 2};  
     int setSize = ARR_SIZE( set );  // add a set to list2
     list2.addSet(set, setSize);

     int primeset[] = { 1, 2, 3, 5, 7, 11, 13, 17, 23, 29 };
     int primesetSize = ARR_SIZE( primeset );
   myIntListClass list3(primeset, primesetSize);  // uses the other overloaded constructor

   list1.add(9);    // exhibit use of add() member function
   list1.add(5);
   list1.add(7); 
   list1.add(9);
   list1.add(5);
   list1.add(7);

   // output traversed values

   cout << "list2 values: " << endl;
   list2.traverse();
   cout << "list1 values: " << endl;
   list1.traverse();
   cout << endl;
   holdProg();

   // sort the lists, traverse values to display
   list1.sortS2L();
   list2.sortL2S();
   cout << "list2 values, sorted: " << endl;
   list2.traverse();
   cout << endl;
   cout << "list1 values, sorted: " << endl;
   list1.traverse();
   cout << endl;
   holdProg();

   list1.deleteVal(7);
   list2.deleteVal(4);


   cout << "list2 values with a 4 deleted: " << endl;
   list2.traverse();
   cout << "list1 values with a 7 deleted: " << endl;
   list1.traverse();
   cout << endl;
   holdProg();

   cout << "list3, traversed:" << endl;
   list3.traverse();
   cout << endl;
   holdProg();

   cout << "list3 traverse_reversed: " << endl;
   list3.traverse_reverse();
   cout << endl;
   holdProg();

   // return 0 upon successful completion
   return 0;
   }



//============================================================================//
//= Supporting function implementations ======================================//
//============================================================================//

