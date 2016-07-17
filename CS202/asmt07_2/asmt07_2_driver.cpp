////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      asmt07_driver.cpp  
//  Created By: Terence Henriod
//  Course:     CS202
//
//  Summary:    The driver program to display the capabilities of the myStringClass
//              string class.
// 
//  Last Modified: 4/18/2013 20:30
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
#include "myStringClass.h"


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
	//Define a string object using default constructor
	myStringClass list;
	int i;

	//Display length of string
	i = list.length();
	cout << i << endl;

	//Add suffix to existing string
	char* suffix = "list";
	list.myStringCat( suffix );
	cout << list << endl;

	//Display length of string
	i = list.length();
	cout << i << endl;

	//Add suffix to existing string using operator 
	list + "object";
	cout << list << endl;

	//Display new length of string
	i = list.length();
	cout << i << endl;

   // hold for user
   holdProg();

   // return 0 upon successful completion
   return 0;
   }



//============================================================================//
//= Supporting function implementations ======================================//
//============================================================================//

