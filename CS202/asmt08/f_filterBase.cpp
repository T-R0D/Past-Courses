////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      f_filterBase.cpp
//  Created By: Terence Henriod
//  Course:     CS202
//
//  Summary:    The implementation of the class defined in abstractBase.h
// 
//  Last Modified: 4/25/2013 20:40
//
////////////////////////////////////////////////////////////////////////////////


//============================================================================//
//= Header Files =============================================================//
//============================================================================//


// class definition header
#include "f_filterBase.h"


// other headers
using namespace std;


//============================================================================//
//= Global (Static) Variables ================================================//
//============================================================================//


//============================================================================//
//= Function Implementation ==================================================//
//============================================================================//


// Constructor(s) //////////////////////////////////////////////////////////////
f_filterBase::f_filterBase()   // default
   {
   // vars
   inFName = "";
   outFName = "";

   // no return - constructor
   }

f_filterBase::f_filterBase( const string &inF, const string &outF )
   {
   // initialize file names with given parameters
   inFName = inF;
   outFName = outF;

   // constructor - no return
   }

// Destructor //////////////////////////////////////////////////////////////////
f_filterBase::~f_filterBase()
   {
   // currently no tasks

   // no return - destructor
   }

// Internal/Maintenance ////////////////////////////////////////////////////////
bool f_filterBase::readWrite(ifstream &in, ofstream &out)
   {
   // vars
   bool success = false;

   // process each char
   currChar = in.get();
     if( in.good() )
       {
       success = true;
       }

   while(in.good())
     {
     out << transform( currChar );
     currChar = in.get();
     }
 
   // return success state
   return true;
   }

// Accessors ///////////////////////////////////////////////////////////////////
string f_filterBase::view_inF() const
   {
   return inFName;
   }

string f_filterBase::view_outF() const
   {
   return outFName;
   }

// Mutators ////////////////////////////////////////////////////////////////////
void f_filterBase::set_inF( const string &inF )
   {
   // internalize the given parameter
   inFName = inF;

   // no return - void
   }

void f_filterBase::set_outF( const string &outF )
   {
   // internalize the given parameter
   outFName = outF;

   // no return - void
   }

bool f_filterBase::doFilter(ifstream &in, ofstream &out)
   {
   // vars
   bool success = false;   // indicates at least one char processed.

   // open the files, if they open, proceed
   openFile(inFName, in);

   if(in.good())
     {
     // open the output file
     openFile(outFName, out);  
  
     // perform filtering
     success = readWrite(in, out);   
     }
   else   // indicate that file filtering failed
     {
     cout << "File filtering failed" << endl;
     holdProg();
     return success;
     }
   
   // close the files
   in.close();
   out.close();

   // return success
   return success;
   }

// Overloaded Operators ////////////////////////////////////////////////////////





