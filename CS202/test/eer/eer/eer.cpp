// File:        eer.cpp

// Purpose:     Employee expense reporting class

// Programmer:  Anonymous

#include <iostream>
#include "eer.hpp"

using namespace std;

// class specific global variables
int EmployeeExpenseReport::counter = 0; 



// These methods are accessible anywhere within the scope of
// the EmployeeExpenseReport class

// Member functions

// NOTE:    Constructors and destructors are called automatically.
//          You're not permitted to call them yourself.
//          Also, note that they have no return type.

// Constructors
EmployeeExpenseReport::EmployeeExpenseReport ( ) // default
   {
   counter ++;
   recNum                 = counter;
   expense_misc           = 0;          
   expense_lodging        = 0;       
   expense_food           = 0;    
   expense_travel_to_dest = 0; 
   expense_travel_in_dest = 0; 
   totalExpense           = 0;  
   name_first             = "";        
   name_last              = "";         
   name_mi                = ' ';
   destName               = "";   
            
   }

EmployeeExpenseReport::EmployeeExpenseReport ( double m, double l, double f, double t, double i,
                                               string first, string last, char mi, string dest ) 
   {
   counter ++;
   recNum  = counter;

   expense_misc           = m;          
   expense_lodging        = l;       
   expense_food           = f;    
   expense_travel_to_dest = t; 
   expense_travel_in_dest = i; 
   sumExpenses();
   name_first             = first;        
   name_last              = last;         
   name_mi                = mi; 
   destName               = dest;                         
   }

EmployeeExpenseReport::~EmployeeExpenseReport ( ) //Destructor
   {

   // currently nothing to destruct

   }

// private functions
void EmployeeExpenseReport::sumExpenses()
   {
   // vars
   double sum = 0;

   // re-sum all expenses
   sum += expense_misc;          
   sum += expense_lodging;       
   sum += expense_food;    
   sum += expense_travel_to_dest; 
   sum += expense_travel_in_dest; 
 
   totalExpense = sum;

   // no return - void
   }

// Mutator methods
void EmployeeExpenseReport::set_expense_travel_to_dest ( const double t )
{ 
    expense_travel_to_dest=t;
    sumExpenses();
}

void EmployeeExpenseReport::set_expense_travel_in_dest ( const double i )
{ 
    expense_travel_in_dest=i;
    sumExpenses();
}

void EmployeeExpenseReport::set_expense_lodging ( const double l)
{ 
    expense_lodging=l;
    sumExpenses();
}

void EmployeeExpenseReport::set_expense_food ( const double f )
{ 
    expense_food=f; 
    sumExpenses();    
}

void EmployeeExpenseReport::set_expense_misc  ( const double m )
{ 
    expense_misc=m;
    sumExpenses();

}

void EmployeeExpenseReport::set_name_last  ( const string last)
{
    name_last=last;
    
}

void EmployeeExpenseReport::set_name_first  ( const string first)
{
    name_first=first;

}
void EmployeeExpenseReport::set_name_mi   ( const char mi)
{
    name_mi=mi;

}
void EmployeeExpenseReport::set_destName (const string dest) 
{
    destName=dest;

}

// Accessor Methods
void EmployeeExpenseReport::write_report ( )
   {
    cout << "Employee Expense Report"
         << endl << endl

         << "    " << recNum                                   << endl
         << "    Destination:      " << destName               << endl << endl

         << "    Last  Name:       " << name_last              << endl
         << "    First Name:       " << name_first             << endl
         << "    Middle Initial:   " << name_mi                << endl
         << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"                   << endl
         << "    Travel to Event:  " << expense_travel_to_dest << endl
         << "    Travel at Event:  " << expense_travel_in_dest << endl << endl

         << "    Lodging:          " << expense_lodging        << endl
         << "    Food:             " << expense_food           << endl
         << "    Miscellaneous:    " << expense_misc           << endl
         << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"                   << endl
         << "    Total Expense:    " << totalExpense           <<endl

         << endl << endl;
    }

// overloaded operators
   void EmployeeExpenseReport::operator = ( const EmployeeExpenseReport &right )
      {
      // copy appropriate values
      recNum = right.recNum;

      destName = right.destName;

      name_last              = right.name_last;
      name_first             = right.name_first;         
      name_mi                = right.name_mi;

      expense_travel_to_dest = right.expense_travel_to_dest;
      expense_travel_in_dest = right.expense_travel_in_dest;

      expense_lodging        = right.expense_lodging;
      expense_food           = right.expense_food;
      expense_misc           = right.expense_misc;

      // re-sum the expenses
      sumExpenses();

      }


    ostream& operator<< (ostream &out, const EmployeeExpenseReport &object )
      {
      // vars
      
      // write each member, followed by the delimiter
      out << object.recNum                 << object.DELIM
          << object.destName               << object.DELIM
          << object.name_last              << object.DELIM
          << object.name_first             << object.DELIM
          << object.name_mi                << object.DELIM
          << object.totalExpense           << object.DELIM
          << object.expense_travel_to_dest << object.DELIM
          << object.expense_travel_in_dest << object.DELIM
          << object.expense_lodging        << object.DELIM
          << object.expense_food           << object.DELIM
          << object.expense_misc           << endl;
 
      // return ofstream object
      return out;
      }


   istream& operator>> (istream &in, const EmployeeExpenseReport &object )
      {

      // return the stream object
      return in;
      }

 

