#ifndef EMP_EXP_RPT_HPP
#define EMP_EXP_RPT_HPP

// File:        eer.hpp

// Purpose:     Employee expense reporting class

// Programmer:  Anonymous

#include <string>

using namespace std;

class EmployeeExpenseReport
{
   private:
     // constants
     static const char DELIM = ',';

     static int counter;

     // Data
     int recNum;
     double expense_misc;           // Miscellaneous costs
     double expense_lodging;        // Lodging costs
     double expense_food;           // Total of all food bills
     double expense_travel_to_dest; // Air, boat, train, etc.
     double expense_travel_in_dest; // Taxi, rental car, bus, subway, etc.
     double totalExpense;

     string name_first;             // First name of employee
     string name_last;              // Last name of employee
     char   name_mi;                // Middle initial of employee

     string destName;            

     // Functions
     void EmployeeExpenseReport::sumExpenses();       

   public:
     // Functions

     // Constructors/Destructor
      EmployeeExpenseReport ( );

      // Overloaded Constructor
      EmployeeExpenseReport::EmployeeExpenseReport ( double m, double l, double f, double t, double i,   
                                                     string first, string last, char mi, 
                                                     string dest ); 

      ~EmployeeExpenseReport ( );

      // mutators
      void EmployeeExpenseReport::set_expense_travel_to_dest ( const double t );
      void EmployeeExpenseReport::set_expense_travel_in_dest ( const double i );
      void EmployeeExpenseReport::set_expense_lodging ( const double l);
      void EmployeeExpenseReport::set_expense_food ( const double f );
      void EmployeeExpenseReport::set_expense_misc ( const double m );
      void EmployeeExpenseReport::set_name_last ( const string last);
      void EmployeeExpenseReport::set_name_first ( const string first);
      void EmployeeExpenseReport::set_name_mi ( const char mi);
      void EmployeeExpenseReport::set_destName (const string dest);

      // accessors
        // getters
        inline double EmployeeExpenseReport::getMiscExp() const
           {
           return expense_misc;          
           }

        inline double EmployeeExpenseReport::getLodge() const
           {
           return expense_lodging;          
           }

        inline double EmployeeExpenseReport::getfood() const
           {
           return expense_food;          
           }

        inline double EmployeeExpenseReport::getTravToDest() const
           {
           return expense_travel_to_dest;          
           }

        inline double EmployeeExpenseReport::getTravAtDest() const
           {
           return expense_travel_in_dest;          
           }

        inline double EmployeeExpenseReport::getTotal() const
           {
           return totalExpense;          
           }

        inline string EmployeeExpenseReport::getFirstName() const
           {
           return name_first;          
           }

        inline string EmployeeExpenseReport::getLastName() const
           {
           return name_last;          
           }

        inline char EmployeeExpenseReport::getMI() const
           {
           return name_mi;          
           }

        inline string EmployeeExpenseReport::getDest() const
           {
           return destName;          
           }


        // displayers
        void EmployeeExpenseReport::write_report ( );

        inline void EmployeeExpenseReport::dispMiscExp()
           {
           cout << expense_misc << endl;          
           }

        inline void EmployeeExpenseReport::dispLodge()
           {
           cout << expense_lodging << endl;          
           } 

        inline void EmployeeExpenseReport::dispFood()
           {
           cout << expense_food << endl;          
           } 

        inline void EmployeeExpenseReport::dispTravToDest()
           {
           cout << expense_travel_to_dest << endl;          
           } 

        inline void EmployeeExpenseReport::dispTravAtDest()
           {
           cout << expense_travel_in_dest << endl;          
           } 

        inline void EmployeeExpenseReport::dispTotal()
           {
           cout << totalExpense << endl;          
           } 

        inline void EmployeeExpenseReport::dispFirstName()
           {
           cout << name_first << endl;          
           } 

        inline void EmployeeExpenseReport::dispLastName()
           {
           cout << name_last << endl;          
           } 

        inline void EmployeeExpenseReport::dispMI()
           {
           cout << name_mi << endl;          
           } 

        inline void EmployeeExpenseReport::dispDest()
           {
           cout << destName << endl;          
           } 

   // overloaded operators
   void operator = ( const EmployeeExpenseReport &right );
 
   friend ostream& operator<< (ostream &out, const EmployeeExpenseReport &object );
   friend istream& operator>> (istream &in, const EmployeeExpenseReport &object );   
   };

#endif
