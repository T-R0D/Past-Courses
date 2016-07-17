// File:        driver.cpp

// Purpose:     Employee expense reporting, test program

// Programmer:  Anonymous

// Header Files
#include <iostream>
#include <fstream>

  // project-specific class headers
  #include "eer.hpp"

using namespace std;

int main()
   {
   // vars
   EmployeeExpenseReport employee1;
     employee1.set_destName( "Reno" );    
     employee1.set_expense_food( 68.92 );   // Total of all food bills
     employee1.set_expense_lodging( 768.42 );  // Total of all lodging costs
     employee1.set_expense_misc( 0.0 );     // Total of all miscellaneous costs
     employee1.set_expense_travel_in_dest( 34.56 );   // Taxi, rental car, bus, subway, etc.
     employee1.set_expense_travel_to_dest( 1000.04 ); // Air, bus, train, etc.
     employee1.set_name_first( "James" ); // First name of employee
     employee1.set_name_last ( "Kirk" );  // Last name of employee
     employee1.set_name_mi( 'T' );     // Middle initial of employee

   EmployeeExpenseReport employee2( 10.92, 768.42, 98.16, 1000.04, 14.78, "Xavier", "Spock", 'V', "Reno");
   EmployeeExpenseReport employee3;  
   EmployeeExpenseReport employee4;
     employee4 = employee1;
   
   ofstream fout;
  
   char* fname = "record.csv";

   // implement functions

     // clear and open
     fout.clear();
     fout.open( fname );

     // display and write each record
     employee1.write_report ();
     fout << employee1;
	  system("PAUSE");

     employee2.write_report ();
     fout << employee2;
	  system("PAUSE");

     employee3.write_report ();
     fout << employee3;
	  system("PAUSE");

     employee4.write_report ();
     fout << employee4;
	  system("PAUSE");

     // close file
     fout.close();

   // return 0
   return 0;
   }





    //employee2 = new EmployeeExpenseReport;

    //employee2->expense_food           = 98.16;    // Total of all food bills
    //employee2->expense_lodging        = 768.42;   // Total of all lodging costs
    //employee2->expense_misc           = 10.92;    // Total of all miscellaneous costs
    //employee2->expense_travel_in_dest = 14.78;    // Taxi, rental car, bus, subway, etc.
    //employee2->expense_travel_to_dest = 1000.04;  // Air, bus, train, etc.
    //employee2->name_first             = "Xavier"; // First name of employee
    //employee2->name_last              = "Spock";  // Last name of employee
    //employee2->name_mi                = 'V';      // Middle initial of employee