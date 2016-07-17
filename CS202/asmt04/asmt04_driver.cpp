#include <iostream>
#include "asmt04.hpp"
using namespace std;


unsigned int* factors;

int main()
   {
   int response = 1;
   int ndx = 0;

   cout << "What number would you like to factor? ";
   cin  >> response;
   cout << endl << endl;

   factors = new unsigned int [response];
     memset(factors, 0, response * (sizeof(unsigned int)));

   factors = find_factors_using_recursion(response);

   cout << "The factors of " << response << " are: ";

   while(*(factors + ndx) != 0)
   {
   cout << *(factors + ndx) << " ";
   ndx ++;
   }

   delete [] factors;

   system ("PAUSE");

   return 0;
   }
