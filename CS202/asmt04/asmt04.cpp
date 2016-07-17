#include "asmt04.hpp"

///////////////////////////////////////////////////////////////////////////////
//
//  This file defines the function for asmt04: a function that will find the 
//  factors of a given number using recursion.
//
///////////////////////////////////////////////////////////////////////////////


// function definition
unsigned int * find_factors_using_recursion ( unsigned int x )
   {
   // vars
   static unsigned int i = 1;      // factor candidate variable
   static int ndx = 0;             // factor list index
   extern unsigned int * factors;  

   // algorithm //
   
   // base case
   if(i == x)   // last factor has been found
     {
     *(factors + ndx) = i;

     i = 0;    // reset for next time function is used
	 ndx = 0;

     return factors;
     }

   // otherwise, 
     // if a factor has been found, store it
     if((x % i) == 0)        
       {
       *(factors + ndx) = i;
       ndx ++;
       }

     // regardless, search for next factor in sequence
     i ++;               
     factors = find_factors_using_recursion(x);

   // return pointer
   return factors;
   }

