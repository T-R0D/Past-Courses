// Header files
#include <iostream>
#include <cmath>

using namespace std;

// Global constants

   // none

// Function prototypes

/*
Name: displayTitle
Process: display title with underline
Function Input/Parameters: none
Function Output/Parameters: none
Function Return: none
Device Input: none
Device Output/Monitor: title with underline
Dependencies: iostream I/O tools
*/
void displayTitle();

/*
Name: getCoefficent
Process: output prompt, input value
Function Input/Parameters: user prompt (string)
Function Output/Parameters: none
Function Return: requested coefficient (int)
Device Input/Keyboard: user input
Device Output/Monitor: user prompt
Dependencies: none
*/
int getCoefficient( const string &prompt );

/*
Name: calcDiscriminant
Process: b^2 - 4*a*c calculated
Function Input/Parameters: coefficients a, b, & c (int)
Function Output/Parameters: none
Function Return: discriminant (int)
Device Input: none
Device Output: none
Dependencies: none
*/
int calcDiscriminant( int c_a, int c_b, int c_c );

/*
Name: calcDenom
Process: 2 * a calculated
Function Input/Parameters: coefficient a (int)
Function Output/Parameters: none
Function Return: denominator (int)
Device Input: none
Device Output: none
Dependencies: none
*/
int calcDenom( int c_a );

/*
Name: calcRoot
Process: ( -b + discriminant ) / denominator calculated
Function Input/Parameters: coefficient b, denominator, discriminant (int)
Function Output/Parameters: none
Function Return: calculated root (int)
Device Input: none
Device Output: none
Dependencies: none
*/
int calcRoot( int c_b, int den, int dsc );
   
/*
Name: displayResult
Process: display description string and result to console
Function Input/Parameters: display description (string), root (int)
Function Output/Parameters: none
Function Return: none
Device Input: none
Device Output/Monitor: root description string with root value
Dependencies: none
*/
void displayResult( const string &disp, int result );

// Main program
int main()
   {
    // initialize the program/variables

       // initialize variables
       int coef_a, coef_b, coef_c;
       int disc, discRoot, denom;
       int result_one, result_two;

       // create and initialize number of roots to zero - default
       int numRoots = 0;

       // show title
          // function: displayTitle
       displayTitle();

    // get coefficients (3x, a, b, c )

       // get coefficient a
          // function: getCoefficient
       coef_a = getCoefficient( "Enter coefficient a: " );

       // get coefficient b
          // function: getCoefficient
       coef_b = getCoefficient( "Enter coefficient b: " );

       // get coefficient c
          // function: getCoefficient
       coef_c = getCoefficient( "Enter coefficient c: " );
    
    // process the roots

       // calculate the discriminant
          // function: calcDiscriminant
       disc = calcDiscriminant( coef_a, coef_b, coef_c );

       // test for discriminant greater than zero
       if( disc > 0 )
          {    
           // get the disc root
           discRoot = (int) sqrt( disc );
    
           // calculate the denominator
              // function: calcDenom
           denom = calcDenom( coef_a );
    
           // calculate the result (2x)

              // calc root one
                 // function: calcRoot
              result_one = calcRoot( coef_b, denom, discRoot );

              // calc root two
                 // function: calcRoot
              result_two = calcRoot( coef_b, denom, -discRoot );
    
           // identify number of roots (2)
           numRoots = 2;
          }

       // test for discriminant equal to zero
       else if( disc == 0 )
          {
           // get the disc root
              // function: sqrt
           discRoot = (int) sqrt( disc );
    
           // calculate the denominator
              // function: calcDenom
           denom = calcDenom( coef_a );
    
           // calculate the result (1x)      
              // function: calcRoot
           result_one = calcRoot( coef_b, denom, discRoot );
    
           // identify number of roots (1)
           numRoots = 1;
          }

    // display results

       // create vertical spaces
          // function: iostream <<
       cout << endl << endl;

       // check for two roots
       if( numRoots == 2 )
          {
           // display root 1
              // function: displayResult
           displayResult( "Root 1: ", result_one );

           // display root 2
              // function: displayResult
           displayResult( "Root 2: ", result_two );
          }

       // check for one root
       else if( numRoots == 1 )
          {
           // display root 1
              // function: displayResult
           displayResult( "Single Root: ", result_one );
          }

       // otherwise, if no roots
       else
          {
           // display error message
              // function: iostream <<
           cout << "No real roots exist for these coefficients"
                << endl;
          }

    // end program

       // shift cursor down
          // function: iostream <<
       cout << endl << endl;

       // hold program
          // function: system/pause
       system( "PAUSE" );
    
       // return success
       return 0;       
   }

// Supporting functions

void displayTitle()
   {
    // initialize function/variables
       // none

    // display title with underline and two vertical spaces
       // function: iostream <<
    cout << "Quadratic Equation Roots Calculator" << endl;
    cout << "===================================" 
         << endl << endl;
   }

int getCoefficient( const string &prompt )
   {
    // initialize function/variables
    int coef;

    // output prompt
       // function: iostream <<
    cout << prompt;

    // input user response
       // function: iostream >>
    cin >> coef;

    // return user response
    return coef;
   }

int calcDiscriminant( int c_a, int c_b, int c_c )
   {
    // initialize function/variables
    int result;

    // calculate result
    result = c_b * c_b - 4 * c_a * c_c;

    // return result
    return result;
   }                                             

int calcDenom( int c_a )
   {
    // initialize function/variables
    int den;

    // calc denominator
    den = 2 * c_a;

    // return denominator
    return den;
   }

int calcRoot( int c_b, int den, int dsc )
   {
    // initialize function/variables
    int result;

    // calculate result
    result = ( -1 * c_b + dsc ) / den;

    // return result
    return result;
   }                
   
void displayResult( const string &disp, int result )
   {
    // initialize function/variables
       // none

    // show result
       // function: iostream <<
    cout << disp << result << endl;
   }


