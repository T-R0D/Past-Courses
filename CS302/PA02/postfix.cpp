////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      
//  Created By:
//  Reviewed By:
//  Course:     
//
//  Summary:    
// 
//  Last Modified: 
//
////////////////////////////////////////////////////////////////////////////////

//============================================================================//
//= Header Files =============================================================//
//============================================================================//

// common headers/namespaces
#include <iostream>
#include <cstdlib>
#include <iomanip>
using namespace std;

#include "StackLinked.cpp"


//============================================================================//
//= Global Constants =========================================================//
//============================================================================//
 const int MAX_EXPRESSION = 61;

//============================================================================//
//= Function Prototypes ======================================================//
//============================================================================//

void runPostfixCalculator();

double calculatePostfixExpression(char* expression);

//============================================================================//
//= Main Program Definition ==================================================//
//============================================================================//
int main() {
  // variables
    // none

  // implement calculator
  runPostfixCalculator();

  // indicate successful completion, return 0
  return 0;
}


//============================================================================//
//= Function Implementation ==================================================//
//============================================================================//

void runPostfixCalculator() {
  // variables
  char* expression = new char [MAX_EXPRESSION];
  double opResult = -8.88;

  // prompt user for expression
  cout << endl;
  cout << "Enter expression in postfix form : ";
  cin.getline(expression, (MAX_EXPRESSION - 1));

  // calculate the expression 
  opResult = calculatePostfixExpression(expression);

  // display result
  cout << " = " << setprecision(2)
       << opResult << endl;

  // return dynamic memory
  delete expression;

  // no return - void
}

double calculatePostfixExpression(char* expression) {
  // variables
  int ndx = 0;
  char current_char = char(4);
  StackLinked<double> C;
  StackLinked<double>* contents = new StackLinked<double>;
  double operand1 = -8.88;    // arbitrary values for garbage
  double operand2 = -8.88;    // clearing and debugging
  double result = -8.88;

  // calculate the expression
  for(ndx = 0; ndx < MAX_EXPRESSION; ndx++) {
    // extract item from expression
    current_char = expression[ndx];

    // take appropriate action
    switch(current_char) {
      case '\0':
        // stop processing when \0 is encountered 
        ndx = MAX_EXPRESSION;   // this will break loop
        break;

      // ignore whitespace
      case ' ':
        // do nothing
        break;

      // store numbers in stack
      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
        contents->push( double(current_char - '0') );
        break;

      // perform specified operations
      case '+':
         // gather operands
         operand2 = contents->pop();
         operand1 = contents->pop();

         // do the math
         result = operand1 + operand2;

         // store the result
         contents->push(result);
         break;

      case '-':
         // gather operands
         operand2 = contents->pop();
         operand1 = contents->pop();

         // do the math
         result = operand1 - operand2;

         // store the result
         contents->push(result);
         break;

      case '*':
         // gather operands
         operand2 = contents->pop();
         operand1 = contents->pop();

         // do the math
         result = operand1 * operand2;

         // store the result
         contents->push(result);
         break;

      case '/':
         // gather operands
         operand2 = contents->pop();
         operand1 = contents->pop();

         // do the math
         result = operand1 / operand2;

         // store the result
         contents->push(result);
         break;

      default:
        // ignore all other characters, i.e. do nothing
        break;
    }
  }

  // save the result before returning dynamic memory
  result = contents->pop();
  
  // return dynamic memory
  delete contents;
  contents = NULL;

  // return the result
  return result;
}
