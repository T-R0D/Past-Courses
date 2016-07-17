//--------------------------------------------------------------------
//
//  Laboratory 6                                           test6.cpp
//
//  Test program for the operations in the Stack ADT
//
//--------------------------------------------------------------------
#include <iostream>

using namespace std;

#include "config.h"

#if !LAB6_TEST1
#  include "StackArray.cpp"
#else
#  include "StackLinked.cpp"
#endif

void print_help()
{
    cout << endl << "Commands:" << endl;
    cout << "  H  : Help (displays this message)" << endl;
    cout << "  +x : Push x" << endl;
    cout << "  -  : Pop" << endl;
    cout << "  C  : Clear" << endl;
    cout << "  E  : Empty stack?" << endl;
    cout << "  F  : Full stack?" << endl;
    cout << "  Q  : Quit the test program" << endl;
    cout << endl;
}

template <typename DataType>
void test_stack(Stack<DataType>& testStack) 
{
    DataType testDataItem;            // Stack data item
    char cmd;                         // Input command

    print_help();

    do
    {
        testStack.showStructure();                    // Output stack

        cout << endl << "Command: ";                  // Read command
        cin >> cmd;
        if ( cmd == '+' )
           cin >> testDataItem;

	try {
	    switch ( cmd )
	    {
	      case 'H' : case 'h':
		   print_help();
		   break;

	      case '+' :                                  // push
		   cout << "Push " << testDataItem << endl;
		   testStack.push(testDataItem);
		   break;

	      case '-' :                                  // pop
		   cout << "Popped " << testStack.pop() << endl;
		   break;

	      case 'C' : case 'c' :                       // clear
		   cout << "Clear the stack" << endl;
		   testStack.clear();
		   break;

	      case 'E' : case 'e' :                       // isEmpty
		   if ( testStack.isEmpty() )
		      cout << "Stack is empty" << endl;
		   else
		      cout << "Stack is NOT empty" << endl;
		   break;

	      case 'F' : case 'f' :                       // isFull
		   if ( testStack.isFull() )
		      cout << "Stack is full" << endl;
		   else
		      cout << "Stack is NOT full" << endl;
		   break;

	      case 'Q' : case 'q' :                   // Quit test program
		   break;

	      default :                               // Invalid command
		   cout << "Inactive or invalid command" << endl;
	    }
	} 
	catch (logic_error e) {
	    cout << "Error: " << e.what() << endl;
	}

    } while ( cin && cmd != 'Q' && cmd != 'q' );
}

int main() {
#if !LAB6_TEST1
    cout << "Testing array implementation" << endl;
    StackArray<char> s1;
    test_stack(s1);
#else
    cout << "Testing linked implementation" << endl;
    StackLinked<char> s2;
    test_stack(s2);
#endif
}
