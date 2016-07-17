//--------------------------------------------------------------------
//
//  Laboratory 7                                           test7.cpp
//
//  Test program for the operations in the Queue ADT
//
//--------------------------------------------------------------------

#include <iostream>

#include "config.h"

using namespace std;

#if LAB7_TEST1
#   include "QueueLinked.cpp"
#else
#   include "QueueArray.cpp"
#endif

//--------------------------------------------------------------------

void print_help();

//--------------------------------------------------------------------

template <typename DataType>
void test_queue(Queue<DataType>& testQueue) 
//void test_queue(Queue<char>& testQueue) 
{
    char cmd;                   // Input command
    char testData;              // Queue data item

    print_help();

    do
    {
	try {
	    testQueue.showStructure();                    // Output queue

	    cout << endl << "Command: ";                  // Read command
	    cin >> cmd;
	    if ( cmd == '+'  ||  cmd == '>' )
		cin >> testData;

	    switch ( cmd )
	    {
		case 'H' : case 'h' :
		    print_help();
		    break;

		case '+' :                                  // enqueue
		    cout << "Enqueue " << testData << endl;
		    testQueue.enqueue(testData);
		    break;

		case '-' :                                  // dequeue
		    cout << "Dequeued " << testQueue.dequeue() << endl;
		    break;

		case 'C' : case 'c' :                       // clear
		    cout << "Clear the queue" << endl;
		    testQueue.clear();
		    break;

		case 'E' : case 'e' :                       // empty
		    if ( testQueue.isEmpty() )
			cout << "Queue is empty" << endl;
		    else
			cout << "Queue is NOT empty" << endl;
		    break;

		case 'F' : case 'f' :                       // full
		    if ( testQueue.isFull() )
			cout << "Queue is full" << endl;
		    else
			cout << "Queue is NOT full" << endl;
		    break;

#if LAB7_TEST2
		case '>' :                              // Programming Exercise 2
		    cout << "Put " << testData << " in front " << endl;
		    testQueue.putFront(testData);
		    break;

		case '=' :                              // Programming Exercise 2
		    cout << "Got " << testQueue.getRear() << " from rear "
			<< endl;
		    break;
#endif

#if LAB7_TEST3
		case '#' :                              // Programming Exercise 3
		    cout << "Length = " << testQueue.getLength() << endl;
		    break;
#endif

		case 'Q' : case 'q' :                   // Quit test program
		    break;

		default :                               // Invalid command
		    cout << "Inactive or invalid command" << endl;
	    }
	}
	catch (logic_error e) {
	    cout << "Error: " << e.what() << endl;
	}
    }
    while ( cin && cmd != 'Q'  &&  cmd != 'q' );

    if( !cin ) {
	cout << "input error" << endl;
    }

}

//--------------------------------------------------------------------

int main() 
{
#if !LAB7_TEST1
    cout << "Testing array implementation" << endl;
    QueueArray<char> s1;
    test_queue(s1);
#else
    cout << "Testing linked implementation" << endl;
    QueueLinked<char> s2;
    test_queue(s2);
#endif

    return 0;
}

//--------------------------------------------------------------------

void print_help()
{
    cout << endl << "Commands:" << endl;
    cout << "  H  : Help (displays this message)" << endl;
    cout << "  +x : Enqueue x" << endl;
    cout << "  -  : Dequeue" << endl;
    cout << "  C  : Clear the queue" << endl;
    cout << "  E  : Empty queue?" << endl;
    cout << "  F  : Full queue?" << endl;
    cout << "  >x : Put x at front    ("
#if LAB7_TEST2
         << "  Active "
#else
         << "Inactive "
#endif
	 << ": Programming Exercise 2)"
         << endl;
    cout << "  =  : Get x from rear   ("
#if LAB7_TEST2
         << "  Active "
#else
         << "Inactive "
#endif
	 << ": Programming Exercise 2)"
         << endl;
    cout << "  #  : Length            ("
#if LAB7_TEST3
         << "  Active "
#else
         << "Inactive "
#endif
	 << ": Programming Exercise 3)"
         << endl;
    cout << "  Q  : Quit the test program" << endl;
    cout << endl;
}

