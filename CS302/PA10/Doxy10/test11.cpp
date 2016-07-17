//--------------------------------------------------------------------
//
//  Laboratory 11                                          test11.cpp
//
//  Test program for the operations in the Heap ADT
//
//--------------------------------------------------------------------

#include <iostream>
#include <string>
#include <cctype>

using namespace std;

#include "Heap.cpp"
#include "config.h"

//--------------------------------------------------------------------
// Prototypes

void printHelp();

//--------------------------------------------------------------------
//
// Declaration for the heap data item class
//

template < typename KeyType >
class TestDataItem
{
  public:
    TestDataItem () 
	{ priority = -1; }

    void setPriority ( KeyType  newPty )
        { priority = newPty; }            // Set the priority

    KeyType  getPriority () const
        { return priority; }              // Returns the priority

  private:
    KeyType  priority;                    // Priority for the data item
};

template < typename KeyType=int >
class Greater {
  public:
    bool operator()( const KeyType &a, const KeyType &b) const { return a > b; }
};

int main()
{
    // Greater<> uses the default int type for its KeyType
    Heap<TestDataItem<int>, int, Greater<> > testHeap(8);  // Test heap
    TestDataItem<int> testDataItem;       // Heap data item
    int  inputPty;                        // User input priority
    char cmd;                             // Input command

    printHelp();

    do
    {
        testHeap.showStructure();                     // Output heap

        cout << endl << "Command: ";                  // Read command
        cin >> cmd;
	cmd = toupper( cmd );			      // Upcase input
        if ( cmd == '+' )
           cin >> inputPty;

        switch ( cmd )
        {
          case 'H' :
               printHelp();
               break;

          case '+' :                                  // insert
               testDataItem.setPriority(inputPty);
               cout << "Insert : priority = " << testDataItem.getPriority()
                    << endl;
               testHeap.insert(testDataItem);
               break;

          case '-' :                                  // remove
               testDataItem = testHeap.remove();
               cout << "Removed data item : "
                    << " priority = " << testDataItem.getPriority() << endl;
               break;

          case 'C' :                                  // clear
               cout << "Clear the heap" << endl;
               testHeap.clear();
               break;

          case 'E' :                                  // isEmpty
               if ( testHeap.isEmpty() )
                  cout << "Heap is empty" << endl;
               else
                  cout << "Heap is NOT empty" << endl;
               break;

          case 'F' :                                  // isFull
               if ( testHeap.isFull() )
                  cout << "Heap is full" << endl;
               else
                  cout << "Heap is NOT full" << endl;
               break;

#if LAB11_TEST1
          case 'W' :                              // Programming Exercise 3
               cout << "Levels :" << endl;
               testHeap.writeLevels();
               break;
#endif	// LAB11_TEST1

          case 'Q' :                              // Quit test program
               break;

          default :                               // Invalid command
               cout << "Inactive or invalid command" << endl;
        }
    }
    while ( cmd != 'Q' );

    return 0;
}

//--------------------------------------------------------------------

void printHelp()
{
    cout << endl << "Commands:" << endl;
    cout << "  H    : Help (displays this message)" << endl;
    cout << "  +pty : Insert data item with priority pty" << endl;
    cout << "  -    : Remove highest priority data item" << endl;
    cout << "  C    : Clear the heap" << endl;
    cout << "  E    : Empty heap?" << endl;
    cout << "  F    : Full heap?" << endl;
    cout << "  W    : Write levels   ("
#if LAB11_TEST1
	    "Active   "
#else
	    "Inactive "
#endif	//LAB11_TEST1
	 << ": Programming Exercise 3)" << endl;


    cout << "  Q    : Quit the test program" << endl;
    cout << endl;
}
