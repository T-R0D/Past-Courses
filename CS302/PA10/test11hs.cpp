//--------------------------------------------------------------------
//
//  Laboratory 11, Programming Exercise 2                 test11hs.cpp
//
//  Test program for for the heapSort() function
//
//--------------------------------------------------------------------

#include <iostream>

using namespace std;

#include "heapsort.cpp"

//--------------------------------------------------------------------

class TestData
{
  public:

    void setPriority ( int newPriority )
        { priority = newPriority; }   // Set the priority

    int getPriority () const
        { return priority; }     // Returns the priority

  private:

    int priority;                // Priority for the data item
};

//--------------------------------------------------------------------

const int MAX_NUM_DATA_ITEMS = 10;

int main ()
{
    TestData testList[MAX_NUM_DATA_ITEMS];     // Array
    int size,                               // Number of data items
        inputPty,                           // Input priority
        j;                                  // Loop counter

    // Read in the array.

    cout << endl << "Enter up to " << MAX_NUM_DATA_ITEMS << " priorities (end with EOF) : ";
    size = 0;
    while ( size < MAX_NUM_DATA_ITEMS  &&  cin >> inputPty )
       testList[size++].setPriority(inputPty);

    // Output the unsorted array.

    cout << "Unsorted array :";
    for ( j = 0 ; j < size ; j++ )
        cout << " " << testList[j].getPriority();
    cout << endl;

    // Sort the array using heap sort.

    heapSort(testList,size);

    // Output the sorted array.

    cout << "Sorted array   :";
    for ( j = 0 ; j < size ; j++ )
        cout << " " << testList[j].getPriority();
    cout << endl;
    
    return 0;
}

