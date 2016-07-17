////////////////////////////////////////////////////////////////////////////////
// 
//  Title: QueueArray.cpp 
//  Created By: Terence Henriod
//  Reviewed By:
//  Course:     
//
//  Summary: An array based, templated queue data container that utilizes an
//           array wrapping technique
// 
//  Last Modified: 
//
////////////////////////////////////////////////////////////////////////////////

//============================================================================//
//= Header Files =============================================================//
//============================================================================//

// class definition header
#include "QueueArray.h"

// other headers/namespaces
#include <iostream>
using namespace std;


//============================================================================//
//= Global (Static) Variables ================================================//
//============================================================================//

//============================================================================//
//= Function Implementation ==================================================//
//============================================================================//


// Constructor(s) //////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: QueueArray
// Summary:       The default constructor for an array based queue templated
//                data container
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
QueueArray<DataType>::QueueArray(int maxNumber) {
  // initialize data members
  maxSize = maxNumber;
  front = -1;  // -1 is the traditional way to indicate that front and back do
  back = -1;   // not point to valid indices
  dataItems = NULL;

  // attempt dynamic memory allocation
  dataItems = new DataType [maxNumber];
    // TODO: catch allocation error if any?

  // no return - constructor
}


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: QueueArray
// Summary:       The copy constructor for an array based templated data
//                container. Clones the "other" parameter into *this.
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
QueueArray<DataType>::QueueArray(const QueueArray& other) {
  // initialize data members
  maxSize = NULL;
  front = -1;
  back = -1;
  dataItems = NULL;

  // use the overloaded = operator to avoid code replication
  *this = other;

  // no return - constructor
}


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: operator =
// Summary:       The overloaded assignment operator, clones parameter "other"
//                into this
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
QueueArray<DataType>& QueueArray<DataType>::operator=(const QueueArray<DataType>& other) {
  // variables
  int cursor = -1;

  // check to see if *this is being copied into itself
  if( this == &other ) {
    // return *this
    return *this;
  }
  // otherwise, copy other
  else {
    // case: other has different maxSize => deconstruct *this to prepare for 
    // copying
    if( maxSize != other.maxSize ) {
      // clone maxSize variable
      maxSize = other.maxSize;

      // check to see if there was any dynamic memory used
      // this check assumes that if dynamic memory was used then dataItems will
      // have a non-NULL value, since it is always initialized to NULL in the
      // constructors
      if( dataItems != NULL ) {
        // return the dynamic memory
        delete [] dataItems;
      }

      // get new dynamic memory
      dataItems = new DataType [maxSize];
    }

    // clone the front and back cursors
    front = other.front;
    back = other.back;
    cursor = front;

    // queue up the elements of other until there are no more to copy
    // case: other has one element
    if( (front == back) && (front > -1) ) {
      // copy the data
      dataItems[cursor] = other.dataItems[cursor];
    }

    // case: other has multiple elements
    else if( front > -1 ) {
      // copy all elements
      do {
        // check if cursor needs to wrap around the end of array
        if( cursor == maxSize ) {
          // put cursor at start of array
          cursor = 0;
        }

        // copy element
        dataItems[cursor] = other.dataItems[cursor];

        // increment cursor
        ++ cursor;

      } while( cursor != (back + 1) );
    }

    // case: other is empty
      // do nothing
  }

  // return *this
  return *this;
}

// Destructor //////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: ~QueueArray
// Summary:       The destructor for an array based queue
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
QueueArray<DataType>::~QueueArray() {
  // return the dynamic memory used
  delete [] dataItems;

  // clean up dangling pointers
  dataItems = NULL;

  // no other action is needed since there is nothing else to clean up

  // no return - destructor
}



// Mutators ////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: enqueue
// Summary:       Pushes an element into the back of the queue
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
void QueueArray<DataType>::enqueue(const DataType& newDataItem)
    throw (logic_error) {
  // check if queue can't hold any more items
  if( isFull() ) {
    // display error message
    cout << "Error: Queue is full, object cannot be added."  << endl;
    // TODO: refine by throwing/catching exception?
  }
  // otherwise, add the item
  else {
    // case: queue is empty
    if( isEmpty() ) {
      // update front and back cursors
      front = 0;
      back = 0;
    }
    // case: queue has at least one item
    else {
      // advance the back value
        // increment the back value
        ++ back;

        // case: the queue has wrapped around the end of the array
        if( back == maxSize ) {
          // start back back at start of array
          back = 0;
        }
    }

    // place the data in the array at the now updated back position
    dataItems[back] = newDataItem;
  }
  // no return - void
}


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: dequeue
// Summary:       Pops an element off the front of the queue and returns its
//                value
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
DataType QueueArray<DataType>::dequeue() throw (logic_error){
  // variables
  int return_cursor = front;

  // check to see that queue does contain data
  if( isEmpty() ) {
    // display error message
    cout << "Error: dequeue() on empty queue." << endl;
    // TODO: refine by throwing exception?
  }
  // otherwise, pop the item
  else {
    // adjust front value appropriately
      // increment the front cursor
      ++ front;

      // case: front wrapped around the back of the array
      if( front == maxSize ) {
        // set front to the first element of array
        front = 0;
      }

      // case: queue had only one item and will now be empty
      // that is, front cursor passed the back one |OR| back is in last array
      // position and front wrapped
      if( (front == (back + 1)) 
          || ((front == 0) && (back == (maxSize - 1))) ) {
        // set front and back to values that indicate that the list is empty
        front = -1;
        back = -1;
      }

    // return the data item
    return dataItems[return_cursor];
  }
}


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: clear
// Summary:       Resets the queue to an empty state by setting front and back
//                to values that indicate the queue is empty; all data in the 
//                array is now considered garbage
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
void QueueArray<DataType>::clear() {
  // set front and back to values that indicate empty queue
  front = -1;
  back = -1;
  // Note: since data outside the bounds of front and back is not considered,
  // data in the array does not necessarily need to be destructed

  // no return - void
}



// Accessors ///////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: isEmpty
// Summary:       Returns true if the queue contains no elements
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
bool QueueArray<DataType>::isEmpty() const {
  // return the result of checking the value of the front cursor for validity
  return (front == -1);
}


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: isFull
// Summary:       Returns true if the array had all elements filled with data;
//                i.e. queue contains maxSize elements
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
bool QueueArray<DataType>::isFull() const {
  // make checks to see if the array is full
  // case: front is at start, back at end
  if( (front == 0) && (back == (maxSize - 1)) ) {
    // indicate that queue is full
    return true;
  }
  // case: back is against front
  else if( back == (front - 1) ) {
    // indicate that queue is full
    return true;
  }
  // otherwise, queue is not full
  else {
    // indicate that queue is not full
    return false;
  }
}



// Programming Exercise 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: putFront
// Summary:       Pushes an element into the front of the queue
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
void QueueArray<DataType>::putFront(const DataType& newDataItem)
    throw (logic_error) {
  // check to see if list is full
  if( isFull() ) {
    // display error message
    cout << "Error: putFront() used when queue is full." << endl;
    // TODO: refine by throwing exception
  }
  // otherwise, add data to queue
  else {
    // case: queue is empty
    if( isEmpty() ) {
      // update front and back cursors
      front = 0;
      back = 0;
    }
    // case: queue has at least one element
    else {
      // update front cursor
        // decrement front
        -- front;

        // check to see if front has wrapped around front of array
        if( front < 0 ) {
          // set front to element at end of array
          front = (maxSize - 1);
        }
    }

    // store data to new front position in array
    dataItems[front] = newDataItem;
  }

  // no return - void
}


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: getRear
// Summary:       Pops the element at the end of the queue off and returns its
//                value
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
DataType QueueArray<DataType>::getRear() throw (logic_error) {
  // variables
  int return_cursor = -1;

  // check to see if queue is empty
  if( isEmpty () ) {
    // display error message
    cout << "Error: getRear() on empty queue." << endl;
    // TODO: refine by throwing exception
  }
  // otherwise, get the data item
  else {
    // save the cursor position
    return_cursor = back;

    // case: the queue only had one item
    if( front == back ) {
      // set the front and back values to indicate an empty array
      front = -1;
      back = -1;
    }
    // case: the queue had more than one item
    else {
      // update the back position
        // decrement back
        -- back;

        // check to see if back wrapped around front of array
        if( back < 0 ) {
          // set back to the last element of the array
          back = (maxSize - 1);
        }
    }

    // return the data item
    return dataItems[return_cursor];
  }
}


// Programming Exercise 3 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: getLength
// Summary:       Returns the number of elements currently contained in the 
//                queue
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
int QueueArray<DataType>::getLength() const {
  // variables
  int length = -8;   // garbage clearing/error testing value
  int interim = 0;   // an intermediate arithmetic variable for modularity

  // case: front cursor value is less than that of back
  if( front < back ) {
    // the length is the difference of front and back + 1
    length = (back - front + 1);
  }
  // case: back cursor value is less than that of front
  else if( back < front ) {
    // find difference between maxSize and front for end elements of array
    interim = (maxSize - front);
    // add the value of back for elements at front of array
    interim += back;

    // length is the intermediate result + 1
    length = (interim + 1);
  }
  // case: front == back and they hold a valid reference
  else if( (front == back) && (front >= 0) ) {
    // the queue has onyl 1 element
    length = 1;
  }
  // otherwise the list is empty
  else {
    // length is 0
    length = 0;
  }

  // return length
  return length;
}



////////////////////////////////////////////////////////////////////////////////
//
// Function Name: showStructure
// Summary:       Displays the contents of the array based queue, assuming that
//                the << operator is functional for the contained data type
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
void QueueArray<DataType>::showStructure() const {
  /*  Code provided in the lab materials, merely copied and pasted to preserve
      functionality of testing files */

// Array implementation. Outputs the data items in a queue. If the
// queue is empty, outputs "Empty queue". This operation is intended
// for testing and debugging purposes only.

    int j;   // Loop counter

    if ( front == -1 )
       cout << "Empty queue" << endl;
    else
    {
       cout << "Front = " << front << "  Back = " << back << endl;
       for ( j = 0 ; j < maxSize ; j++ )
           cout << j << "\t";
       cout << endl;
       if ( back >= front )
          for ( j = 0 ; j < maxSize ; j++ )
              if ( ( j >= front ) && ( j <= back ) )
                 cout << dataItems[j] << "\t";
              else
                 cout << " \t";
       else
          for ( j = 0 ; j < maxSize ; j++ )
              if ( ( j >= front ) || ( j <= back ) )
                 cout << dataItems[j] << "\t";
              else
                 cout << " \t";
       cout << endl;
    }
}


