////////////////////////////////////////////////////////////////////////////////
// 
//  Title: QueueLinked.cpp 
//  Created By: Terence Henriod
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

// class definition header
#include "QueueLinked.h"

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
// Function Name: QueueLinked   [default]
// Summary:       The default constructor. Creates a new instance of a node
//                based queue templated data container
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
QueueLinked<DataType>::QueueLinked(int maxNumber /* ignored */) {
  // initialize data members
  front = NULL;
  back = NULL;

  // no return - constructor
}

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: QueueLinked
// Summary:       The copy constructor. Creates a new instance of a node
//                based queue templated data container with data from the 
//                given parameter
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
QueueLinked<DataType>::QueueLinked(const QueueLinked& other) {
  // intitalize data members to indicate empty container
  front = NULL;
  back = NULL;

  // use the overloaded = operator to avoid code replication
  *this = other;

  // no return - constructor
}


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: operator =
// Summary:       Clones the data contained in other into *this
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
QueueLinked<DataType>& QueueLinked<DataType>::operator=(const QueueLinked& other) {
  // variables
  QueueNode* other_cursor = other.front;

  // prevent copying *this into itself
  if( this == &other ) {
    // end funtion execution
    return *this;  // TODO: refine by throwing exception?
  }
  // otherwise, attempt to copy other
  else {
    // clear *this to start fresh
    clear();

    // case: other is empty
    if( other.isEmpty() ) {
      // nothing to copy, return empty queue
      return *this;
    }
    // otherwise, copy nodes
    else {
      // copy nodes until the end of the list has been reached
      while( other_cursor != NULL ) {
        // create a node in *this with the data in other
        enqueue( other_cursor->dataItem );

        // advance other_cursor for the next iteration
        other_cursor = other_cursor->next;
      }
    }
  }

  // end function, return *this
  return *this;
}


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: QueueNode
// Summary:       The default constructor for a single data node for a linked
//                data structure.
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
QueueLinked<DataType>::QueueNode::QueueNode(const DataType& nodeData, QueueNode* nextPtr) {
  // initialize data members with the given parameter data
  dataItem = nodeData;
  next = nextPtr;

  // no return - constructor
}
// Destructor //////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: ~QueueLinked
// Summary:       The destructor for a the QueueLinked templated data container
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
QueueLinked<DataType>::~QueueLinked() {
  // use clear function to remove all nodes and return all dynamic memory
  clear();

  // no return - destructor
}





// Mutators ////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: enqueue
// Summary:       Pushes a new data item into the back of the queue
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
void QueueLinked<DataType>::enqueue(const DataType& newDataItem)
    throw (logic_error) {
  // variables
  QueueNode* new_node = NULL;

  // ensure that an item can be added
  if( isFull() ) {
    // display error message
    cout << "Error: enqueue() on full queue." << endl;
    // TODO: refine by throwing/catching exception
  }
  // otherwise, add the data
  else {
    // create the new node
    new_node = new QueueNode( newDataItem, NULL );

    // add the node to the queue
    // case: queue was empty
    if( isEmpty() ) {
      // update the front pointer
      front = new_node;

      // update the back pointer
      back = new_node;
    }
    // otherwise, append the node to the end of the list (queue)
    else {
      // link the new node to the old back
      back->next = new_node;

      // update the back
      back = new_node;
    }

    // clean up dangling pointers
    new_node = NULL;
  }

  // no return - void
}


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: dequeue
// Summary:       Pops the front-most element from the queue and returns its
//                value
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
DataType QueueLinked<DataType>::dequeue() throw (logic_error){
  // variables
  QueueNode* delete_cursor = NULL;
  DataType the_data;

  // check if queue is empty
  if( isEmpty() ) {
    // display error message
    cout << "Error: dequeue() on empty queue." << endl;
    // TODO: refine by throwing exception
  }
  // otherwise, carry out operation
  else {
    // save location and data of node to be removed
    delete_cursor = front;
    the_data = front->dataItem;

    // case: front is only node
    if( front == back ) {
      // set front and back to NULL, indicating queue will be empty
      front = NULL;
      back = NULL;
    }
    // case: there are more nodes in queue
    else {
      // only update the front pointer
      front = front->next;
    }

    // delete the node
    delete delete_cursor;

    // clean up dangling pointers
    delete_cursor = NULL;

    // return the data
    return the_data;
  }
}


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: clear
// Summary:       Removes all the data from the linked queue
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
void QueueLinked<DataType>::clear() {
  // remove nodes until the queue is empty
  while( !isEmpty() ) {
    // dequeue the current node
    dequeue();
  }

  // no return - void
}



// Accessors ///////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: isEmpty
// Summary:       Returns the status of the queue (if empty, true)
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
bool QueueLinked<DataType>::isEmpty() const {
  // return the result of whether or not front points to anything
  return (front == NULL);
}



////////////////////////////////////////////////////////////////////////////////
//
// Function Name: isFull
// Summary:       Reports if the queue is full or not (if full, true)
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
bool QueueLinked<DataType>::isFull() const {
  /* because this is a trivial implementation so we will assume that the 
     data structure is never full */
  // return false
  return false;
}



// Programming Exercise 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: putFront
// Summary:       Pushes a data item into the front of the queue
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
void QueueLinked<DataType>::putFront(const DataType& newDataItem)
    throw (logic_error) {
  // variables
  QueueNode* new_node = NULL;

  // check if data can't be added to the queue
  if( isFull() ) {
    // display error message
    cout << "Error: putFront() on full queue." << endl;
    // TODO: refine by throwing/catching exception
  }
  // otherwise, add the new node to the front of the queue
  else {
    // create the new node
    new_node = new QueueNode( newDataItem, NULL );

    // case: queue is empty
    if( isEmpty() ) {
      // update front and back to indicate new, solitary element
      front = new_node;
      back = new_node;
    }
    // case: queue not empty
    else {
      // link new node to the old front
      new_node->next = front;

      // update front pointer
      front = new_node;
    }
  }

  // no return - void
}


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: getRear
// Summary:       Pops the back element off the queue and returns its value
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
DataType QueueLinked<DataType>::getRear() throw (logic_error) {
  // variables
  QueueNode* delete_cursor = NULL;
  QueueNode* previous_node = NULL;
  DataType the_data;

  // check if queue has no data
  if( isEmpty() ) {
    // display error message
    cout << "Error: getRear() on empty queue." << endl;
    // TODO: refine by throwing exception?
  }
  // otherwise, perform operation
  else {
    // save back location and data
    delete_cursor = back;
    the_data = back->dataItem;

    // case: queue has only one node
    if( front == back ) {
      // update front and back to indicate list will be empty
      front = NULL;
      back = NULL;
    }
    // case: queue has multiple nodes
    else {
      // find previous node, starting with the front node
      previous_node = front;

      // iterate through queue until previous node is found
      while( previous_node->next != back ) {
        // advance the previous_node pointer
        previous_node = previous_node->next;
      }

      // update the back pointer, be sure to "undangle" next
      back = previous_node;
      back->next = NULL;
    }

    // delete the old back node
    delete delete_cursor;

    // clean up dangling pointers
    delete_cursor = NULL;

    // return the data
    return the_data;
  }
}


// Programming Exercise 3 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: getLength
// Summary:       Returns the number of elements currently in the queue
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
int QueueLinked<DataType>::getLength() const {
  // variables
  QueueNode* cursor = NULL;
  int length = 0;

  // iterate through the queue to find the length
  for( length = 0, cursor = front; cursor != NULL;
       ++ length, cursor = cursor->next ) {} 

  // return the length
  return length;
}


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: showStructure
// Summary:       Displays the elements in the queue, << must be a valid
//                operation for the data type stored in the queue
////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
void QueueLinked<DataType>::showStructure() const {
  /*  Code provided in the lab materials, merely copied and pasted to preserve
      functionality of testing files. Althogh it needed serious debugging to
      become functional */

// Linked list implementation. Outputs the elements in a queue. If
// the queue is empty, outputs "Empty queue". This operation is
// intended for testing and debugging purposes only.

    QueueNode* cursor;   // Iterates through the queue

    if ( isEmpty() )
	cout << "Empty queue" << endl;
    else
    {
	cout << "Front\t";
	for ( cursor = front ; cursor != NULL ; cursor = cursor->next )
	{
	    if( cursor == front ) 
	    {
		cout << '[' << cursor->dataItem << "] ";
	    }
	    else
	    {
		cout << cursor->dataItem << " ";
	    }
	}
	cout << "\trear" << endl;
    }
}



