

//  Headers and Namespace //////////////////////////////////////////////////////
#include <iostream>
#include <stdexcept>
using namespace std;

#include "StackLinked.h"


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: StackLinked   [default]
// Summary:       The default constructor. Creates a new instance of a node
//                based stack templated data container
//
// Parameters:    int   maxNumber   ignored, kept to concurr with array based
//                                  implementation of the stack
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
StackLinked<DataType>::StackLinked(int maxNumber /*defaulted parameter*/) {
  // initialize data members
  top = NULL;

  // no return - constructor
}


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: StackLinked   [copy]
// Summary:       The copy constructor for the StackLinked template data 
//                container
//
// Parameters:    StackLinked other   the StackLinked to be cloned into *this
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
StackLinked<DataType>::StackLinked(const StackLinked& other) {
  //initialize data members
  top = NULL;

  // variables
  StackNode* other_cursor = other.top;
  StackNode* previous_node = NULL;
  StackNode* new_node = NULL;  

  // use the already coded overloaded assignment operator to carry out cloning
  *this = other;

  // no return - constructor
}


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: operator =
// Summary:       Clones other into *this
//
// Parameters:    StackLinked& other   the StackLinked to be cloned
// Returns:       *this
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
StackLinked<DataType>& StackLinked<DataType>::operator=(
    const StackLinked& other) {
  // variables
  StackNode* other_cursor = other.top;
  StackNode* previous_node = NULL;
  StackNode* new_node = NULL;  

  // prevent copying *this into itself - ensure other is not *this
  if( this == &other ) {
    // end execution, return *this
    return *this;
  }

  // case: other stack is empty
  if( other.isEmpty() ) {
    // do nothing
    // TODO: refine by throwing exception?
  }
  // case: other stack has contents
  else {
    // clone the other stack into *this
    // start by copying the top node
    top = new StackNode(other.top->dataItem, NULL);

    // prep variables to continue copying process
    previous_node = top;

    // copy any remaining nodes
    while( other_cursor->next != NULL ) {
      // advance the other cursor
      other_cursor = other_cursor->next;

      // clone a new node with the data contained by the other stack,
      // don't link it yet
      new_node = new StackNode(other_cursor->dataItem, NULL);

      // link the nodes appropriately
      previous_node->next = new_node;

      // update the previous_node pointer for the next node cloning
      previous_node = previous_node->next;

      // loop breaks once we have copied other's bottom node
      }
  }

  // return *this
  return *this;
}


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: ~StackLinked
// Summary:       The destructor. Clears the stack to prepare it for destruction
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
StackLinked<DataType>::~StackLinked() {
  // clear *this
  clear();

  // no return - destructor
}



////////////////////////////////////////////////////////////////////////////////
//
// Function Name: StackNode
// Summary:       The default constructor for a StackNode. Instantiates a new
//                StackNode with the given parameters for data.
//
// Parameters:    DataType& nodeData     the data of the DataType used in the 
//                                       template
//                StackNode* nextPtr     a pointer to the next StackNode in the
//                                       data structure
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
StackLinked<DataType>::StackNode::StackNode(const DataType& nodeData, 
                                            StackNode* nextPtr) {
  // initialize data members with given data
  dataItem = nodeData;
  next = nextPtr;

  // no return - constructor
}


//  Mutators ///////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: push
// Summary:       Pushes (adds) the parameter newDataItem on the stack.
//                Should throw exception if the stack is full.
//
// Parameters:    DataType& newDataItem     A reference to the templated
//                                          DataType item to be added to the 
//                                          stack
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
void StackLinked<DataType>::push(const DataType& newDataItem) 
                                 throw (logic_error){
  // variables
  StackNode* new_node = NULL;

  // ensure stack has room
  if( !isFull() ) { 
    // case: stack is empty
    if( isEmpty() ) {
      // create the first node to be the base
      new_node = new StackNode(newDataItem, NULL);
    }
    // case: stack is not empty
    else {
      // create the new node that points down to the old top
      new_node = new StackNode(newDataItem, top);
    }

    // update top pointer
    top = new_node;

    // clean up dangling pointers
    new_node = NULL;
  }
  // otherwise, stack is full
  else {
    // do nothing
    
    // TODO: refine by throwing exception?
  }

  // no return - void
}


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: pop
// Summary:       pops (removes) the top data node of the stack and returns it.
//                should throw exception if the stack is empty
//
// Parameters:    none
// Returns:       the item of the templated DataType at the top of the stack
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
DataType StackLinked<DataType>::pop() throw (logic_error) {
  // variables
  DataType top_data;
  StackNode* old_top;

  // case: stack is empty
  if( isEmpty() ) {
    // return error value
    return 0;
    // TODO: refine by throwing exception
  }
  // case: stack has contents
  else {
    // save the location of the top item, copy the top data item
    old_top = top;
    top_data = top->dataItem;

    // update top
    top = top->next;  // works even on last node

    // remove the data item from the stack
    delete old_top;
    old_top = NULL;
        
    // return the data item
    return top_data;
  }
}


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: clear
// Summary:       Removes (deletes) all nodes in the stack until the stack is 
//                empty
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
void StackLinked<DataType>::clear() {
  // pop off items until the stack is empty
  while( !isEmpty() ) {
    // pop items off
    pop();
  }

  // no return - void
}


//  Accessors //////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: isEmpty
// Summary:       Returns T/F information pertaining to whether or not the 
//                data container holds any nodes
//
// Parameters:    none
// Returns:       bool   true if stack has no nodes; false otherwise
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
bool StackLinked<DataType>::isEmpty() const{
  // if top points to NULL, list is empty
  if( top == NULL ) {
    // list is empty, indicate that
    return true;
  }
  // otherwise, list is not empty
  else {
    // indicate list is not empty
    return false;
  }

//  // return top, if top == NULL list is empty, if not, list is not empty
//  return top;   // may not work in other languages /* or this one */
}


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: isFull
// Summary:       Returns T/F information pertaining to whether or not the 
//                data container can accept another node. because this is a 
//                trivial implementation, we are assuming that the stack
//                is never full
//
// Parameters:    none
// Returns:       bool   always false
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
bool StackLinked<DataType>::isFull() const {
  /* we will always assume that the stack is not full given that this is a 
     trivial implementation of a stack class */

  // return false to indicate stack is not full
  return false;
}



////////////////////////////////////////////////////////////////////////////////
//
// Function Name: showStructure
// Summary:       displays the contents of the stack from top to bottom. the
//                templated DataType must have the << operator overloaded for 
//                this function to work properly
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
void StackLinked<DataType>::showStructure() const 
/* this function is provided in the collection of student 
   materials */


// Linked list implementation. Outputs the data elements in a stack.
// If the stack is empty, outputs "Empty stack". This operation is
// intended for testing and debugging purposes only.

{
    if( isEmpty() )
    {
	cout << "Empty stack" << endl;
    }
    else
    {
        cout << "Top\t";
	for (StackNode* temp = top; temp != 0; temp = temp->next) {
	    if( temp == top ) {
		cout << "[" << temp->dataItem << "]\t";
	    }
	    else {
		cout << temp->dataItem << "\t";
	    }
	}
        cout << "Bottom" << endl;
    }

}

