#include "ListLinked.h"
#include <iostream>
#include <stdexcept>
using namespace std;

  // Constructor(s) ////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: List   [default]
// Summary:       The default constructor. Creates a new instance of a singly
//                linked templated list with no nodes.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
List<DataType>::List( int ignored ) {
  // intitialize data members
  cursor = NULL;
  head = NULL;
}
  
////////////////////////////////////////////////////////////////////////////////
//
// Function Name: List 
// Summary:       An overloaded constructor for list, specifically a copy
//                constructor. Creates an instance of a list identical to the 
//                one given as a parameter.
//
// Parameters:    List& other   the list given to be copied.
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
List<DataType>::List( const List& other ) {
  // variables
  ListNode* tempOtherCursor = NULL;
  ListNode* cursorPosition = NULL;

  // initialize data members
  cursor = NULL;
  head = NULL;

  // case: other is empty
  if(other.isEmpty()) {
    // do nothing; nothing to copy
  }
  // case: other has data
  else {
    // initialize the temporary cursor
    tempOtherCursor = other.head;

    // keep copying the other list nodes until there are no more
    do {
      // create a node with the data in the other list
      insert(tempOtherCursor->dataItem);

      // if the other node is the cursored one, save the corresponding 
      // node address in *this
      if(tempOtherCursor == other.cursor) {
        // save the address for later
        cursorPosition = cursor;
      }

      // advance the temporary other cursor
      tempOtherCursor = tempOtherCursor->next;

    } while(tempOtherCursor != NULL);

    // finish up by setting the pointing the cursor to the node corresponding to
    // the cursored node in other
    cursor = cursorPosition;
  }

  // no return - constructor
}

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: operator=
// Summary:       Overloaded assignment operator. Clears the list and duplicates
//                the right hand side parameter called other. 
//
// Parameters:    List& other
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
List<DataType>& List<DataType>::operator= ( const List& other ) {
  // variables
  ListNode* tempOtherCursor = NULL;
  ListNode* cursorPosition = NULL;

  // clear the list
  clear();

  // case: other is empty
  if(other.isEmpty()) {
    // do nothing; nothing to copy
  }
  // case: other has data
  else {
    // initialize the temporary cursor
    tempOtherCursor = other.head;

    // keep copying the other list nodes until there are no more
    do {
      // create a node with the data in the other list
      insert(tempOtherCursor->dataItem);

      // if the other node is the cursored one, save the corresponding 
      // node address in *this
      if(tempOtherCursor == other.cursor) {
        // save the address for later
        cursorPosition = cursor;
      }

      // advance the temporary other cursor
      tempOtherCursor = tempOtherCursor->next;

    } while(tempOtherCursor != NULL);

    // finish up by setting the pointing the cursor to the node corresponding to
    // the cursored nod in other
    cursor = cursorPosition;
  }

  // return *this
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: ~List
// Summary:       The destructor. Returns all dynamic memory used by the list.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
List<DataType>::~List() {
  // clear the list to return the memory
  clear();

  // no other actions at this time required
}


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: ListNode     [default]
// Summary:       The default constructor for ListNode. Creates an instance
//                of a new node for the list using the data and next pointer
//                parameters given.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
List<DataType>::ListNode::ListNode( const DataType& nodeData,
                                    ListNode* nextPtr ) {
  // intitialize data members with given parameters
  dataItem = nodeData;
  next = nextPtr;

  // no return - constructor
}



  // mutators //////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: insert
// Summary:       Inserts a new node into the list after the cursor position. 
//                Creates a new node with the data parameter given, inserts it 
//                into the list, and sets the cursor to the new node.
//
// Parameters:    DataType& newDataItem
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
void List<DataType>::insert( const DataType& newDataItem ) 
                             throw (logic_error) {
  // variables
  ListNode* newNode = NULL;

  // case: list is full
  if(isFull()) {
    // do nothing, stop execution
    return;
    // TODO: if this is to become truly functional, throw exception
  }
  // case: list not full
  else {
    // create a new node with the parameter data
    newNode = new ListNode(newDataItem, NULL);

    // if the list is empty, update the head pointer
    if(isEmpty()) {
      head = newNode;
      cursor = newNode;
    }
    // otherwise the list is not empty
    else {
      // if the new node will not be at the end of the list, link it to what will
      // be the following node
      if(cursor->next != NULL) {
        newNode->next = cursor->next;
      }
      // complete the linking process
        // link the cursored node to the new node
        cursor->next = newNode;

        // advance the cursor
        cursor = newNode;
    }
  }

  // no return - void
}

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: remove
// Summary:       Removes and deletes the node currently pointed to, then
//                advances the cursor to the next data node if possible. If the
//                node being deleted is the tail, the cursor is moved to the 
//                head. If the list is empty, nothing is done.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
void List<DataType>::remove() throw (logic_error) {
  // variables
  ListNode* deletingCursor = cursor;

  // case: list is empty
  if(isEmpty()) {
    // end function execution
    return;
    // TODO: refine by throwing exception
  }
  // case: list is not empty
  else {
    // re-link as appriopriate
    // case: cursored node is only node
    if((cursor == head) && (cursor->next == NULL)) {
      // reset cursor and head
      cursor = NULL;
      head = NULL;
    }
    // case: cursored node is first node in multinode list
    else if(cursor == head) {
    // advance cursor and head
    gotoNext();
    head = cursor;
    }
    // case: cursored node is not first in multinode list
    else {
      // find prior node, link prior to next
      gotoPrior();
      cursor->next = deletingCursor->next;

      // if the deleted node is not the last in the list, advance the cursor
      if(deletingCursor->next != NULL) {
      gotoNext();
      }
      // otherwise move cursor to beginning
      else {
      gotoBeginning();
      } 
    }

    // delete the node
    delete deletingCursor;
  }

  // no return - void
}

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: replace
// Summary:       Replaces the data item held by the cursored node. If the list
//                is empty nothing is done.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
void List<DataType>::replace( const DataType& newDataItem ) 
                              throw (logic_error) {
  // case: the list is empty
  if(isEmpty()) {
    // stop execution by returning
    return;
    // TODO: to improve class, throw exception
  } 
  // case: list is not empty
  else { 
    // replace the data contents of the current node
    cursor->dataItem = newDataItem;
  }

  // no return - void
}

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: clear
// Summary:       Clears the list by removing nodes until the list is empty. 
//                Resets the head and cursor pointers to NULL to indicate
//                emptiness. List will then be empty.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
void List<DataType>::clear() {
  // start at the beginning of the list
  gotoBeginning();

  // while the list is not empty, remove the nodes
  while(!isEmpty()) {
    // remove nodes one at a time
    remove();
  }

  // no return - void
}


  // accessors /////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: isEmpty
// Summary:       Returns the state of the list: true for empty, false if not
//
// Parameters:    none
// Returns:       bool   the state of the list
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
bool List<DataType>::isEmpty() const {
  // if head is pointing to null, the list is empty
  if(head == NULL) {
    // indicate that the list is empty
    return true;
  } 
  // otherwise, return false, list not empty
  else {
    // indicate that the list is not empty
    return false;
  }
}

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: isFull
// Summary:       Indicates the capacity state of the list. Because this is a 
//                trivial list implementation, we are assuming the list will
//                never be full - and will always return false.
//
// Parameters:    none
// Returns:        bool     false
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
bool List<DataType>::isFull() const {
  /* Per the lab manual, we are to assume that we will never run out of memory,
     i.e. that the list is never full. */ 
  return false;
}

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: gotoBeginning
// Summary:       Moves the cursor to the beginning of the list. Sets cursor to
//                NULL if list is empty
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
void List<DataType>::gotoBeginning() throw (logic_error) {
  // simply move the cursor to the beginning of the list
  cursor = head;  // note: this will work even if the list is empty because 
                  // cursor will simply be made to point to NULL

  // no return - void
}

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: gotoEnd
// Summary:       Advances the cursor to the last node in the list. Does nothing
//                if the list is empty.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
void List<DataType>::gotoEnd() throw (logic_error) {
  // if the list is not empty, advance the cursor to the last node
  if(!isEmpty()) {
    while(gotoNext()) {
      // the gotoNext function will provide its own signal to continue
    }
  }
  // otherwise the list is empty and nothing should be done
  else {
  // TODO: refine class by throwing exception?
  }

  // no return - void
}

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: gotoNext
// Summary:       Attempts to advance the cursor to the next node. Returns a
//                value to indicate the success of the operation.
//
// Parameters:    none
// Returns:       bool     success of operation
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
bool List<DataType>::gotoNext() throw (logic_error) {
  // if the list is not empty, attempt the operation
  if(cursor != NULL) {
    // if there is a next node, update the cursor
    if(cursor->next != NULL) {
      //update the cursor
      cursor = cursor->next;

      // operation was successful, return true
      return true;
    }
    // otherwise, operation cannot be completed
    else {
      // return false to indicate failure
      return false;
    }
  }
  // otherwise, the list is empty and operation cannot be completed
  else {
    // return false to indicate failure
    return false;
  }
}

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: gotoPrior
// Summary:       Locates the node previous to the cursored node if possible and
//                then places the cursor at the previous node. Does nothing if 
//                the list is empty or there is not prior node. Returns a value
//                to indicate the success of the operation.
//
// Parameters:    none
// Returns:       bool     the success of the operation
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
bool List<DataType>::gotoPrior() throw (logic_error) {
  // variablea
  ListNode* tempCursor = head;

  // if the list is not empty, find the node before the cursor
  if(!isEmpty()) {
    // case: cursor is at first node && list has one node
    if(cursor == head) {
      // operation can't be performed, signal that
      return false;
    }
    // otherwise find the prior node
    else {
      // iterate through the list
      while(tempCursor->next != cursor) {
        // advance the temporary cursor
        tempCursor = tempCursor->next;
      }

      // now that the prior node has been found, point hte cursor at it
      cursor = tempCursor;

      // indicate that the operation was successfull
      return true;
    }
  }
  // otherwise the list is empty
  else {
    // the operation will not work, indicate by returning false
    return false;
  }
}

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: getCursor
// Summary:       Simply returns the data item held by the cursored node.
//                Returns 0 if the list is empty.
//
// Parameters:    none
// Returns:       DataType     the cursored data item
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
DataType List<DataType>::getCursor() const throw (logic_error) {
  // case: cursor indicates a node
  if(cursor != NULL) {
    // return the cursored data
    return cursor->dataItem;
  }
  // case: list is empty
  else {
    // return the equivalent of NULL
    return 0;
    // TODO: refine class by throwing exception
  }
}

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: showStructure
// Summary:       Code provided in the lab documents. Displays the contents of 
//                the list assuming the << operator has been overloaded.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
void List<DataType>::showStructure() const {
 if ( isEmpty() )
    {
       cout << "Empty list" << endl;
    } 
    else
    {
	for (ListNode* temp = head; temp != NULL; temp = temp->next) {
	    if (temp == cursor) {
		cout << "[";
	    }

	    // Assumes that dataItem can be printed via << because
	    // is is either primitive or operator<< is overloaded.
	    cout << temp->dataItem;	

	    if (temp == cursor) {
		cout << "]";
	    }
	    cout << " ";
	}
	cout << endl;
    }
}

// exercises 2 and 3 ///////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// Function Name: moveToBeginning
// Summary:       Moves the cursored node to the beginning. If list is empty,
//                does nothing.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
void List<DataType>::moveToBeginning () throw (logic_error){
  // variables
  ListNode* newHead = NULL;

  // case: list is empty
  if(isEmpty()) {
    // do nothing; stop execution
    return;
    // TODO: refine by throwing exception?
  }
  // case: list is not empty
  else {
    // case: cursored node is at head (also node is only node)
    if(cursor == head) {
      // node is at beginning; do nothing; return to end execution
      return;
    }
    // case: cursored node is not first in multinode list
    else {
      // create a new node to be the first Node, link it to the current head
      newHead = new ListNode(getCursor(), head);

      // update the head
      head = newHead;

      // remove the cursored node
      remove();

      // ensure cursor points at the new head node
      cursor = head;
    }
  }

  // no return - void
}

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: insertBefore
// Summary:       Inserts a new node into the list with the given data
//                parameter before the cursored node. If list is empty, the
//                inserted node will be the only node in the list. The cursor
//                is positioned at the newly created node.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
void List<DataType>::insertBefore(const DataType& newDataItem)
                                  throw (logic_error){
  // variables
  ListNode* newNode = NULL;

  // case: list is empty
  if(isEmpty()) {
    // create a new node with parameter data
    newNode = new ListNode(newDataItem, NULL);

    // make the node the first and only node
    cursor = newNode;
    head = newNode;
  }
  // case: list is not empty
  else {
    // create the new node using the current cursored node's data
    newNode = new ListNode(getCursor(), NULL);

    // link the new node to the node that will follow it
    newNode->next = cursor->next;

    // replace the cursored node's data with the parameter data
    replace(newDataItem);

    // re-link the cursored node to the new one for correctly ordered list
    cursor->next = newNode;
  
  /* head does not need to be updated; if the new node took the place of the 
     first node, then head is already pointing at the now first node */

  /* after these operations, cursor points at the new item; all items after the
     new item have been advanced one position in the list */
  }

  // no return - void
}



