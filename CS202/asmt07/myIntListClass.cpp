////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      myIntListClass.cpp
//  Created By: Terence Henriod
//  Course:     CS202
//
//  Summary:    The implementation of the class defined in myIntListClass.h
// 
//  Last Modified: 4/12/2013 14:45
//
////////////////////////////////////////////////////////////////////////////////

//============================================================================//
//= Header Files =============================================================//
//============================================================================//

// class definition header
#include "myIntListClass.h"

// other headers
using namespace std;


//============================================================================//
//= Global (Static) Variables ================================================//
//============================================================================//



//============================================================================//
//= Function Implementation ==================================================//
//============================================================================//

// Constructor(s) //////////////////////////////////////////////////////////////
  myIntListClass::myIntListClass()   // default
    {
    // initialize ptrs
    head = NULL;
    tail = NULL;  

    // initialize summary data
    numNodes = 0;

    // no return - constructor
    }

  myIntListClass::myIntListClass( int i )
    {
    // initialize vars
    head = NULL;
    tail = NULL;
    numNodes = 0;

    // create origin node
    head = add(i);

    // no return - constructor    
    }

  myIntListClass::myIntListClass( int* set, int setSize )
    {
    // initialize vars
    head = NULL;
    tail = NULL;
    numNodes = 0;

    // create origin node
    head = addSet( set, setSize );

    // no return - constructor    
    }

// Destructor //////////////////////////////////////////////////////////////////
  myIntListClass::~myIntListClass()
    {
    // vars
    node* currentNode = head;
    node* nextNode; 

    // delete the nodes one at a time until all have been returned
    while( currentNode != NULL )
      {
      // grab the ptr value to the next node
      nextNode = currentNode->next;

      // delete the current node
      delete currentNode;

      // move focus to the next node
      currentNode = nextNode;
      }

    // no return - destructor
    }

// Internal/Maintenance ////////////////////////////////////////////////////////
  void myIntListClass::countNodes()
    {
    // vars
    node* currentNode = head;
    numNodes = 0; 
 
    // count the nodes
    if( currentNode != NULL )
      {
      numNodes ++;

      while( currentNode->next != NULL )
        {
        numNodes ++;
        currentNode = currentNode->next;
        }
      }   

    // no return - void
    }

// Accessors ///////////////////////////////////////////////////////////////////
  myIntListClass::node* myIntListClass::get_head() const
    {
    return head;
    };

  myIntListClass::node* myIntListClass::get_tail() const
    {
    return tail;
    }

  myIntListClass::node* myIntListClass::find( int i) const
    {
    // vars
    node* currentNode = head;

    // search for value of interest
    while( currentNode != NULL )
      {
      if( currentNode->value == i )
        {
        // value is found 
        return currentNode;
        }
      else
        {
        currentNode = currentNode->next;
        }
      }  

     // if the function doesn't find the value, return a null ptr
     currentNode = NULL;
     return NULL;
     }

  void myIntListClass::traverse() const
    {
    // vars
    node* currentNode = head;

    // access each node
    while( currentNode != NULL )
      {
      // display
      cout << currentNode->value << endl;
  
      // move to next
      currentNode = currentNode->next;
      }

    // no return - void
    }

  void myIntListClass::traverse_reverse() const
    {
    // vars
    node* currentNode = tail;

    // access each node
    while( currentNode != NULL )
      {
      // display
      cout << currentNode->value << endl;
  
      // move to next
      currentNode = currentNode->previous;
      }

    // no return - void
    }

  int myIntListClass::size() const
    {
    return numNodes;
    }

// Mutators ////////////////////////////////////////////////////////////////////
  bool myIntListClass::setValue( const int i, const int ndx )
    {
    // vars
    int pos = 0;
    node* ndxNode = head;
    bool success = false;
 
    // prevent an invalid (negative) ndx from being used
    if( ndx < 0 )
      {
      return success;
      }

    // find the node corresponding to the ndx value
    while( pos < ndx )
      {
      // check to see if the end of list has been reached
      if( ndxNode->next == NULL )   
        {
        // ndx isn't valid, return failure
        return success;
        }
      
      // move to next ndx position
      ndxNode = ndxNode->next;
      pos ++;
 
      // loop breaks if appropriate ndxNode is reached
      }

    // replace the list element's value
    ndxNode->value = i;
    success = true;

    // return success
    return success;
    }

  myIntListClass::node* myIntListClass::add( const int i )
    {
    // vars
    node* newNode = new node;   // create new node
      newNode->next = NULL;     // it will be the new tail
      newNode->previous = NULL; // update later if necessary

    // set the value(s) of the node
    newNode->value = i;

    // relink everything
    if(head == NULL )
      {
      head = newNode;
      }
    else
      {
      tail->next = newNode;   // the ptr of the previous tail node
      newNode->previous = tail;
      }

    tail = newNode;         // newNode is now tail

    // increment size
    numNodes ++;

    // return ptr to new node (same as tail)
    return newNode;
    }

  myIntListClass::node* myIntListClass::addSet( const int* set, const int setSize )
    {
    // vars  
    int counter = 0;
    node* newNode = NULL;
    node* first = NULL; 
    
    // add each value to the list
    while( counter < setSize )
      {
      newNode = add( set[counter] );
        if(counter == 0)
          {
          first = newNode;
          }

      counter ++;
      numNodes ++;
      }    
 
    // return ptr to first element of set
    return first;
    }

  bool myIntListClass::deleteVal( const int i )
    {
    // vars
    node* currentNode = head;
    node* previousNode;
    bool success = false;

    // find the first occurrence of i
    while( (currentNode != NULL) && (currentNode->value != i) )
      {
      previousNode = currentNode;
      currentNode = currentNode->next;
      }
      
      // if i is not found, return failure
      if( currentNode == NULL )
        {
        return success;
        }
    
    // delete, relink, reset head/tail if necessary
    if( currentNode == head )
      {
      head = currentNode->next;
      currentNode->next->previous = NULL;
      delete currentNode;
      }
    else if( currentNode == tail )
      {
      tail = previousNode;
      delete currentNode;
      }    
    else
      {
      previousNode->next = currentNode->next;
      currentNode->next->previous = previousNode;
      delete currentNode;
      }

    // delete (likely) occurred
    success = true;

    // decrement size
    numNodes --;

    // no return success
    return success; 
    }

  void myIntListClass::sortS2L()
    {
    // vars
    int swapCount = 9; // != 0 so loop can be entered
    int compCount = 0;
    int passCount = 0;
    int temp;
    node* currentNode = head;

    // iterate through list until no swaps are made
    while( swapCount != 0 )
      {
      // reset swapCount
      swapCount = 0;

      while( (currentNode->next != NULL) && (compCount < (numNodes - passCount)) )
        {
        // make swap if necessary
        if( currentNode->value > currentNode->next->value )
          {
          temp = currentNode->value;
          currentNode->value = currentNode->next->value;
          currentNode->next->value = temp;

          swapCount ++;
          }

        // set up next comparison
        currentNode = currentNode->next;
        }

      // a pass through the list is complete, get set for next pass
      passCount ++;
      currentNode = head;
      }

    // no return - void
    }

  void myIntListClass::sortL2S()
    {
    // vars
    int swapCount = 9; // != 0 so loop can be entered
    int compCount = 0;
    int passCount = 0;
    int temp;
    node* currentNode = head;

    // iterate through list until no swaps are made
    while( swapCount != 0 )
      {
      // reset swapCount
      swapCount = 0;

      while( (currentNode->next != NULL) && (compCount < (numNodes - passCount)) )
        {
        // make swap if necessary
        if( currentNode->value < currentNode->next->value )
          {
          temp = currentNode->value;
          currentNode->value = currentNode->next->value;
          currentNode->next->value = temp;

          swapCount ++;
          }

        // set up next comparison
        currentNode = currentNode->next;
        }

      // a pass through the list is complete, get set for next pass
      passCount ++;
      currentNode = head;
      }

    // no return - void
    }

// Overloaded Operators ////////////////////////////////////////////////////////

