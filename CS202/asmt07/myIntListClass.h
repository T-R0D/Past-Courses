#ifndef ___MYINTLISTCLASS_H___
#define ___MYINTLISTCLASS_H___

////////////////////////////////////////////////////////////////////////////////
// 
//  Title:      myIntListClass.h
//  Created By: Terence Henriod
//  Course:     CS202
//
//  Summary:    The header file of the implementation of a doubly linked integer
//              list.
// 
//  Last Modified: 4/12/2013 14:45
//
////////////////////////////////////////////////////////////////////////////////

//============================================================================//
//= Header Files =============================================================//
//============================================================================//

// headers/namespaces
#include <iostream>
#include <fstream>
#include <conio.h>
#include <stdlib.h>
#include <assert.h>
#include "Ttools.h"
using namespace std;

//============================================================================//
//= Class Definition =========================================================//
//============================================================================//

class myIntListClass
   {
private:
   
  // Data Members //////////////////////////////////////////////////////////////
  struct node
    {
    int value;      // value contained in the node
    node* next;     // ptr to next node, if any
    node* previous; // ptr to the previous node, if any
    };

  node * head;       // ptr to the head of the list
  node * tail;       // ptr to the tail of the list

  int numNodes;     // current count of nodes in list   

public:

  // Constructor(s) ////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: myIntListClass   [0]
// Summary:       The default constructor. Creates a new instance of a doubly
//                linked list with no nodes.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  myIntListClass();  

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: myIntListClass   [1]
// Summary:       Constructs a doubly linked list with one node holding the 
//                value of the given node.
//
// Parameters:    int i   The value to be stored in the initial node.
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  myIntListClass( int i );

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: myIntListClass   [2]
// Summary:       Constructs a doubly linked list, initialized with the values
//                of the provided set.
//
// Parameters:    int* set      A pointer to the array of numbers to be used.
//                int setSize   The size of the set or number of values from
//                              given set to be initialized in the list.
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  myIntListClass( int* set, int setSize );

  // Destructor ////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: ~myIntListClass
// Summary:       Destroys an instance of this doubly linked list class,
//                thoroughly freeing any memory used.
//
// Parameters:    none
// Returns:       none
//
////////////////////////////////////////////////////////////////////////////////
  ~myIntListClass();

private:
  // Internal/Maintenance //////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: countNodes
// Summary:       Can be used to refresh the count of the nodes in the list
//                at any time.
//                *Currently Not Useful*
//
// Parameters:    none
// Returns:       void
//
////////////////////////////////////////////////////////////////////////////////
  void myIntListClass::countNodes();


public:
  // Accessors /////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: get_head
// Summary:       Returns a pointer to the head of the list, possibly useful
//                for linking lists to other lists.
//
// Parameters:    none
// Returns:       node*   A ptr to the head of the linked list's head
//
////////////////////////////////////////////////////////////////////////////////
  node* myIntListClass::get_head() const;

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: get_tail
// Summary:       Returns a pointer to the tail of the list.
//
// Parameters:    none
// Returns:       node*   A ptr to the head of the linked list's tail.
//
////////////////////////////////////////////////////////////////////////////////
  node* myIntListClass::get_tail() const;

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: find
// Summary:       Finds the memory location of the first node in the linked list
//                containing the specified value. If the search fails, the
//                returned pointer will be null.
//
// Parameters:    int i   The sought value.
// Returns:       node*   The memory location of the sought value. Null if not
//                        found. 
//
////////////////////////////////////////////////////////////////////////////////
  node* myIntListClass::find( int i) const;

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: traverse
// Summary:       Displays all of the values in a linked list vertically.
//
// Parameters:    none
// Returns:       void
//
////////////////////////////////////////////////////////////////////////////////
  void myIntListClass::traverse() const;

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: traverse_reverse
// Summary:       Displays all the values of a linked list in reverse order.
//
// Parameters:    none
// Returns:       void
//
////////////////////////////////////////////////////////////////////////////////
  void myIntListClass::traverse_reverse() const;

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: size
// Summary:       Returns the size of the list, as a count of the nodes in the 
//                list.
//
// Parameters:    none
// Returns:       int   The current count of the nodes in the linked list. 
//
////////////////////////////////////////////////////////////////////////////////
  int myIntListClass::size() const;

  // Mutators //////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: setValue
// Summary:       Sets the value of the nth (ndx) node to the given value. 
//                Returns whether or not the index position was found so the 
//                value set could be performed.
//
// Parameters:    int i     The value to be stored.
//                int ndx   The nth postition of the list.
// Returns:       bool   The success status of the operation.
//
////////////////////////////////////////////////////////////////////////////////
  bool myIntListClass::setValue( const int i, const int ndx );

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: add
// Summary:       Adds a node with the given value to the linked list.
//
// Parameters:    int i   The value of the node to be added.
// Returns:       node*   A ptr to the recently added (now tail) node.
//
////////////////////////////////////////////////////////////////////////////////
  node* myIntListClass::add( const int i );

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: addSet
// Summary:       Adds a set of nodes with the values specified by the given 
//                array to the linked list.
//
// Parameters:    int* set      A pointer to the array of numbers to be used.
//                int setSize   The size of the set or number of values from
//                              given set to be initialized in the list.
// Returns:       node*   A ptr to the first node added by the operation.
//
////////////////////////////////////////////////////////////////////////////////
  node* myIntListClass::addSet( const int* set, const int setSize );

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: deleteVal
// Summary:       Searches the linked list for the first occurence of the given 
//                value, then deletes the node, and performs any necessary
//                re-linking. Returns the success state of the operation
//                (false if i is not found).
//
// Parameters:    int i   The value to be deleted
// Returns:       bool   The success of the operation.
//
////////////////////////////////////////////////////////////////////////////////
  bool myIntListClass::deleteVal( const int i );

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: sortS2L
// Summary:       Implements an optimized bubble sort on the linked list by 
//                switching values held by nodes as appropriate. Sorts the
//                values [S]mallest -> [L]argest. 
//                ***IMPLEMENT BETTER SORTING ALGORITHM***
//
// Parameters:    none
// Returns:       void
//
////////////////////////////////////////////////////////////////////////////////
  void myIntListClass::sortS2L();

////////////////////////////////////////////////////////////////////////////////
//
// Function Name: sortL2S
// Summary:       Implements an optimized bubble sort on the linked list by 
//                switching values held by nodes as appropriate. Sorts the
//                values [L]argest -> [S]mallest. 
//                ***IMPLEMENT BETTER SORTING ALGORITHM***
//
// Parameters:    none
// Returns:       void
//
////////////////////////////////////////////////////////////////////////////////
  void myIntListClass::sortL2S();

  // Overloaded Operators //////////////////////////////////////////////////////

   };

#endif
