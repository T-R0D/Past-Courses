/**
    @file PriorityQueue.cpp

    @author Terence Henriod

    Project Name

    @brief Class implementations declarations for the PriorityQueue ADT
           (inherits from the array based Heap ADT). 

    @version Original Code 1.00 (11/8/2013) - T. Henriod
*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

// Class Declaration
#include "PriorityQueue.h"

// Other Dependencies
  // none


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
================================================================================
                   CLASS FUNCTION IMPLEMENTATIONS
================================================================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   CONSTRUCTOR(S) / DESTRUCTOR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
PriorityQueue

The default constructor for the PriorityQueue ADT. Calls the Heap constructor
with the given size parameter value.

@param maxNumber   The size value for the PriorityQueue. Defaults to the
                   constant value defMaxQueueSize specified in PriorityQueue.h.

@pre
-# A valid identifier is selected for the PriorityQueue.

@post
-# An empty PriorityQueue will be instantiated.

@code
@endcode
*/
template <typename DataType, typename KeyType, typename Comparator>
PriorityQueue<DataType, KeyType, Comparator>::PriorityQueue( int maxNumber ) 
: Heap<DataType, KeyType, Comparator>( maxNumber )
{
  // the Heap constructor is called with the given parameter

  // no return - constructor
}


/**
PriorityQueue

The copy constructor for the PriorityQueue ADT. Calls the Heap copy constructor
with the given Heap parameter value in order to create a clone of other into
*this.

@param other   A Heap to be cloned into *this.

@pre
-# A valid identifier is selected for the PriorityQueue.
-# Heap other is a valide Heap instantiation.

@post
-# An empty PriorityQueue will be instantiated.

@code
@endcode
*/
template <typename DataType, typename KeyType, typename Comparator>
PriorityQueue<DataType, KeyType, Comparator>::PriorityQueue(
    const Heap<DataType, KeyType, Comparator>& other )
: Heap<DataType, KeyType, Comparator>( other )
{
  // the Heap copy constructor is called with the given parameter

  // no return - constructor
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   MUTATORS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
enqueue

Adds the data item to the Priority Queue. The item's releative position will
depend on it's priority.

@param newDataItem   The new DataItem to be inserted into the PriorityQueue

@pre
-# DataType must support a valid getPriority() method.

@post
-# If there is room in the PriorityQueue, the new item will be stored in the
   appropriate position. If there is no room, an exception of type logic_error
   is thrown.

@detail @bAlgorithm
-# The Heap's insert method is called.

@exception logic_error   The Heap base class may throw an exception if insertion
                         is attempted when the data array is full.

@code
@endcode
*/
template <typename DataType, typename KeyType, typename Comparator>
void PriorityQueue<DataType, KeyType, Comparator>::enqueue(
    const DataType &newDataItem )
{
  // insert the item
  Heap<DataType, KeyType, Comparator>::insert( newDataItem );

  // no return - void
}


/**
dequeue

Dequeues the highest priority item. The item is returned along with its removal.

@return dequeuedItem   The item of type DataType that was removed from the Heap,
                       and therefore, the PriorityQueue.

@pre
-# The PriorityQueue must not be empty, otherwise an exception will be thrown.

@post
-# If the PriorityQueue is not empty, the highest priority item is removed from
   the queue and returned. If dequeueing is attempted on an empty PriorityQueue,
   an exception of type logic_error is thrown.

@detail @bAlgorithm
-# The remove method of the Heap base class is utilized.

@exception logic_error   This exception is thrown by the Heap base class if
                         removal is attempted when the PriorityQueue is empty.

@code
@endcode
*/
template <typename DataType, typename KeyType, typename Comparator>
DataType PriorityQueue<DataType, KeyType, Comparator>::dequeue()
{
  // call the remove function of the Heap, return the result
  return Heap<DataType, KeyType, Comparator>::remove();
}

