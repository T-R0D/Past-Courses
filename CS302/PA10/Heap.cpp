/**
    @file Heap.cpp

    @author Terence Henriod

    Lab 10: Heap

    @brief Class implementations declarations for the Heap ADT

    @version Original Code 1.00 (11/8/2013) - T. Henriod
*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

// Class Declaration
#include "Heap.h"

// Other Dependencies
#include <stdexcept>
#include <iostream>
using namespace std;


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
================================================================================
                   CLASS FUNCTION IMPLEMENTATIONS
================================================================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   CONSTRUCTOR(S) / DESTRUCTOR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
Heap

The default constructor for the heap. Instantiates an empty heap.

@param maxNumber   The maximum capacity given to the heap. It is recommended
                   that a size that is 2^n - 1 is chosen. This parameter
                   defaults to DEFAULT_MAX_HEAP_SIZE (defined in Heap.h).

@pre
-# A valid identifier for the Heap is given
-# The given template parameter DataType should support comparisons and some
   for key identification.
-# The given number for the parameter int maxNumber must be greater than zero.

@post
-# An empty Heap of the given types and size (if it is specified) is
   instantiated.

@code
@endcode
*/
template <typename DataType, typename KeyType, typename Comparator>
Heap<DataType, KeyType, Comparator>::Heap( int maxNumber )
{
  // initialize data members
  maxSize = maxNumber;
  size = 0;
  dataItems = new DataType [ maxSize ];

  // no return - constructor
}


/**
Heap

The copy constructor for the heap. Instantiates a Heap that is a clone of the
given other parameter.

@param other   A Heap to be cloned into *this.

@pre
-# A valid identifier for the Heap is given
-# The given template parameter DataType should support comparisons and some
   for key identification.
-# The given number for the parameter int maxNumber must be greater than zero.

@post
-# A Heap that is a clone of the given other parameter is instantiated.

@code
@endcode
*/
template <typename DataType, typename KeyType, typename Comparator>
Heap<DataType, KeyType, Comparator>::Heap(
    const Heap<DataType, KeyType, Comparator>& other )
{
  // initialize data members
  maxSize = 0;
  size = 0;
  dataItems = NULL;

  // clone other into *this
  *this = other;

  // no return - constructor
}


/**
operator=

The overloaded assignment operator for the Heap ADT. Clones the given Heap other
parameter into *this.

@param other   A Heap whose data will be cloned into this one.

@return *this   *this is returned by reference for multi-line assignments.

@pre
-# Both *this and other are valid Heaps

@post
-# The contents of other will be cloned into *this. The original data will be
   more.

@detail @bAlgorithm
-# If *this is not being assigned to *this, the current data is abandoned.
-# If other has a different maxSize than *this, the dataItems array is re-sized
-# The dataItems array is made equivalent to the one in other

@code
@endcode
*/
template <typename DataType, typename KeyType, typename Comparator>
Heap<DataType, KeyType, Comparator>&
Heap<DataType, KeyType, Comparator>::operator=( const Heap& other )
{
  // case: other is not *this
  if( this != &other )
  {
    // clear
    clear();

    // case: the array sizes are different
    if( maxSize != other.maxSize )
    {
      // copy the array size
      maxSize = other.maxSize;

      // re-allocate the array
      delete [] dataItems;
      dataItems = new DataType [ maxSize ];   
    }

    // make copies of all the elements
    for( size = 0; size < maxSize; size++ )
    {
      // copy the element
      dataItems[ size ] = other.dataItems[ size ];
    }
  }
  // otherwise, do not perform assignment

  // return *this
  return *this;
}


/**
~Heap

The destructor for the heap ADT. Ensures that all dynamically allocated memory
is returned.

@pre
-# There is a Heap to destruct.

@post
-# The dynamic memory allocated for the array will be returned.
-# The Heap object (*this) will be destroyed.

@detail @bAlgorithm
-# The dynamically allocated array is deleted.
-# The rest of the heap is appropriately destroyed in the usual manner.

@code
@endcode
*/
template <typename DataType, typename KeyType, typename Comparator>
Heap<DataType, KeyType, Comparator>::~Heap()
{
  // return dynamic memory
  delete [] dataItems;

  // no return - destructor
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   MUTATORS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
insert

Inserts the given value into the appropriate place in the array if there is
room. Throws an expression of type logic_error if the Heap is full.

@param newDataItem   The new item to be inserted into the Heap.

@pre
-# A valid Heap instantiation exists.
-# The given parameter newDataItem is of appropriate type.
-# The Heap must not be full for successful insertion.

@post
-# If there was room in the Heap, the newDataItem will be placed in the Heap
   such that it will be less than any "parent" that it may have.
-# If the heap is full and insertion is attempted, an exception of type
   logic_error is thrown, indicating that the heap is full.

@detail @bAlgorithm
-# If the Heap is full, an exception is thrown, otherwise the DataType
   newDataItem is placed in the first available location.
-# The newDataItem is then percolated up to the appropriate level in the heap by
   comparing the newly inserted item with its current "parent" and swapping if
   necessary until the new item cannot rise any further.

@exception logic_error   This exception is thrown if an attempt is made to
                         insert into a full heap.

@code
@endcode
*/
template <typename DataType, typename KeyType, typename Comparator>
void Heap<DataType, KeyType, Comparator>::insert( const DataType &newDataItem )
                                                  throw ( logic_error )
{
  // variables
  int newItemCursor = size;
  int parentCursor = size;
  bool keepPercolating = true;
  DataType temp;

  // case: the heap is not full
  if( size < maxSize )
  {
    // place the new item
    dataItems[ newItemCursor ] = newDataItem;

    // incremente the size of the Heap
    size++;

    // percolate until the new item has risen enough
    while( keepPercolating )
    {
      // reset the percolation flag
      keepPercolating = false;

      // case: the item is not in the "root" location
      if( newItemCursor > 0 )
      {
        // calculate the parent location
        parentCursor = ( newItemCursor - 1 ) / 2;

        // case: the new item needs to rise
        if( comparator( dataItems[ newItemCursor ].getPriority(),
                        dataItems[ parentCursor ].getPriority() ) )
        {
          // swap the new item and the "parent"
          temp = dataItems[ newItemCursor ];
          dataItems[ newItemCursor ] = dataItems[ parentCursor ];
          dataItems[ parentCursor ] = temp;

          // update the new item's cursor
          newItemCursor = parentCursor;

          // indicate that percolating may still be necessary
          keepPercolating = true;
        }
      }
    }
  }
  // case: the heap is full
  else
  {
    // throw an exception to indicate that the heap is full
    throw logic_error( "Error - can't insert into a full heap.\n" );
  }

  // no return - void
}


/**
remove

Removes the item at the top ("root") of the Heap. The heap is then reorderd
appropriately to maintain the properties of a heap. The removed item is
returned. An exception is thrown if removal is attempted on an empty heap.

@return removedItem   An item of type DataType. This item was the item at the
                      top of the heap.

@pre
-# A valid Heap has been instantiated.
-# The heap has an item to be removed, otherwise, and excpetion will be thrown.

@post
-# If an item at the top of the Heap, it is removed from the heap and returned.
-# If the Heap is empty, an exception of type logic_error is thrown with a
   message indicating that removal cannot be performed.

@detail @bAlgorithm
-# If the Heap is empty, an exception of type logic_error is thrown. If the Heap
   is not empty, the top item is stored for returning.
-# The bottom-right-most item is placed at the top of the Heap and the size is
   reduced.
-# The newly placed top item is worked downward until it sits in the appropriate
   place. This is accomplished by swapping the item with any child that
   compares greater than the new top item.

@exception logic_error   This exception is thrown if removal on an empty heap
                         is attempted.

@code
@endcode
*/
template <typename DataType, typename KeyType, typename Comparator>
DataType Heap<DataType, KeyType, Comparator>::remove() throw ( logic_error )
{
  // variables
  DataType removedItem;
  DataType temp;
  bool keepSinking = true;
  bool doSwap = false;
  int sinkerCursor = 0;
  int childCursor = 0;

  // case: the heap is not empty
  if( size != 0 )
  {
    // store the top item
    removedItem = dataItems[ 0 ];

    // copy the bottom-right most item to the top
    dataItems[ 0 ] = dataItems[ size - 1 ];

    // decrement size since an item was removed
    size--;

    // sink the new top down as many times as necessary
    while( keepSinking )
    {
      // reset the sinking and swapping flags
      keepSinking = false;
      doSwap = false;

      // compute the position of the first child
      childCursor = ( ( 2 * sinkerCursor ) + 1 );

      // case: there is room to sink
      if( childCursor < size )
      {
        // case: the sinking item is less than the first child
        if( comparator( dataItems[ childCursor ].getPriority(),
                        dataItems[ sinkerCursor ].getPriority() ) )
        {
          // a swap should be made
          doSwap = true;
        }

        // case: there is a second child to compare to
        if( ( childCursor + 1 ) < size )
        {
          // case: the second child is larger than the sinking item
          if( comparator( dataItems[ childCursor + 1 ].getPriority(),
                          dataItems[ sinkerCursor ].getPriority() ) )
          {
            // a swap should be made
            doSwap = true;

            // case: the second child is greater than the first
            if( comparator( dataItems[ childCursor + 1 ].getPriority(),
                            dataItems[ childCursor ].getPriority() ) )
            {
              // the second child should be swapped with the sinker
              childCursor++;
            }
          }
        }

        if( doSwap )
        {
          // swap the sinker and the child
          temp = dataItems[ childCursor ];
          dataItems[ childCursor ] = dataItems[ sinkerCursor ];
          dataItems[ sinkerCursor ] = temp;

          // update the sinker position
          sinkerCursor = childCursor;

          // there may be a need to sink further
          keepSinking = true;
        }
      }
    }
  }
  // case: the heap is empty
  else
  {
    // throw an exception to indicate the heap is empty
    throw logic_error( "Error - can't remove from an empty heap.\n" );
  }

  // return the removed item
  return removedItem;
}


/**
clear

Empties the heap.

@pre
-# A valid Heap instantiation exists.

@post
-# Data items of the heap are "discarded" by setting the size member to zero.

@code
@endcode
*/
template <typename DataType, typename KeyType, typename Comparator>
void Heap<DataType, KeyType, Comparator>::clear()
{
  // "empty" the Heap
  size = 0;

  // no return - void
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   ACCESSORS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
isEmpty

Indicates if the Heap is full by returning true if it is empty and false
otherwise.

@return empty   A boolean containing the truth of the emptiness of the Heap.

@pre
-# A valid Heap instantiation exists.

@post
-# The Heap will remain unchanged.
-# If the Heap is empty, true is returned, and false otherwise.

@code
@endcode
*/
template <typename DataType, typename KeyType, typename Comparator>
bool Heap<DataType, KeyType, Comparator>::isEmpty() const
{
  // return the truth of the heap being empty
  return ( size == 0 );
}


/**

isFull

Indicates if the Heap is full by returning true if it is full, and false
otherwise.

@return full   A boolean containing the truth of the fullness of the heap

@pre
-# A valid Heap instantiation exists.

@post
-# The Heap will remain unchanged.
-# If the Heap is full, true is returned, and false otherwise.

@exception

@code
@endcode

*/
template <typename DataType, typename KeyType, typename Comparator>
bool Heap<DataType, KeyType, Comparator>::isFull() const
{
  // return the truth of the Heap being full
  return ( size == maxSize );
}


/**

showStructure

Outputs the priorities of the data items in a heap in both array and tree form.
If the heap is empty, outputs "Empty heap". This operation is intended for
testing/debugging purposes only.

@pre
-# A valid Heap instantiation exists.
-# The type DataType supports operator<<

@post
-# The Heap will remain unchanged.
-# The Heap will be displayed on the screen, first in array form, then in tree
   form.

@detail @bAlgorithm
-# The array elements are iteratively displayed.
-# The showSubtree helper is then called to display the Heap as a tree

@code
@endcode

*/
template <typename DataType, typename KeyType, typename Comparator>
void Heap<DataType, KeyType, Comparator>::showStructure () const
{
    int j;   // Loop counter

    cout << endl;
    if ( size == 0 )
       cout << "Empty heap" << endl;
    else
    {
       cout << "size = " << size << endl;       // Output array form
       for ( j = 0 ; j < maxSize ; j++ )
           cout << j << "\t";
       cout << endl;
       for ( j = 0 ; j < size ; j++ )
           cout << dataItems[j].getPriority() << "\t";
       cout << endl << endl;
       showSubtree(0,0);                        // Output tree form
    }
}


/**

writeLevels

Writes the priorities (keys) of the contents of the Heap wo the screen, one
level at a time, beginning at the top.

@pre
-# A valid instantiation of the Heap exists.
-# Type DataType must support a getPriority() method.

@post
-# The Heap will remain unchanged.
-# The priorities (keys) of the Heap will be listed by level, from top to
   bottom. If the Heap is empty, it is reported.

@detail @bAlgorithm
-# 

@code
@endcode

*/
template <typename DataType, typename KeyType, typename Comparator>
void Heap<DataType, KeyType, Comparator>::writeLevels() const
{
  // variables
  int cursor = 0;
  int inLevelCursor = 0;
  int numOnLevel = 1;

  // case: the Heap is not empty
  if( size != 0 )
  {
    // write all keys until there are no more
    while( cursor < size )
    {
      // write the next key
      cout << dataItems[ cursor ].getPriority();

      // update the "within" level counter
      inLevelCursor++;

      // case: all keys of a given level have been written
      if( inLevelCursor >= numOnLevel )
      {
        // reset the "within" level counter
        inLevelCursor = 0;

        // update the number of keys on the next level
        numOnLevel *= 2;

        // move to the next line
        cout << endl;
      }
      // case: there are more keys in this level
      else
      {
        // write a space
        cout << ' ';
      }

      // advance the cursor
      cursor++;
    }
  }
  // case: the Heap is empty
  else
  {
    // report that the Heap is empty
    cout << "Empty heap.";
  }

  // no return - void
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            PRIVATE HELPER FUNCTIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*=====  Accessors  ==========================================================*/

/**
FunctionName

Helper function for the showStructure() function. Outputs the subtree (subheap)
whose root is stored in dataItems[index]. Argument level is the level of this
dataItems within the tree.

@pre
-# A valid Heap instantiation exists.
-# The type DataType supports operator<<
-# recursive calls to this function may have been previously made.

@post
-# The Heap will remain unchanged.
-# The Heap subtree will be displayed in a right-ward growing tree on the
   screen.

@detail @bAlgorithm
-# A reversed in-order traversal is used to display the items as a tree.
-# Children are found using the formula 2*parentIntex + (1, 2)

@code
@endcode
*/
template <typename DataType, typename KeyType, typename Comparator>
void Heap<DataType, KeyType, Comparator>::showSubtree( int index,
                                                       int level ) const
{
     int j;   // Loop counter

     if ( index < size )
     {
        showSubtree(2*index+2,level+1);        // Output right subtree
        for ( j = 0 ; j < level ; j++ )        // Tab over to level
            cout << "\t";
        cout << " " << dataItems[index].getPriority();//OutputdataItem'spriority
        if ( 2*index+2 < size )                // Output "connector"
           cout << "<";
        else if ( 2*index+1 < size )
           cout << "\\";
        cout << endl;
        showSubtree(2*index+1,level+1);        // Output left subtree
    }
}

