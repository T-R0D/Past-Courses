/**
    @file ossim.cpp

    @author Terence Henriod

    Lab 10: Heap Sort

    @brief Contains definitions for cunctions that together comprise a heap sort
           utility. The heapSort() shell function is provided by the lab manual
           package, while the moveDown helper function was written by
           T. Henriod. This heapSort is for arrays.

    @version Original Code 1.00 (11/8/2013) - T. Henriod
*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
moveDown

Converts a binary search tree (array implementation) subtree into a heap.
Assumes any lower subtrees are alrady heaps. Restores the binary tree that is
rooted at root to a heap by moving dataItems[root] downward until the tree
satisfies the heap property. Parameter size is the number of data items in the
array.

@param dataItems[]   The array to be heapified.
@param root          The index of the array indicating the root of a sub-tree.
@param size          The number of items in the array as a whole.

@pre
-# All parameters are valid.
-# Any subtrees are already heaps (assumed, not checked)

@post
-# The sub-array will be heapified.

@code
@endcode
*/
template < typename DataType >
void moveDown( DataType dataItems[], int root, int size )
{
  // variables
  bool keepSinking = true;
  bool doSwap = false;
  int cursor = root;
  int childCursor = root;
  DataType temp;

  // sink the root as much as necessary until it either can't sink or has
  // nowhere to go 
  while( keepSinking )
  {
    // reset the sinking and swapping flags
    keepSinking = false;
    doSwap = false;

    // compute the position of the first child
    childCursor = ( ( 2 * cursor ) + 1 );

    // case: there is somewhere to sink to
    if( childCursor < size )
    {
      // case: the child is greater than the sinking item
      if( dataItems[ cursor ].getPriority() <
          dataItems[ childCursor ].getPriority() )
      {
        // a swap should be made
        doSwap = true;
      }

      // case: there is another child to be compared with
      if( ( childCursor + 1 ) < size )
      {
        // case: the second child is greater than the sinking item
        if( dataItems[ cursor ].getPriority() <
            dataItems[ childCursor + 1 ].getPriority() )
        {
          // a swap should be made
          doSwap = true;

          // case: the second child is greater than the first
          if( dataItems[ childCursor ].getPriority() <
              dataItems[ childCursor + 1 ].getPriority() )
          {
            // choose the second child to be swapped
            childCursor++;
          }
        }
      }

      // case: a swap should be performed
      if( doSwap )
      {
        // perform the swap
        temp = dataItems[ cursor ];
        dataItems[ cursor ] = dataItems[ childCursor ];
        dataItems[ childCursor ] = temp;

        // update the sinking item's position
        cursor = childCursor;

        // more sinking may be necessary
        keepSinking = true;
      }
    }
    // otherwise, the sub-array is already heapified
  }

  // no return - void
}


/**
heapSort

Heap sort routine. Sorts the data items in the array in ascending order based on
priority.

@param dataItems[]   The array to be heapified.
@param size          The number of items in the array as a whole.

@pre
-# The array should contain elements arranged as in a binary search tree.

@post
-# The items in the array will be heapified.

@code
@endcode
*/
template < typename DataType >
void heapSort( DataType dataItems [], int size )
{
  DataType temp;   // Temporary storage
  int j;           // Loop counter

  // Build successively larger heaps within the array until the
  // entire array is a heap.

  for ( j = ( ( size - 1 ) / 2 ); j >= 0 ; j-- )
  {
    moveDown( dataItems, j, size);
  }

  // Swap the root data item from each successively smaller heap with
  // the last unsorted data item in the array. Restore the heap after
  // each exchange.

  for ( j = ( size - 1 ); j > 0 ; j-- )
  {
    temp = dataItems[j];
    dataItems[j] = dataItems[0];
    dataItems[0] = temp;
    moveDown( dataItems, 0, j);
  }
}

