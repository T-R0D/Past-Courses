/**
    @file CS677_HW3_code.cpp

    @author Terence Henriod

    CS677 HW3: Max-finding and Sorting

    @brief This program explores max finding and sorting algorithms of
           the divide and conquer variety.

    @version Original Code 1.00 (3/2/2014) - T. Henriod
*/

/*==============================================================================
=======     HEADER FILES     ===================================================
==============================================================================*/
#include <iostream>
#include <cstdlib>
#include <cstring>
using namespace std;

/*==============================================================================
=======     GLOBAL CONSTANTS     ===============================================
==============================================================================*/
#define kPracticeInputSize 64
#define kSentinel '~'  // only works for char

/*==============================================================================
=======     FUNCTION PROTOTYPES     ============================================
==============================================================================*/

/**
Max

Finds the maximum element of an array using a divide and conquer
strategy.

@param A           The array under consideration
@param left_idx    The index of the left-most element considered
                   for a given recursive call
@param right_idx   The index of the right-most element considered
                   for a given recursive call
*/
template <typename ItemType>
ItemType Max( ItemType* A, unsigned int left_idx, unsigned int right_idx );


/**
bottomUpMergeSort

Sorts an array by merging sub-arrays in proper order.

@param A    The array under consideration
@param n    The size of the array given
*/
template <typename ItemType>
void bottomUpMergeSort( ItemType* A, int n );


/**
Merge

Merges two sub arrays in sorted order.
LARGELY ADAPTED FROM LECTURE SLIDES.

@param A    The array under consideration
@param p    The right_most index of the first sub-array
@param q    The left_most index of the first sub-array
@param r    The left_most index of the second sub-array
*/
template <typename ItemType>
void Merge( ItemType* A, int p, int q, int r );


/**
threePartQuicksort

Performs a 3 partition quicksort with partitions for elements less than,
equal to, and greater than the pivot element

@param A                 The array under consideration
@param partition_start   The index of the first element of the considered
                         partitition
@param partition_end     The index of the element just past the end of
                         the considered partition

*/
template <typename ItemType>
void threePartQuicksort( ItemType* A, int partition_start, int partition_end );


/**
arrayPrint

Neatly displays the contents of an array and various indices of interest.

@param many parameters...
*/
template <typename ItemType>
void arrayPrint( ItemType* A, int partition_start, int left_equal_ndx,
                 int left_ndx, int right_ndx, int right_equal_ndx,
                 int pivot_ndx );


/*==============================================================================
=======     MAIN FUNCTION     ==================================================
==============================================================================*/

/**
main

Simply calls the assigned functions with the specified test input given
to give sample output for the homework problems.
*/
int main()
{
  char A[kPracticeInputSize] = "TINYEXAMPLE";
  int problem_n = strlen( A );
  int left_idx = 0;
  int right_idx = problem_n - 1;

  // execute the Problem 1 code
  cout << "Testing Problem 1 code by outputting" << endl
       << "<Return_Value, Max(left_idx, right_idx)> tuples" << endl
       << "=======================================================" << endl;
  Max( A, left_idx, right_idx );
  cout << endl << endl;

  // execute the Problem 2 code
  strcpy( A, "ASORTINGEXAMPLE" );
  problem_n = strlen( A );
  cout << "Testing Problem 2 code by outputting" << endl
       << "the sub-arrays to be processed" << endl
       << "=======================================================" << endl;
  bottomUpMergeSort( A, problem_n );
  cout << endl;

  // execute the Problem 4 code
  strcpy( A, "ABRACACABRABCDC" );
  problem_n = strlen( A );
  cout << "Testing Problem 4 code by outputting" << endl
       << "the array with various indices shown" << endl
       << "=======================================================" << endl;
  threePartQuicksort( A, 0, problem_n );

  // end the program
  return 0;
}


/*==============================================================================
=======     FUNCTION IMPLEMENTATIONS     =======================================
==============================================================================*/

template <typename ItemType>
ItemType Max( ItemType* A, unsigned int left_idx, unsigned int right_idx )
{
  // variables
  ItemType Return_Value;
  ItemType left_temp;
  ItemType right_temp;
  int current_array_size = ( right_idx - left_idx + 1 );
  int left_upper_bound = 0;
  int right_lower_bound = 0;

  // base case: the current sub_array size is less than or equal to 2
  if( current_array_size == 1 )
  {
    // the only item is the greatest item
    Return_Value = A[left_idx];
  }
  // case: the current sub_array size is greater than 2
  else
  {
    // find the boundary indices of the two sub-arrays
    left_upper_bound = ( left_idx + right_idx ) / 2;
    right_lower_bound = ( ( left_idx + right_idx ) / 2 ) + 1;

    // find the max of each sub_array
    left_temp = Max( A, left_idx, left_upper_bound );
    right_temp = Max( A, right_lower_bound, right_idx );

    // find the greater of the two values
    Return_Value = ( left_temp < right_temp ? right_temp : left_temp );
  }

  // state the effect of this action for homework proposes
  cout << "<" << Return_Value << ", Max(" << left_idx << ", " << right_idx
       << ")>" << endl;

  // return the maximum value found for this array/sub-array
  return Return_Value;
}


template <typename ItemType>
void bottomUpMergeSort( ItemType* A, int n )
{
  // variables
  int m = 1; // the sub-array size to merge into size 2m
  int right_start = 0;
  int right_end = 0;
  int left_end = 0;

  // scan array, performing merges until two sub-arrays have been merged
  // to the original size
  while( m < n )
  {
    // perform the merge-scanning for the current sub-array size
    for( right_start = 0; right_start < n; right_start += ( 2 * m ) )
    {
      // find the end of the right sub-array
      right_end = right_start + m - 1;

      // find the end of the sub_arrays combined
      left_end = right_end + m;
      if( left_end >= n )
      {
        // prevent the end marker for the sub-arrays from overreaching the array
        left_end = n - 1;
      }

      // merge each pair of sub-arrays
      Merge( A, right_start, right_end, left_end );
    }

    // move up to the next sub-array size
    m *= 2;
  }

  // no return - void
}


template <typename ItemType>
void Merge( ItemType* A, int p, int q, int r )
{
  // variables
  int main_ndx = p;
  int left_ndx = 0;
  int right_ndx = 0;
  int left_size = q - p + 1;
  int right_size  = r - q;
  ItemType temp;
  ItemType left_array[left_size + 1];
    left_array[left_size] = kSentinel;
  ItemType right_array[right_size + 1];
    right_array[right_size] = kSentinel;

  // print the array being processed for hw purposes
  cout << "p: " << p << ", q: " << q << ", r: " << r << endl;
  cout << "Before: ";
  for( main_ndx = p; main_ndx <= r; main_ndx++ )
  {
    cout << A[main_ndx] << ' ';
  }
  cout << endl;

  // load up the sub-arrays
  for( main_ndx = p, left_ndx = 0; main_ndx <= q; left_ndx++, main_ndx++ ) {
    left_array[left_ndx] = A[main_ndx];
  }
  for( right_ndx = 0; main_ndx <= r; right_ndx++, main_ndx++ )
  {
    // copy the element over
    right_array[right_ndx] = A[main_ndx];
  }


  // perform the merging
  for( main_ndx = p, right_ndx = 0, left_ndx = 0; main_ndx <= r; main_ndx++ )
  {
    // case: the element of the left sub-array is less or equal
    if( left_array[left_ndx] <= right_array[right_ndx] )
    {
      // store the element in the original array
      A[main_ndx] = left_array[left_ndx];

      // move on to the next element of the right sub-array
      left_ndx++;
    }
    // case: the item in the right array is the lesser of the two elements
    else
    {
      // store the element in the original array
      A[main_ndx] = right_array[right_ndx];

      // move on to the next element of the right sub-array
      right_ndx++;
    }
  }

  // print the merge result for hw purposes
  cout << "After:  ";
  for( main_ndx = p; main_ndx <= r; main_ndx++ )
  {
    cout << A[main_ndx] << ' ';
  }
  cout << endl << endl;

  // no return - void
}


template <typename ItemType>
void threePartQuicksort( ItemType* A, int partition_start, int partition_end )
{
  // variables
  int pivot_ndx = partition_end - 1;
  int left_ndx = partition_start;
  int right_ndx = pivot_ndx - 1;
  int left_equal_ndx = left_ndx;
  int right_equal_ndx = right_ndx;
  ItemType pivot = A[pivot_ndx];

  // output the array for hw purposes
  cout << "At the start of a new call:" << endl;
  arrayPrint( A, partition_start, left_equal_ndx, left_ndx,
              right_ndx, right_equal_ndx, pivot_ndx );

  // case: the partition size is larger than 1
  if( partition_start < pivot_ndx )
  {
    // perform scanning and swapping to create the left and right partitions
    while( left_ndx < right_ndx )
    {
      // swap the two items to correctly partition them
      swap( A[left_ndx], A[right_ndx] );

      // case: the item at the left index is equal to the pivot
      //       and there is a not-equal element to swap it with
      if( ( A[left_ndx] == pivot ) && ( left_equal_ndx < left_ndx ) )
      {
        // make the swap
        swap( A[left_ndx], A[left_equal_ndx] );

        // update the left-equal sub-partition and the left partition
        left_equal_ndx++;
        left_ndx++;
      }

      // case: the item at the right index is equal to the pivot
      //       and there is a-not equal element to swap it with
      if( ( A[right_ndx] == pivot ) && ( right_ndx < right_equal_ndx ) )
      {
        // make the swap
        swap( A[right_ndx], A[right_equal_ndx] );

        // update the right-equal sub-partition and the right partition
        right_equal_ndx--;
        right_ndx--;
      }

      // scan from the left for an element that is not less than the pivot
      while( ( A[left_ndx] < pivot ) && ( left_ndx < pivot_ndx ) )
      {
        // advance the left index
        left_ndx++;
      }

      // scan from the right for an element that is not greater than the pivot
      while( ( A[right_ndx] > pivot ) && ( right_ndx > partition_start ) )
      {
        // advance the left index
        right_ndx--;
      }

      // output the array for hw purposes
      cout << "After an iteration of scan/swapping:" << endl;
      arrayPrint( A, partition_start, left_equal_ndx, left_ndx,
                  right_ndx, right_equal_ndx, pivot_ndx );
    }

    // case: the pivot is greater than all other elements in the partition
    //       (no left partition can/should be made)
    if( left_ndx < pivot_ndx )
    {
      // move the pivot to create 3 partitions: right, middle, left
      swap( A[left_ndx], A[pivot_ndx] );
      right_ndx = left_ndx - 1;
      left_ndx++;

      // bump everything from the left-equal sub-partition to the middle
      while( left_equal_ndx > partition_start )
      {
        left_equal_ndx--;
        swap( A[left_equal_ndx], A[right_ndx] );
        right_ndx--;
      }

      // bump everything from the right-equal sub-partition to the middle
      while( right_equal_ndx < pivot_ndx )
      {
        right_equal_ndx++;
        swap( A[right_equal_ndx], A[left_ndx] );
        left_ndx++;
      }
    }

    // display the effects of this function call for homework purposes
    cout << "After pivot swapping and creating the middle partition:" << endl;
    arrayPrint( A, partition_start, left_equal_ndx, left_ndx,
                right_ndx, right_equal_ndx, pivot_ndx );

    // sort the left partition
    threePartQuicksort( A, partition_start, right_ndx + 1 );

    // sort the right partition
    threePartQuicksort( A, left_ndx - 1, partition_end );
  }
  // base case: the partition was of size 1 or less
    // do nothing

  // display the effects of this function call for homework purposes
  cout << "At the end of a call:" << endl;
  arrayPrint( A, partition_start, left_equal_ndx, left_ndx,
              right_ndx, right_equal_ndx, pivot_ndx );

  // no return - void
}


template <typename ItemType>
void arrayPrint( ItemType* A, int partition_start, int left_equal_ndx,
                 int left_ndx, int right_ndx, int right_equal_ndx,
                 int pivot_ndx )
{
  // variables
  int ndx = 0;

  // print the contents of the array
  for( ndx = 0; ndx < partition_start; ndx++ )
  {
    cout << ' ';
  }
  for( ; ndx <= pivot_ndx; ndx++ )
  {
    cout << A[ndx];
  }
  cout << endl;

  // print spaces and carats as to indicate points of interest
  for( ndx = 0; ndx < partition_start; ndx++ )
  {
    cout << ' ';
  }
  for( ; ndx <= pivot_ndx; ndx++ )
  {
    // case: under a point of interest
    if( ndx == partition_start ||
        ndx == left_equal_ndx  ||
        ndx == left_ndx        ||
        ndx == right_ndx       ||
        ndx == right_equal_ndx ||
        ndx == pivot_ndx )
    {
      // make an arrow pointing to it
      cout << '^';
    }
    // case: uninteresting
    else
    {
      // put a space
      cout << ' ';
    }
  }
  cout << endl;

  // output informative markers where necessary
  for( ndx = 0; ndx < partition_start; ndx++ )
  {
    cout << ' ';
  }
  for( ; ndx <= pivot_ndx; ndx++ )
  {
    // output an appropriate marker
    if( ndx == pivot_ndx )
    {
      cout << 'r';
    }
    else if( ndx == partition_start )
    {
      cout << 'l';
    }
    else if( ndx == left_ndx )
    {
      cout << 'i';
    }
    else if( ndx == right_ndx )
    {
      cout << 'j';
    }
    else if( ndx == left_ndx && ndx == right_ndx )
    {
      cout << '#';
    }
    else if( ndx == left_equal_ndx )
    {
      cout << 'p';
    }
    else if( ndx == right_equal_ndx )
    {
      cout << 'q';
    }
    else
    {
      cout << ' ';
    }
  }

  // move down a line
  cout << endl << endl;

  // no return - void
}

