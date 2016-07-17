/**
    @file Heap.h

    @author Terence Henriod

    Lab 10: Heap

    @brief Class declarations for the Heap ADT.


    @version Original Code 1.00 (11/8/2013) - T. Henriod
*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   PRECOMPILER DIRECTIVES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#ifndef HEAP_H
#define HEAP_H


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#include <stdexcept>
#include <iostream>
using namespace std;


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   GLOBAL CONSTANTS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

// none


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
================================================================================
                   CLASS DEFINITION(S)
================================================================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
@class Less

A class with an overloaded operator() (function operator) for use as the
comparator of a Heap object. Subtle manipulations make this class act more as a
function, rather than a class.
*/
template < typename KeyType = int >
class Less
{
 public:
  /*---   Overloded Function Operator   ---*/
  bool operator()(const KeyType &a, const KeyType &b) const { return a < b; }
};


/**
@class Heap

A Heap ADT. The Heap is a data structure for storing data as though it were a
tree whose requirements are that no parent is less than a child and that the
tree must be completely full except for the bottom level. The data is stored in
an array, granting fast access and easy reordering of data. Because the data is
stored in an array, the bottom level of data fills up from left to right.
*/
template < typename DataType, typename KeyType = int,
           typename Comparator = Less< KeyType > >
class Heap
{
 public:
  /*---   Constructor(s) / Destructor   ---*/
  Heap( int maxNumber = DEFAULT_MAX_HEAP_SIZE );
  Heap( const Heap& other );
  Heap& operator=( const Heap& other );
  ~Heap();


  /*---   Mutators   ---*/
  void insert( const DataType &newDataItem ) throw ( logic_error );
  DataType remove() throw ( logic_error );
  void clear();


  /*---   Accessors   ---*/
  bool isEmpty() const;
  bool isFull() const;
  void showStructure() const;
  void writeLevels() const;


  /*---   Default Values   ---*/
  static const int DEFAULT_MAX_HEAP_SIZE = 10;


 private:
  /*---   Helpers   ---*/
  // Accessor Helpers
  void showSubtree ( int index, int level ) const;


  /*---   Data Members   ---*/
  int maxSize;
  int size;
  DataType *dataItems;
  Comparator comparator;
};


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   TERMINATING PRECOMPILER DIRECTIVES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#endif	//#ifndef HEAP_H

