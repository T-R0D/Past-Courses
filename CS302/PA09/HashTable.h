/**
    @file HashTable.h

    @author Terence Henriod

    Hash Table

    @brief Class declarations for the Hash Table ADT. Utilizes the Binary Search
           Tree ADT to chain data items to mitigate collisions.


    @version Original Code 1.00 (11/2/2013) - T. Henriod
*/


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   PRECOMPILER DIRECTIVES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#ifndef HASHTABLE_H
#define HASHTABLE_H


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#include <stdexcept>
#include <iostream>

using namespace std;

#include "BSTree.cpp"


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
@class HashTable

The HashTable ADT aims to reduce search time by placing data items in an array
based on som value returned by a hashing function. It is possible that the
hashing function could return similar values for different items, resulting in
collisions. These collisions are mitigated through the use of chaining: each
element of the has table is another data structure, in this case a binary search
tree to sort items that have the same hash value. The HashTable ADT only works
with DataTypes that support hash() and getKey() member functions.
*/
template <typename DataType, typename KeyType>
class HashTable {
 public:
  /*---   Constructor(s) / Destructor   ---*/
  HashTable(int initTableSize);
  HashTable(const HashTable& other);
  HashTable& operator=(const HashTable<DataType, KeyType>& other);
  ~HashTable();

  /*---   Mutators   ---*/
  void insert(const DataType& newDataItem);
  bool remove(const KeyType& deleteKey);
  bool retrieve(const KeyType& searchKey, DataType& returnItem) const;
  void clear();

  /*---   Accessors   ---*/
  bool isEmpty() const;
  void showStructure() const;
  double standardDeviation() const;

 private:
  /*---   Helpers   ---*/
  // Clone Helper
  void copyTable(const HashTable& source);

  /*---   Data Members   ---*/
  int tableSize;
  BSTree<DataType, KeyType>* dataTable;
};

#endif	// ifndef HASHTABLE_H
