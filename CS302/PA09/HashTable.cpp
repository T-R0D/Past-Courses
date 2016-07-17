/**
    @file HashTable.cpp

    @author Terence Henriod

    Hash Table

    @brief Class implementations declarations for the Hash Table ADT. Utilizes
           the Binary Search Tree to mitigate collisions.

    @version Original Code 1.00 (11/2/2013) - T. Henriod
*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

// Class Declaration
#include "HashTable.h"

// Other Dependencies
#include <iostream>
#include <cmath>
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
The default constructor for the hash table ADT. Constructs an empty hash table.

@pre
-# There is memory available for a hash table.
-# The new hash table is given an appropriate identifier.
-# the int initTableSize parameter is a valid table size.

@post
-# An empty hash table of the given size is constructed.

@code
@endcode
*/
template <typename DataType, typename KeyType>
HashTable<DataType, KeyType>::HashTable(int initTableSize)
{
  // initialize data member(s)
  tableSize = initTableSize;
  dataTable = new BSTree<DataType, KeyType> [tableSize];

  // no return - constructor
}


/**
The copy constructor for the hash table ADT. Initializes *this to be equivalent
to the given HashTable other parameter.

@param other   The given hash table to be cloned into ths one.

@pre
-# There is memory available for another hash table.
-# HashTable other is a valid hash table of same type(s).
-# The hash table to be constructed has a valid identifier.

@post
-# An equivalent clone of the given hash table parameter will be creaed in
   *this.

@detail @bAlgorithm
-# Calls the overloaded assignment operator to complete the task of cloning.


@code
@endcode
*/
template <typename DataType, typename KeyType>
HashTable<DataType, KeyType>::HashTable(const HashTable& other)
{
  // initialize data member(s)
  tableSize = 0;
  dataTable = NULL;

  // use assignment operator
  *this = other;

  // no return - constructor
}


/**
The overloaded assignment operator for the hash table ADT. Creates a clone of
the given hash table parameter in *this.

@param other   A valid HashTable of same type(s)

@return *this   A reference to the HashTable having a value assigned to it.

@pre
-# *this and HashTable other are valid instances of HashTables
-# Both HashTables are of same type(s)

@post
-# *this will contain equivalent data to HashTable other.

@code
@endcode
*/
template <typename DataType, typename KeyType>
HashTable<DataType, KeyType>& HashTable<DataType, KeyType>::operator=(
    const HashTable<DataType, KeyType>& other)
{
  // case: a different hash table than *this is being assigned
  if( this != &other )
  {
    // clear the data from this
    clear();

    // case: other is not empty
    if( !other.isEmpty() )
    {
      // copy other's table
      copyTable( other );
    }
  }

  // return *this
  return *this;
}


/**
The HashTable destructor. Returns all dynamic memory allocated for the HashTable
instance.

@pre
-# A HashTable was constructed.

@post
-# All dynamic memory will be returned before the object's destruction.

@detail @bAlgorithm
-# Calls delete to return the memory used by the dataTable member

@code
@endcode
*/
template <typename DataType, typename KeyType>
HashTable<DataType, KeyType>::~HashTable()
{
  // return the dataTable member's dynamic memory
  delete [] dataTable;
    dataTable = NULL;
    tableSize = 0;

  // no return - destructor
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   MUTATORS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
insert

Inserts the newDataItem into the tree of the appropriate table location. If an
item with the same key already exists in the data structure, then the data of
the existing item is replaced with the data of newDataItem.

@param newDataItem   The new data item to be inserted into the HashTable.

@pre
-# *this is a valid HashTable instance.
-# DataType newDataItem is of same type(s) as *this.

@post
-# DataType newDataItem will be inerted into a tree at the appropriate table
   location based on DataType's hashing function.

@detail @bAlgorithm
-# DataType's hash function is called using the key of newDataItem.
-# The result of the hashing is used to determine which table location
   newDataItem belongs in.
-# newDataItem is added to the appropriate tree.

@code
@endcode
*/
template <typename DataType, typename KeyType>
void HashTable<DataType, KeyType>::insert(const DataType& newDataItem)
{
  // variable(s)
  unsigned int hashNdx = 0;

  // get the hash result
  hashNdx = DataType::hash( newDataItem.getKey() );

  // use modulo to ensure that the item will fit in the table
  hashNdx %= tableSize;

  // insert the new item in the appropriate tree
  dataTable[ hashNdx ].insert( newDataItem );

  // no return - void
}


/**
remove

Locates and removes the item with the given key, if possible. Returns a boolean
flag to indicate if removal was successful (true), or otherwise (false).

@param deleteKey   The key of the item to be removed.

@return result   Indicates is a removal was performed.

@pre
-# A valid instance of a HashTable exists.

@post
-# The item with the given key will be removed, assuming it is present in the
   HashTable.
-# If the removal of an item with the given key was successful, true is
   returned, otherwise false is returned.

@detail @bAlgorithm
-# The hashing function is used to determine which table location the item will
   be in.
-# The appropriate tree's removal function is called.
-# The success of the operation is determined by the success of the removal
   operation performed by the tree.

@code
@endcode
*/
template <typename DataType, typename KeyType>
bool HashTable<DataType, KeyType>::remove(const KeyType& deleteKey)
{
  // variable(s)
  bool result = false;
  unsigned int hashNdx = 0;

  // use the hashing function to determine the table location
  hashNdx = DataType::hash( deleteKey );
  
  // use modulo to stay in table bounds
  hashNdx %= tableSize;

  // perform the removal operation
  result = dataTable[ hashNdx ].remove( deleteKey );

  // return the result
  return result;
}


/**
retrieve

Attempts to retrieve the item in the search table with the given key. Returns
a boolean flag, true if the item was found, false otherwise. The sought item is
passed back by reference if found.

@param searchKey    The key pertaining to the sought item.
@param returnItem   The object passed by reference through which the sought data
                    item will be retrieved.

@return result   A boolean flag indicating the success of the retrieval
                 operation. True is returned if an object with a matching key is
                 found, and false otherwise.

@pre
-# A valid instance of a HashTable exists.
-# A valid object of DataType is given for retrieving the sought data item.

@post
-# If found, the item with given key is copied into DataType returnItem to be
   passed back by reference.
-# A boolean flag indicating the success of the retrieval operation is returned,
   true if an item with a matching key was found, false otherwise.

@detail @bAlgorithm
-# The key is hashed to find the location of the item in the table.
-# The retrieve function of the appropriate tree is then called to locate an
   item with a matching key.
-# If the item is found, the reference parameter returnItem is overwritten with
   the found data item, otherwise, returnItem is in an undefined state.
-# The success of the operation, as determined by the retrieve operation
   performed by the tree, is returned, true for successful retrieval, false
   otherwise.

@code
@endcode
*/
template <typename DataType, typename KeyType>
bool HashTable<DataType, KeyType>::retrieve(const KeyType& searchKey,
                                            DataType& returnItem) const
{
  // variable(s)
  bool result = false;
  unsigned int hashNdx = 0;

  // find the position of the sought key in the table
  hashNdx = DataType::hash( searchKey ) % tableSize;

  // attempt the retrieval
  result = dataTable[ hashNdx ].retrieve( searchKey, returnItem );

  // return the retrieval result
  return result;
}


/**
clear

Clears the dataTable member by clearing every tree.

@pre
-# The function is called from a valid HashTable instance.

@post
-# The dataTable member of the HashTable will point to an empty forrest (array
   of empty trees).
-# The dataTable member will still indicate an array of tableSize, which will
   remain unmodified.

@detail @bAlgorithm
-# Iterates across the dataTable array and clears all trees.

@code
@endcode
*/
template <typename DataType, typename KeyType>
void HashTable<DataType, KeyType>::clear()
{
  // variables
  int ndx = 0;

  // iterate across the dataTable array
  for( ndx = 0; ndx < tableSize; ndx++ )
  {
    // clear the tree at this location
    dataTable[ ndx ].clear();
  }

  // no return - void
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   ACCESSORS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
isEmpty

Determines the state of the HashTable; returns true if it is empty, returns
false otherwise.

@return emtpy   A boolean flag indicating the state of the HashTable, returns
                true if empty, false otherwise.

@pre
-# A valid instance of the HashTable exists.

@post
-# The HashTable will remain unchanged.
-# The state of the emptyness is returned, true if empty, false otherwise.

@detail @bAlgorithm
-# If the HashTable contains a dataTable member of size 0 then it is empty.
-# Otherwise, if every tree in the array indicated by the dataTable member is
   empty, then the table is empty.

@code
@endcode
*/
template <typename DataType, typename KeyType>
bool HashTable<DataType, KeyType>::isEmpty() const
{
  // variable(s)
  bool empty = true;
  int ndx = 0;

  // iterate accross the array
  // this will not occur if the table size is 0
  for( ndx = 0; ndx < tableSize; ndx++ )
  {
    // case: the tree at the current location is not empty
    if( !dataTable[ ndx ].isEmpty() )
    {
      // the hash table is not empty
      empty = false;

      // there is no need to continue checking, break the loop
      break;
    }
  }

  // return the empty status
  return empty;
}


/**
showStructure

Displays the contents of the hash table by sequentially displaying the keys of
each location of the hash table.

PROVIDED BY THE LAB MANUAL PACKAGE

@pre
-# A valid instance of the HashTable exists.

@post
-# The keys of each hash table location will be sequentially displayed.

@code
@endcode
*/
template <typename DataType, typename KeyType>
void HashTable<DataType, KeyType>::showStructure() const
{
  // iterate over the table
  for (int i = 0; i < tableSize; ++i)
  {
    // print the hash table index
    cout << i << ": ";

    // write the keys in the tree at the current location
    dataTable[i].writeKeys();
  }

  // no return - void
}


/**
standardDeviation

Computes the standard deviation for the key distribution of the HashTable in its
current state.

@return result   The resulting standard deviation for the item distribution of
                 the HashTable's current storage state

@pre
-# A valid instance of the HashTable exists.

@post
-# The HashTable will remain unchanged.
-# The standard deviation of the hash table's item distribution will be
   returned.

@detail @bAlgorithm
-# The number of items in the table is found
-# Simultaneously, the average number of items per table location is found.
-# The differences between the number of entries found at each location are
   squared, then summed.
-# The previous result is then divided by the number of items in the table minus
   one.
-# The square root of the previous result is taken and the result is the
   standard deviation.

@code
@endcode
*/
template <typename DataType, typename KeyType>
double HashTable<DataType, KeyType>::standardDeviation() const
{
  // variable(s)
  double result = 0;

  // return the resulting standard deviation
  return result;
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            PRIVATE HELPER FUNCTIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*=====  Clone Helper  =======================================================*/

/**
copyTable

Creates a table equivalent to the one found in HashTable source.

@param source   A HashTable of same type(s) as *this.

@pre
-# Both HashTables are valid instances.
-# Both HashTables are of same type(s)

@post
-# The dataTable member of this will have its data deleted and then it will be
   rebuilt to be equivalent to the dataTable member of source.

@detail @bAlgorithm
-# The dataTable member of this has it's dynamic memory returned.
-# The tableSize member of this is then made equivalent to the one in source.
-# New dunamic memory is allocated for the dataTable member in *this.
-# The trees of source's dataTable are iteratively cloned into the dataTable of
   *this.

@code
@endcode
*/
template <typename DataType, typename KeyType>
void HashTable<DataType, KeyType>::copyTable(const HashTable& source)
{
  // variable(s)
  int ndx = 0;

  // case: dataTable has dynamic memory already
  if( dataTable != NULL )
  {
    // return the dynamic memory
    delete [] dataTable;
      dataTable = NULL;
  }

  // copy the table size member
  tableSize = source.tableSize;

  // allocate new dynamic memory
  dataTable = new BSTree<DataType, KeyType> [tableSize];

  // iteratively clone source's forrest
  for( ndx = 0; ndx < tableSize; ndx++ )
  {
    // clone the current tree
    dataTable[ ndx ] = source.dataTable[ ndx ];
  }

  // no return - void
}


/*=====  Mutators  ===========================================================*/

  // none


/*=====  Accessors  ==========================================================*/

  // none


