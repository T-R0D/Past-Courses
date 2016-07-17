/**
    @file BSTree.h

    @author Terence Henriod

    Labaratory 9

    @brief Class declarations for the linked implementation of the Binary Search
           Tree ADT -- including the recursive helpers of the public member
           functions




    @version Original Code 1.00 (10/29/2013) - T. Henriod
*/


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   PRECOMPILER DIRECTIVES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#ifndef BSTREE_H
#define BSTREE_H


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#include <iostream>
#include <stdexcept>
using namespace std;


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   GLOBAL CONSTANTS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

const bool LEFT = true;
const bool RIGHT = false;


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
================================================================================
                   CLASS DEFINITION(S)
================================================================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
@class BSTree

The class implementations of the Binary Search Tree ADT. This class offers all
basic functionality of the Binary Search Tree.
*/
template <typename DataType, typename KeyType>
class BSTree
{
 public:
  /*---   Constructor(s) / Destructor   ---*/
  BSTree();
  BSTree( const BSTree<DataType, KeyType>& other );
  BSTree& operator= ( const BSTree<DataType, KeyType>& other );
  ~BSTree();


  /*---   Mutators   ---*/
  void clear();  
  void insert( const DataType& newDataItem );  // Insert data item
  bool remove( const KeyType& deleteKey );            // Remove data item


  /*---   Accessors   ---*/

  bool isEmpty() const;                        // Tree is empty
  // !! isFull() has been retired. Not very useful in a linked structure.
  int getHeight() const;                       // Height of tree
  int getCount() const;			  // Number of nodes in tree
  bool retrieve( const KeyType& searchKey, DataType& searchDataItem ) const;
                                                // Retrieve data item
  void showStructure() const;   // Output tree structure for testing/debugging
  void writeKeys() const;                      // Output keys
  void writeLessThan( const KeyType& searchKey ) const; // Output keys < searchKey



 protected:
  /*---   Forward Declaration of Inner Class   ---*/
  class BSTreeNode;


  /*---   Helpers   ---*/
  // Clone Helper
  void clone_sub( BSTreeNode*& currentNode, const BSTreeNode* otherNode );

  // Mutator Helpers
  void clear_sub( BSTreeNode*& currentNode );
  void insert_sub( BSTreeNode*& currentNode, const DataType& newDataItem,
                   const KeyType& key );
  bool remove_sub( const KeyType& deleteKey, BSTreeNode*& currentNode );

  // Accessor Helpers
  int getHeight_sub( BSTreeNode* currentNode ) const;
  int getCount_sub( BSTreeNode* currentNode ) const;
  bool retrieve_sub( BSTreeNode* currentNode, const KeyType& searchKey,
                     DataType& searchDataItem ) const;
  void showHelper( BSTreeNode* p, int level ) const;
  void writeKeys_sub( BSTreeNode* currentNode ) const;
  void writeLessThan_sub( KeyType& searchKey,
                          BSTreeNode* start,
                          BSTreeNode* predecessor,
                          bool& keysWerePrinted ) const;


  /*---   Data Members   ---*/
  class BSTreeNode   // Inner class: facilitator for the BSTree class
  {
   public:
    /*---   Constructor   ---*/
    BSTreeNode( const DataType &nodeDataItem, BSTreeNode *leftPtr,
                BSTreeNode *rightPtr );

    /*---   Data Members   ---*/
    DataType dataItem;   // Binary search tree data item
    BSTreeNode* left;    // Pointer to the left child
    BSTreeNode* right;   // Pointer to the right child
  };

  BSTreeNode* root;   // Pointer to the root node


 private:

};

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   TERMINATING PRECOMPILER DIRECTIVES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#endif		// #ifndef BSTREE_H


