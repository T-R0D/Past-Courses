/**
    @file BSTree.cpp

    @author Terence Henriod

    Laboratory 8

    @brief Class implementations declarations for the linked implementation of
           the Binary Search Tree ADT -- including the recursive helpers of the
           public member functions

    @version Original Code 1.00 (10/29/2013) - T. Henriod
*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

// Class Declaration
#include "BSTree.h"

// Other Dependencies
#include <iostream>
#include <stdexcept>
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
BSTree

The default constructor for the Binary Search Tree ADT. Constructs an empty
tree.

@pre
-# a tree with the calling identifier has not yet been instantiated

@post
-# an empty tree with the calling identifier will have been created

@detail @bAlgorithm
-# BSTreeNode* root data member is set to NULL

@code
@endcode
*/
template <typename DataType, typename KeyType>
BSTree<DataType, KeyType>::BSTree()
{
  // initialize data members
  root = NULL;

  // no return - constructor
}


/**
BSTree

The copy constructor for the Binary Search Tree ADT. Constructs a clone of the
given other tree.

@param other   another binary search tree object of similar types

@pre
-# a tree with the calling identifier has not yet been created

@post
-# a tree that is a clone of the given other tree has been created

@detail @bAlgorithm
-# BSTree* root data member is set to NULL
-# the overloaded operator= is called

@code
@endcode
*/
template <typename DataType, typename KeyType>
BSTree<DataType, KeyType>::BSTree( const BSTree<DataType, KeyType>& other )
{
  // initialize data members
  root = NULL;

  // perform the assignment of other to this
  *this = other;

  // no return - constructor
}


/**
operator=

The overloaded assignment operator. Assigns a clone of BSTree other to *this.

@param other   another binary search tree of similar type

@return *this

@pre
-# a tree with the calling identifier has been or is being created

@post
-# a tree that is a clone of the given other tree has been created
-# a reference to the new *this is returned

@detail @bAlgorithm
-# a check to see if *this is being assigned to itself is performed
-# otherwise *this is cleared
-# if BSTree other is not empty, the clone_sub private helper function is called
   to carry out the cloning process
-# if BSTree other is empty, no further action is taken
-# a reference to *this is returned

@code
@endcode
*/
template <typename DataType, typename KeyType>
BSTree<DataType, KeyType>& BSTree<DataType, KeyType>::operator= (
    const BSTree<DataType, KeyType>& other )
{
  // case: not assigning *this to itself
  if( this != &other )
  {
    // clear *this
    clear();

    // case: other is empty
    if( !other.isEmpty() )
    {
      // begin cloning process, starting at the roots
      clone_sub( root, other.root );
    }
    // case: other is empty
      // take no further action
  }
  // case: an attempt to assign *this to itself occurred
    // do nothing

  // return this
  return *this;
}


/**
~BSTree

The destructor for the Binary Search Tree ADT. Ensures all dynamically allocated
memory is returned.

@pre
-# a tree with the calling identifier has been instantiated

@post
-# *this tree will be destructed properly, returning all dynamically allocated
   memory

@detail @bAlgorithm
-# calls the clear function to delete all nodes contained in the tree

@code
@endcode
*/
template <typename DataType, typename KeyType>
BSTree<DataType, KeyType>::~BSTree()
{
  // clear the tree
  clear();

  // no return - destructor
}


/**
BSTreeNode

The default constructor for the BSTreeNode inner class. This constructor is
parameterized. Initializes the BSTreeNode members to the values of the given
parameters. 

@param nodeDataItem   the data item the new BSTreeNode will contain
@param leftPtr        a pointer to a left child
@param rightPtr       a pointer to a right child

@pre
-# a BSTree object has been instantiated for the BSTreeNode to exist in

@post
-# a BSTreeNode has been created with data members equivalent to the given
   parameter data

@detail @bAlgorithm
-# simply sets all data members of the node to the given data

@code
@endcode
*/
template <typename DataType, typename KeyType>
BSTree<DataType, KeyType>::BSTreeNode::BSTreeNode( const DataType &nodeDataItem,
                                                   BSTreeNode *leftPtr,
                                                   BSTreeNode *rightPtr )
{
  // initialize data members
  dataItem = nodeDataItem;
  left = leftPtr;
  right = rightPtr;

  // no return - constructor
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   MUTATORS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
clear

Clears all data from the BSTree.

@pre
-# a BSTree object has been instantiated

@post
-# all data will be removed from the tree
-# all dynamic memory will be returned
-# the the value of the BSTreeNode* root data member will be NULL

@detail @bAlgorithm
-# if the tree has contents the clear_sub private helper function is called with
   the BSTreeNode* root data member as a parameter in order to clear the entire
   tree
-# if the tree is empty, no further action is taken

@code
@endcode
*/
template <typename DataType, typename KeyType>
void BSTree<DataType, KeyType>::clear()
{
  // case: the tree contains data
  if( root != NULL )
  {
    // call the clear_sub helper function
    clear_sub( root );
  }
  // case: the tree is empty
    // no further action is taken

  // no return - void
}


/**
insert

Inserts newDataItem into the tree according to the items key. If a data item
with the same key already exists in the tree, then it is replaced with
newDataItem. Otherwise, a node is created in the appropriate place in the tree
to accomodate newDataItem.

@param newDataItem   an object of type DataItem to be inserted into the tree.

@pre
-# a BSTree object has been instantiated
-# the data type of the templated BSTree supports a getKey() member

@post
-# the tree will contain a BSTreeNode containing newDataItem, appropriately
   located relative to the pre-existing nodes

@detail @bAlgorithm
-# the key of the data item is obtained
-# the insert_sub helper is called, starting at the root, to locate the correct
   insertion location for the newDataItem

@code
@endcode
*/
template <typename DataType, typename KeyType>
void BSTree<DataType, KeyType>::insert( const DataType& newDataItem )
{
  // variables
  KeyType key = newDataItem.getKey();

  // call the helper function
  insert_sub( root, newDataItem, key );

  // no return - void
}


/**
remove

Removes a node with the given key from the tree. Returns the success of the
removal operation.

@param deleteKey   the key used to locate the item to be removed

@pre
-# a BSTree object has been instantiated
-# the data type of the templated BSTree supports a getKey() member

@post
-# if it exists, the node containing the data item with the given key will be
   removed
-# the success of the operation is returned to the calling function

@detail @bAlgorithm
-# the remove_sub helper is called, starting at the root, to locate the given
   key, if possible, and remove the node
-# if a removal occured true is returned, otherwise false is returned

@code
@endcode
*/
template <typename DataType, typename KeyType>
bool BSTree<DataType, KeyType>::remove( const KeyType& deleteKey )
{
  // return removal success
  return remove_sub( deleteKey, root );
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   ACCESSORS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
isEmtpy

Reports the state of the tree, specifically, returns true if the BSTree is
empty, and false if the tree has any contents.

@return bool empty   the truth value of the BSTree containing no nodes

@pre
-# a BSTree object has been instantiated

@post
-# the BSTree remains unchanged
-# the state of the tree is indicated via the return of a boolean flag

@detail @bAlgorithm
-# the remove_sub helper is called, starting at the root, to locate the given
   key, if possible, and remove the node
-# if a removal occured true is returned, otherwise false is returned

@code
@endcode
*/
template <typename DataType, typename KeyType>
bool BSTree<DataType, KeyType>::isEmpty() const
{
  // return the truth of the root pointer pointing to a node
  return (root == NULL);
}


/**
getHeight

Provides public access to find the current height of the BSTree.

@return int height  the height of the tree, that is, the number of vertices in
                    the longest chain in the tree

@pre
-# a BSTree object has been instantiated

@post
-# the BSTree remains unchanged
-# the current height of the tree is returned

@detail @bAlgorithm
-# the getHeight_sub helper is called, starting at the root, to trace the
   longest chain in the tree
-# the value returned by the helper function is the height of the tree

@code
@endcode
*/
template <typename DataType, typename KeyType>
int BSTree<DataType, KeyType>::getHeight() const
{
  // return the height of the tree
  return getHeight_sub( root );
}


/**
getCount

Provides public access to find the current count of items stored in the the
BSTree.

@return int count  the count of items in the tree

@pre
-# a BSTree object has been instantiated

@post
-# the BSTree remains unchanged
-# the current size of the tree is returned

@detail @bAlgorithm
-# the getCount_sub helper is called, starting at the root, to trace the tree
   and count the nodes
-# the value returned by the helper function is the size of the tree

@code
@endcode
*/
template <typename DataType, typename KeyType>
int BSTree<DataType, KeyType>::getCount() const
{
  // size up the tree
  return getCount_sub( root );
}


/**
retrieve

Provides public access to find the item of the given key stored in the tree. The
success of the operation is returned, while searchDataItem is modified by
reference. If the operation fails, searchDataItem is left unchanged.

@param searchKey        the key corresponding to the sought item
@param searchDataItem   the reference variable used to store the sought item if
                        found

@return bool found   the sucess in finding the sought item with the given key

@pre
-# a BSTree object has been instantiated

@post
-# the BSTree remains unchanged
-# the data item with the given key is passed back by reference if found
-# the truth value of whether or not the sought item was found

@detail @bAlgorithm
-# the retrieve_sub helper is called to search the tree for the item with the
   given key
-# if the given key is found, the DataType searchDataItem passed by reference
   will be given the value of the item with the sought key
-# if the item with the given key was found, true is returned, otherwise false
   is returned

@code
@endcode
*/
template <typename DataType, typename KeyType>
bool BSTree<DataType, KeyType>::retrieve( const KeyType& searchKey,
                                          DataType& searchDataItem ) const
{
  // attempt to retrieve the item, return the success of the operation
  return retrieve_sub( root, searchKey, searchDataItem );
}


/**
showStructure

Outputs the keys in a binary search tree. The tree is output
rotated counterclockwise 90 degrees from its conventional
orientation using a "reverse" inorder traversal. This operation is
intended for testing and debugging purposes only.

PROVIDED BY THE LAB MANUAL PACKAGE

@pre
-# a BSTree object has been instantiated

@post
-# the BSTree remains unchanged
-# the structure of the tree is displayed on the screen

@detail @bAlgorithm
-# the showHelper function is called to display the tree, starting from the root

@code
@endcode
*/
template <typename DataType, typename KeyType>
void BSTree<DataType, KeyType>::showStructure() const
{
  if ( root == NULL )
  {
    cout << "Empty tree" << endl;
  }
  else
  {
    cout << endl;
    showHelper(root,1);
    cout << endl;
  }
}


/**
writeKeys

Provides public access to have the keys currently contained in the tree listed
in increasing order on the screen. The DataType used must support the <<
operator.

@pre
-# a BSTree object has been instantiated

@post
-# the BSTree remains unchanged
-# the keys of the data items will be listed in increasing order on the screen,
   separated by a space

@detail @bAlgorithm
-# the writeKeys_sub helper function is called to carry out the process of
   displaying the keys, starting from the root
-# if the tree is empy, it is reported

@code
@endcode
*/
template <typename DataType, typename KeyType>
void BSTree<DataType, KeyType>::writeKeys() const
{
  // case: the tree is empty
  if( root == NULL )
  {
    // report that tree is empty
    cout << "The tree is empty." << endl;
  }
  // case: the tree has contents
  else
  {
    // call the helper function
    writeKeys_sub( root );
  }

  // no return - void
}


/**
writeLessThan

Provides public access to have the keys currently contained in the tree listed
in increasing order up to the given bound on the screen. The DataType used must
support the << operator.

@param searchKey   the upper bound of the keys to be listed. If this key is in
                   the list, it is listed, otherwise the key nearest the given
                   parameter is the larges key listed

@pre
-# a BSTree object has been instantiated

@post
-# the BSTree remains unchanged
-# the keys of the data items will be listed in increasing order on the screen,
   up to the upper bound, separated by a space

@detail @bAlgorithm
-# the writeLessThan_sub helper is called to search the tree for the item with
   the given key, or the nearest one that is less than the given key, starting
   at the root
-# all keys less than and including the given bound KeyType searchKey are listed
   if they are present in the tree
-# if no such keys exist, this is reported.

@code
@endcode
*/
template <typename DataType, typename KeyType>
void BSTree<DataType, KeyType>::writeLessThan( const KeyType& searchKey ) const
{
  // variables
  KeyType givenKey = searchKey;
  bool keysAlreadyWritten = false;

  // case: the tree is empty
  if( root == NULL )
  {
    // report that tree is empty
    cout << "The tree is empty." << endl;
  }
  // case: the tree has contents
  else
  {
    // call the helper function
    writeLessThan_sub( givenKey, root, NULL, keysAlreadyWritten );
                                           // no keys printed yet
  }

  // no return - void
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            PRIVATE HELPER FUNCTIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*=====  Clone Helper  =======================================================*/

/**
clone_sub

The private helper function that carries out a cloning operation by recursively
following the nodes of a given BSTree of same type and duplicating its nodes
using an in-order traversal. 

@param currentNode   the node currently being considered by the function
@param otherNode     the counterpart node in the other tree that is being cloned

@pre
-# a BSTree has been, or is being, instantiated
-# the clone_sub function has been called using a node pointer within the tree
   that points to NULL
-# the BSTreeNode* otherNode must point to valid data
-# other recursive calls previous to this one may have been made

@post
-# BSTreeNode* currentNode will point to an equivalent object to the one poined
   by BSTreeNode* otherNode
-# recursive calls will be made to continue the process down to the leaves of
   the tree

@detail @bAlgorithm
-# a new BSTreeNode is created with BSTreeNode* currentNode and given a copy of
   the dataItem held by other, the left and right pointers are set to NULL
-# checks are made to see if otherNode's left and right pointers are NULL or not
-# the function is called again to clone the other tree's left and right
   subtrees if possible

@code
@endcode
*/
template <typename DataType, typename KeyType>
void BSTree<DataType, KeyType>::clone_sub( BSTreeNode*& currentNode,
                                           const BSTreeNode* otherNode )
{
  // create the new node
  currentNode = new BSTreeNode( otherNode->dataItem, NULL, NULL );

  // case: there is a left subtree in other
  if( otherNode->left != NULL )
  {
    // continue the cloning process
    clone_sub( currentNode->left, otherNode->left );
  }

  // case: there is a left subtree in other
  if( otherNode->right != NULL )
  {
    // continue the cloning process
    clone_sub( currentNode->right, otherNode->right );
  }

  // no return - void
}

/*=====  Mutators  ===========================================================*/

/**
clear_sub

The private helper function that carries out the clearing of the tree. This
fucntion clears a given subtree by clearing the left and right subtrees and then
deleting the current node (post-order traversal). Ensures all dynamic memory is
returned for the given tree. 

@param currentNode   the node currently being considered by the function

@pre
-# a BSTree has been instantiated
-# the clear_sub function is called using a pointer that points to valid data

@post
-# the BSTree subtrees will be empty
-# the BSTreeNode* currentNode will point to NULL

@detail @bAlgorithm
-# each branch pointer is checked to see if it points to data
-# if the branch pointers point to valid subtrees, these are cleared
-# once any subtrees are cleared, the currentNode is deleted

@code
@endcode
*/
template <typename DataType, typename KeyType>
void BSTree<DataType, KeyType>::clear_sub( BSTreeNode*& currentNode )
{
  // attempt to clear any subtrees first
  // case: there is a left subtree
  if( currentNode->left != NULL )
  {
    // clear the left subtree
    clear_sub( currentNode->left );
  }

  // case: there is a left subtree
  if( currentNode->right != NULL )
  {
    // clear the left subtree
    clear_sub( currentNode->right );
  }

  // return the memory for the current node
  delete currentNode;
    currentNode = NULL;

  // no return - void
}


/**
insert_sub

The private helper function that inserts new data into the tree. If a node
containing a data item with a key equivalent to the given one, then the data
is replaced with the given data. If no match for the given key exists in the
tree, a new node is created to accomodate the data.

@param currentNode   the node currently being considered by the function
@param newDataItem   the data item to be inserted in the tree
@param key           the key of the item to be inserted

@pre
-# a BSTree has been instantiated

@post
-# if there is a node with a key equivalent to the given one, then its data is
   replaced with DataType neDataItem
-# if there is no matching key, a new node is created in the appropriate
   position of the tree to accommodate newDataItem

@detail @bAlgorithm
-# if the BSTreeNode* points to NULL, a new node is created and given the data
   of DataType newDataItem
-# otherwise, a check is made to see if it contains an equivalent key to the
   given one, if so, the data replacement operation is conducted
-# if the keys still don't match, checks to see which subtree the data with the
   given key belongs in, and the helper function is called to insert the data in
   the appropriate subtree

@code
@endcode
*/
template <typename DataType, typename KeyType>
void BSTree<DataType, KeyType>::insert_sub( BSTreeNode*& currentNode,
                                            const DataType& newDataItem,
                                            const KeyType& key )
{
  // variables
  KeyType currentKey;

  // case: the current node pointer is null
  if( currentNode == NULL )
  {
    // create a new node with the given data
    currentNode = new BSTreeNode( newDataItem, NULL, NULL );
  }
  // case: the pointer points to data
  else
  {
    // get the key of the current node 
    currentKey = currentNode->dataItem.getKey();

    // case: the keys match
    if( key == currentKey )
    {
      // replace the data in the current node with the given data
      currentNode->dataItem = newDataItem;
    }
    // case: the given key is less than the current one but greater than the
    //       previous one
    else if( key < currentKey )
    {
      // follow the subtree to find the appropriate place for the data
      insert_sub( currentNode->left, newDataItem, key );
    }
    // case: the given key is greater than the current one
    else
    {
      // follow the subtree to find the appropriate place for the data
      insert_sub( currentNode->right, newDataItem, key );
    }
  }

  // no return - void
}


/**
remove_sub

The private helper function used to delete data from the tree. In general, this
function performs similar to a remove function in a linked list. In the case
that a node to be removed has two children, the node is replaced with its
"in-order predecessor."

@param deleteKey     fad
@param currentNode   the pointer that points to the node currently being
                     considered. Note: this parameter is passed by reference,
                     so it is actually the pointer belonging to the node's
                     predecessor (or the root data member).

@return bool removed   a flag to indicate whether or not a node was located and
                       removed (true for removal, false for failure to remove)

@pre
-# a BSTree has been instantiated
-# BSTreeNod* currentNode points to the current location of the tree being
   considered currently
-# recursive calls to this function may have been made previously

@post
-# if a node containing data with a key equivalent to KeyType searchKey is
   found, it is removed in an appropriate manner and true is returned
-# otherwise, nothing occurs and false is returned
-# there are no guarantees as to the relative structure of the tree other than
   the tree will still fit the definition of a binary search tree

@detail @bAlgorithm
-# a search for a key equivalent to KeyType searchKey is conducted
-# if the function is called on a NULL pointer, then the search has failed,
   no removal can be performed, and false is returned
-# if the key matches a leaf, then the leaf is simply deleted
-# if the key matches a node with one child, then the tree is simply re-linked
   and the BSTreeNode* currentNode is linked to the appropriate child, and the
   excluded node is deleted
-# if the key matches a node with 2 children, then the "in-order predecessor"
   of the node is found, and the original node's data is replaced with that of
   the predecessor, and the now reduntant predecessor node is then deleted

@code
@endcode
*/
template <typename DataType, typename KeyType>
bool BSTree<DataType, KeyType>::remove_sub( const KeyType& deleteKey,
                                            BSTreeNode*& currentNode )
{
  // variables
  bool removed = false;
  KeyType currentKey;
  BSTreeNode* deleteCursor = currentNode;

  // case: currentNode points to valid data
  if( currentNode != NULL )
  {
    // get the key of the currentNode
    currentKey = currentNode->dataItem.getKey();

    // case: the keys match and we have found data to remove
    if( deleteKey == currentKey )
    {
      // remove the data in the appropriate manner
      // case: the node is a leaf
      if( ( currentNode->left == NULL ) &&
          ( currentNode->right == NULL ) )
      {
        // delete the node
        delete currentNode;
          currentNode = NULL;
      }
      // case: the node only has a left child
      else if( ( currentNode->left != NULL ) &&
               ( currentNode->right == NULL ) )
      {
        // re-link the list, the child takes the place of current node
        currentNode = currentNode->left;

        // delete the node (still pointed by deleteCursor)
        delete deleteCursor;
          deleteCursor = NULL;
      }
      // case: the node only has a right child
      else if( ( currentNode->left == NULL ) &&
               ( currentNode->right != NULL ) )
      {
        // re-link the list, the child takes the place of current node
        currentNode = currentNode->right;

        // delete the node (still pointed by deleteCursor)
        delete deleteCursor;
          deleteCursor = NULL;
      }
      // case: the node has two children
      else
      {
        // find the "in-order predecessor"
        // move to the left
        deleteCursor = deleteCursor->left;

        // move as far right as possible
        while( deleteCursor->right != NULL )
        {
          // advance the cursor
          deleteCursor = deleteCursor->right;
        }

        // move the data for the in-order predecessor to the current node
        currentNode->dataItem = deleteCursor->dataItem;

        // remove the redundant node
        remove_sub( deleteCursor->dataItem.getKey(), currentNode->left );
      }

      // the data was removed
      removed = true;
    }
    // case: the search continues
    else
    {
      // case: delete key is less than the current one
      if( deleteKey < currentKey )
      {
        // follow the left sub-tree
        removed = remove_sub( deleteKey, currentNode->left );
      }
      // case: delete key is greater than the current one
      else
      {
        // follow the right sub-tree
        removed = remove_sub( deleteKey, currentNode->right );
      }
    }
  }
  // otherwise we have reached a dead end
    // do nothing

  // return removal result
  return removed;
}


/*=====  Accessors  ==========================================================*/

/**
getHeight_sub

The private helper function that determines the height of the tree by counting
the vertices of the maximum length chain.

@param currentNode   the node currently being considered by the function
@param level         the level a node pointed by currentNode would be on

@return int height   the height of the subtree including currentNode

@pre
-# a BSTree has been instantiated
-# this function may have been called previously

@post
-# every chain in the train will have been followed in order to count the
   maximum length chain

@detail @bAlgorithm
-# if BSTreeNode* currentNode points to NULL, the value of level - 1 is returned
   (base case)
-# otherwise, the function is called to trace its subtrees to continue counting
   the tree height
-# the subtree heights are compared to find the greatest one

@code
@endcode
*/
template <typename DataType, typename KeyType>
int BSTree<DataType, KeyType>::getHeight_sub( BSTreeNode* currentNode ) const
{
  // variables
  int height = 0;
  int rightHeight = 0;

  // case: there may be subtrees to follow
  if( currentNode != NULL )
  {
    // get height of left subtree
    height = getHeight_sub( currentNode->left );

    // get height of right subtree
    rightHeight = getHeight_sub( currentNode->right );

    // case: the height of the right subtree is greater than the left one
    height = ( 1 + ( ( height < rightHeight ) ? rightHeight : height ) );
  }
  // case: the fuction has passed a leaf
    // do nothing

  // return the maximum height of the subtree including this node
  return height;
}


/**
getHeight_sub

The private helper function that counts all nodes in the current tree.

@param currentNode   the node currently being considered by the function

@pre
-# a BSTree has been instantiated
-# this function may have been called previously

@post
-# every valid node in the tree will be visited
-# the count of nodes in the subtree evaluated by the function is returned

@detail @bAlgorithm
-# 

@code
@endcode
*/
template <typename DataType, typename KeyType>
int BSTree<DataType, KeyType>::getCount_sub( BSTreeNode* currentNode ) const
{
  // variables
  int count = 0;

  // case: there may be subtrees to count
  if( currentNode != NULL )
  {
    // add the counts
    count = ( 1 + getCount_sub( currentNode->left ) +
              getCount_sub( currentNode->right ) );
  }

  // return the count of nodes for the subtree containing this node
  return count;
}


/**
retrieve_sub

The private helper function that counts all nodes in the current tree.

@param currentNode   the node currently being considered by the function

@pre
-# a BSTree has been instantiated
-# this function may have been called previously

@post
-# if the item with the given key is found, it is passed back by reference
   in the DataType searchDataItem, otherwise it is left unmodified and false
   is returned
-# the BSTree remains unchanged

@detail @bAlgorithm
-# if the item with the given key is found, it is passed back by reference and
   true is returned to indicate that the item was found
-# if the function is called with a NULL pointer, the item was not found,
   DataType searchDataItem is not modified, and true is returned

@code
@endcode
*/
template <typename DataType, typename KeyType>
bool BSTree<DataType, KeyType>::retrieve_sub( BSTreeNode* currentNode,
                                              const KeyType& searchKey,
                                              DataType& searchDataItem ) const
{
  // variables
  bool found = false;
  KeyType currentKey;

  // case: currentNode points to valid data
  if( currentNode != NULL )
  {
    // save the current key
    currentKey = currentNode->dataItem.getKey();

    // case: the sought key matches the key at the current node
    if( searchKey == currentKey )
    {
      // signal that the item was found
      found = true;

      // overwrite the reference parameter
      searchDataItem = currentNode->dataItem;
    }
    // case: the sought key is less than the current one
    else if( searchKey < currentKey )
    {
      // continue the search in the left subtree
      found = retrieve_sub( currentNode->left, searchKey, searchDataItem );
    }
    // case: the sought key is greater than the current one
    else
    {
      // continue the search in the right subtree
      found = retrieve_sub( currentNode->right, searchKey, searchDataItem );
    }
  }

  // return whether or not the item was found
  return found;
}


/**
showHelper

The private helper function that works to output the contents of the tree to the
screen.

@param p       a pointer to the node currently being considered
@param level   the level of the nodes to be output by this function call

@pre
-# a BSTree has been instantiated
-# this function may have been called previously
-# the DataType must support the << operator

@post
-# the tree will remain unchanged
-# the contents of the given subtree will be displayed on the screen

@detail @bAlgorithm
-# 

@code
@endcode
*/
template < typename DataType, typename KeyType >
void BSTree<DataType,KeyType>::showHelper( BSTreeNode *p, int level ) const
{
     int j;   // Loop counter

     if ( p != 0 )
     {
        showHelper(p->right,level+1);         // Output right subtree
        for ( j = 0 ; j < level ; j++ )    // Tab over to level
            cout << "\t";
        cout << " " << p->dataItem.getKey();   // Output key
        if ( ( p->left != 0 ) &&           // Output "connector"
             ( p->right != 0 ) )
           cout << "<";
        else if ( p->right != 0 )
           cout << "/";
        else if ( p->left != 0 )
           cout << "\\";
        cout << endl;
        showHelper(p->left,level+1);          // Output left subtree
    }
}


/**
writeKeysSub

The private helper function that works to output all keys in ascending order for
a given subtree.

@param currentNode   a pointer to the node currently being considered

@pre
-# a BSTree has been instantiated
-# BSTreeNode* currentNode must point to valid data
-# this function may have been called previously
-# the DataType must support the << operator

@post
-# the tree will remain unchanged
-# the keys of the given subtree will be listed on the screen

@detail @bAlgorithm
-# a check to ensure the given pointer is not null is performed
-# all keys in the left subtree of the tree currently being considered are
   listed first
-# the key of the current node is listed
-# all keys of the right subtree of the tree currently being considered are then
   listed

@code
@endcode
*/
template <typename DataType, typename KeyType>
void BSTree<DataType, KeyType>::writeKeys_sub( BSTreeNode* currentNode ) const
{
  // case: there is a left subtree
  if( currentNode->left != NULL )
  {
    // trace the left subtree
    writeKeys_sub( currentNode->left );
  }

  // output the current Node's key
  cout << currentNode->dataItem.getKey() << ' ';

  // case: there is a left subtree
  if( currentNode->right != NULL )
  {
    // trace the left subtree
    writeKeys_sub( currentNode->right );
  }

  // no return - void
}


/**
writeLessThan_sub

Lists all keys that are less than or equal to the given key (searchKey) on the
screen.

@param searchKey         the key given to indicate the upper (inclusive) bound
                         for the keys to be printed
@param start             a pointer for the node to start at
@param predecessor       a pointer to the start node's predecessor, call with
                         NULL if such a node does not exist
@param keysWerePrinted   a flag to be passed through all calls to indicate
                         wether or not keys less than the search key have been
                         found

@pre
-# a BSTree has been instantiated
-# BSTreeNode* start must point to valid data
-# BSTreeNode* predecessor must point to either a parent of start or contain the
   value NULL
-# previous recursive calls may have been made

@post
-# the tree will remain unchanged
-# once all recursive calls have resolved, all keys less than KeyType searchKey
   will be written in ascending order to the screen

@detail @bAlgorithm
-# the currentNode "cursor" is moved right as far as possible
-# with each move, the previous node and all nodes with keys less than it are
   listed
-# once the "cursor" has moved too far to the right, it is then moved left in an
   attempt to find more nodes with keys meeting the search criteria
-# a recursive call is made to repeat the process
-# the task completes when a call is made and no keys could be successfully
   printed (base case)
-# if no keys can be printed, this is indicated

@code
@endcode
*/
template <typename DataType, typename KeyType>
void BSTree<DataType, KeyType>::writeLessThan_sub( KeyType& searchKey,
                                                   BSTreeNode* start,
                                                   BSTreeNode* predecessor,
                                                   bool& keysWerePrinted
                                                 ) const
{
  // variables
  BSTreeNode* currentNode = start;
  BSTreeNode* previousNode = predecessor;
  KeyType currentKey = currentNode->dataItem.getKey();

  // attempt to find a node that is an "upper bound"
  // case: it is possible to move right on the tree
  if( searchKey >= currentKey )
  {
    // there will be keys listed
    keysWerePrinted = true;

    // go as far right on the tree is possible
    while( ( currentNode != NULL ) && ( searchKey >= currentKey ) )
    {
      // update the previousNode pointer
      previousNode = currentNode;

      // advance the currentNode pointer
      currentNode = currentNode->right;

      // if possible, get the new currentKey
      if( currentNode != NULL )
      {
        currentKey = currentNode->dataItem.getKey();
      }

      // at this point everything left of and including the previous node
      // has a key less than that of the currentNode, print these keys
      // case: previousNode has a left subtree
      if( previousNode->left != NULL )
      {
        // write the keys
        writeKeys_sub( previousNode->left );
      }
      // write the key of the previousNode
      cout << previousNode->dataItem.getKey() << ' ';
    }
  }

  // move left to find the "upper bound"
  while( ( searchKey < currentKey ) && ( currentNode != NULL ) )
  {
    // update the previousNode pointer
    previousNode = currentNode;

    // advance the currentNode pointer
    currentNode = currentNode->left;

    // if possible, get the new currentKey
    if( currentNode != NULL )
    {
      currentKey = currentNode->dataItem.getKey();
    }
  }

  // case: such a "bound" was found
  if( currentNode != NULL )
  {
    // continue writing the keys that are less than this one
    writeLessThan_sub( searchKey, currentNode, previousNode,
                       keysWerePrinted ); 
  }


/*
  // case: there are no keys less than or equal to the given key
  if( ( currentNode == NULL ) && !keysWerePrinted )
  {
    // report this to user
    cout << "No items with keys <=" << searchKey << " exist in this tree."
         << endl;
  }
*/

  // no return - void
}



/**
FunctionName

A short description

@param

@return

@pre
-# 

@post
-# 

@detail @bAlgorithm
-# 

@exception

@code
@endcode
*/

