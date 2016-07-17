/**
    @file ExpressionTree.cpp

    @author Terence Henriod

    Laboratory 8

    @brief Class implementations for the linked implementation of the Expression
           Tree ADT -- including the recursive helpers for the public member
           functions. This class defines a binary tree capable of holding prefix
           notation arithmetic expressions and evaluating them.

    @version Original Code 1.00 (10/18/2013) - T. Henriod
*/

/*     QUESTIONS

1. Can we assume that we will be given only valid expressions
  a. Do we need to ensure that the tree contains at least one operator and two
     operands OR one operand (a single number)?

  b. Should we throw errors if someone tries to get an expression or result
     from an empty tree?

*/


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

// Class Declaration
#include "ExpressionTree.h"

// Other Dependencies
#include <iostream>


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
================================================================================
                   CLASS FUNCTION IMPLEMENTATIONS
================================================================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   CONSTRUCTOR(S) / DESTRUCTOR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
ExprTree

The default constructor for the Expression Tree ADT Class. Constructs an empty
list.

@pre
-# there is available memory for the expression tree

@post
-# an object of type ExprTree will be constructed
-# the root pointer member will hold the value NULL
-# the tree will contain no expression data

@detail @bAlgorithm
-# mean is calculated by adding all the values
in scoreData[] and dividing them by numScores
-# median is found by sorting the scoreData[] list by value,
and then finding the middle value of an odd numbered list,
or the average of the two middle values
in an even numbered list
-# mode is found by sorting scoreData[] list by value,
and then iterating through the list searching
for the largest number of identical scores

@exception out_of_range if numScores <= 0, zero (0) is returned
without further processing

@code
@endcode
*/

/**
ExprTree

The default constructor for the Expression Tree ADT Class. Constructs an empty
expression tree.

@pre
-# there is available memory for the expression tree

@post
-# an object of type ExprTree will be constructed
-# the root pointer member will hold the value NULL
-# the tree will contain no expression data

@detail @bAlgorithm
-# the object is simply instantiated and the root pointer is given the value
NULL

@code
@endcode
*/
template <typename DataType>
ExprTree<DataType>::ExprTree()
{
  // initialize data members
  root = NULL;

  // no return - constructor
}


/**
ExprTree

The copy constructor for the Expression Tree ADT Class. Constructs a cloned copy
of the given source tree.

@param source   another Expression tree of same type as *this

@pre
-# there is available memory for the expression tree

@post
-# an object of type ExprTree will be constructed
-# the tree will contain identical data to the provided source tree

@detail @bAlgorithm
-# the object is simply instantiated, cleared, and then cloned using operator=

@code
@endcode
*/
template <typename DataType>
ExprTree<DataType>::ExprTree(const ExprTree<DataType>& source)
{
  // initialize data members
  root = NULL;

  // clone the source tree into *this
  *this = source;

  // no return - constructor
}


/**
operator=

The overloaded assignment operator for the expression tree. Simply clones the 
given source tree into *this.

@param source  another Expression tree of same type as *this

@return *this   a dereferenced pointer to the object using operator=

@pre
-# there is available memory for the expression tree

@post
-# the tree will contain identical data to the provided source tree

@detail @bAlgorithm
-# a check is performed to avoid copying the tree into itself
-# if the source is a different object, *this is cleared
-# every node of source is then cloned into *this using the clone_sub helper
function


@code
@endcode
*/
template <typename DataType>
ExprTree<DataType>& ExprTree<DataType>::operator=(
    const ExprTree<DataType>& source)
{
  // ensure we are not copying *this
  if( this != &source ) 
  {
    // clear *this
    clear();

    // ensure we are not copying an empty list
    if( source.root != NULL )
    {
      // follow the source tree down to the leaves, adding nodes in a pre-order
      // traversal manner
      clone_sub( root, source.root );
    }
  }

  // return *this
  return *this;
}

/**
~ExprTree

The destructor for the Expression Tree ADT class. Ensures memory is not leaked
upon destruction of an expression tree.

@pre
-# an Expression Tree exists

@post
-# the tree will be destroyed with all dynamic memory returned to the heap

@detail @bAlgorithm
-# calls the clear function to delete any dynamically allocated data nodes

@code
@endcode
*/
template <typename DataType> 
ExprTree<DataType>::~ExprTree()
{
  // clear the tree
  clear();

  // no return - destructor
}


/**
ExprTreeNode

The only constructor for the inner class ExprTreeNode. Requires parameters that
correspond to all data members of an ExprTreeNode.

@param elem   the expression element character the node will contain
@param leftPtr   the pointer to any children on the left the node will have
@param leftPtr   the pointer to any children on the right the node will have

@pre
-# a tree containing an expression has been created

@post
-# a new tree node has been instantiated with the parameterized data used as its
member data

@detail @bAlgorithm
-# a new node is created
-# the char dataItem member is given the value of char elem
-# the ExprTreeNode* left member is given the value of ExprTreeNode* leftPtr
-# the ExprTreeNode* right member is given the value of ExprTreeNode* rightPtr

@code
@endcode
*/
template <typename DataType>
ExprTree<DataType>::ExprTreeNode::ExprTreeNode( char elem,
                       ExprTreeNode *leftPtr, ExprTreeNode *rightPtr )
{
  // set all data members to the given  parameter values
  dataItem = elem;
  left = leftPtr;
  right = rightPtr;

  // no return - constructor 
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   MUTATORS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
build

Offers public interface for building an Expression Tree from an expression
entered via the keyboard.

@pre
-# there is an instantiated Expression Tree
-# the user has been prompted to enter an expression via the keyboard

@post
-# the tree will contain the entered prefix arithmetic expression

@detail @bAlgorithm
-# clears the tree to accomodate the new expression
-# simply calls the build_sub helper function to accomplish the task

@code
@endcode
*/
template <typename DataType>
void ExprTree<DataType>::build()
{
  // clear *this to make way for a new expression
  clear();

  // start building starting with the root
  build_sub( root );

  // no return - void
}


/**
clear

Offers public interface to clear the tree.

@pre
-# there is tree object currently instantiated

@post
-# the tree will no longer contain any data

@detail @bAlgorithm
-# simply calls the clear_sub helper function


@code
@endcode

*/
template <typename DataType>
void ExprTree<DataType>::clear()
{
  // call the clearing helper to start deleting at the root
  // once this is done, we can expect an empty tree with root == NULL
  clear_sub( root );

  // no return - void
}


/**
commute

Offers public interface to commute (read: flip) the entire tree and
its sub-trees.

@pre
-# a tree containint a valid expression is instantiated

@post
-# the tree will contain an equivalent expression to the original one, but
commuted about appropriate operators

@detail @bAlgorithm
-# a check is performed to avoid working on an empty tree
-# the commute_sub helper function is called to accomplish the task

@code
@endcode
*/
template <typename DataType>
void ExprTree<DataType>::commute()
{
  // case: tree is not empty
  if( root != NULL )
  {
    // call the commute helper function
    commute_sub( root );
  }

  // no return - void
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   ACCESSORS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
expression

Offers public interface for the expression contained in the tree to be output
to the screen in standard fully parenthesized infix form.

@pre
-# the tree contains an arithmetic expression

@post
-# the expression contained in the tree will be displayed on the screen in fully
parenthesized infix form

@detail @bAlgorithm
-# a check to ensure the tree is not empty is performed
-# otherwise, the expression_sub helper function is called to accomplish the
task

@code
@endcode
*/
template <typename DataType>
void ExprTree<DataType>::expression() const
{
  // case: there is no expression to evaluate
  if( root == NULL )
  {
    // notify the user
    cout << "The tree contains no expression." << endl; 
  }
  // otherwise, get the expression
  else
  {
    // construct the expression in parenthesised infix form
    expression_sub( root, LEFT );
  }

  // no return - void
}


/**
evaluate

Evaluates the arithmetic expression contained in the tree.

@return result   the arithmetic result represented as the given type

@pre
-# there is a valid arithmetic expression contained in the tree

@post
-# the expression in the tree will be evaluated and returned

@detail @bAlgorithm
-# a check is performed to ensure the tree is not empty
-# the evaluate_sub helper function is called to accomplish the task

@exception logic_error   thrown if the function is called on an empty tree

@code
@endcode
*/
template <typename DataType>
DataType ExprTree<DataType>::evaluate() const throw (logic_error)
{
  // variables
  DataType result;

  // case: the tree does not contain an expression
  if( root == NULL )
  {
    // throw and error to indicate such
    throw logic_error( "Error - No expression to evaluate" );
  }
  // otherwise, continue evaluating
  else
  {
  // evaluate the expression stored in the tree, start at the root
  result = evaluate_sub( root );
  }

  // return the result
  return result; 
}


/**
isEquivalent

Offers public access to check for equivalency between *this and a source tree.
Does not check for an expression that evaluates to the same value, but rather
checks for an appropriately commuted expression.

@param source   an Expression Tree of same type to compare to *this

@return result  a boolean value indicating the state of the two trees'
equivalency

@pre
-# there are two instantiated trees to compare

@post
-# both trees will remain unchanged
-# the equivalency of the two trees will be reported

@detail @bAlgorithm
-# a check is performed to see if the two trees are one and the same, in which
case they are equivalent
-# a check is performed to see if the two trees are empty, in which case they
are equivalent
-# if either of the previous checks pass, the isEquivalent_sub helper function
is called to finish the evaluation

@code
@endcode
*/
template <typename DataType>
bool ExprTree<DataType>::isEquivalent(const ExprTree& source) const
{
  // variables
  bool result = false;

  // case: source tree actually is *this tree
  if( this == &source )
  {
    result = true;
  }
  // case: both trees are empty
  else if( ( root == NULL ) && ( source.isEmpty() ) )
  {

    // the trees are equivalent
    result = true;
  }
  // case: the trees may not both be empty
  {
    // call the equivalency helper function
    result = isEquivalent_sub( root, source.root );
  }

  // return the result
  return result;
}


/**
isEmpty

Offers public access for checking the empty status of the tree.

@return result   the boolean result of evaluating if the root pointer points to
NULL

@pre
-# the Expression Tree has been instantiated

@post
-# the tree remains unchanged while it's empty status is reported

@detail @bAlgorithm
-# test to see if the root member is a NULL pointer

@code
@endcode
*/
template <typename DataType>
bool ExprTree<DataType>::isEmpty() const
{
  // return the state of the tree
  return (root == NULL);
}


/**
showStructure

Offers public interface for displaying the tree contents. Outputs an expression
tree. The tree is output rotated counter-clockwise 90 degrees from its
conventional orientation using a "reverse" inorder traversal. This operation is
intended for testing and debugging purposes only.

PROVIDED BY LAB MANUAL PACKAGE

@pre
-# a tree has been instantiated

@post
-# the contents of the tree will be output to the screen
-# the tree will remain unchanged

@detail @bAlgorithm
-# if the tree is empty, this is reported via displaying so to the screen
-# otherwise, the showHelper function is called starting at the root

@code
@endcode
*/
template <typename DataType>
void ExprTree<DataType>::showStructure() const
{
  if ( isEmpty() )
    cout << "Empty tree" << endl;
  else
  {
    cout << endl;
    showHelper(root,1);
    cout << endl;
  }
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            PRIVATE HELPER FUNCTIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
clone_sub

The private helper function that recursively clones another tree.

@param currentNode   a pointer passed by reference for the current recursive
call to act on

@param otherCurrent   a pointer to the current node's counter part in the other
tree, used for copying and following the other tree down

@pre
-# there is an instantiated tree
-# the other tree is not empty
-# currentNode currently points to NULL

@post
-# the current node will contain the same dataItem as the other tree's node
-# appropriate recursive calls to maintain lower level nodes have been made

@detail @bAlgorithm
-# a NULL node pointer is passed in
-# a new node is dynamically allocated and given the same dataItem value as the
counterpart node in the other tree. the left and right pointers in the newly
created node are set to point to NULL
-# if a subtree on the otherCurrent node's left exists, the clone_sub is called
again, with the left pointers of each current node as parameters
-# if a subtree on the otherCurrent node's tight exists, the clone_sub is called
again, with the right pointers of each current node as parameters

@code
@endcode
*/
template <typename DataType>
void ExprTree<DataType>::clone_sub( ExprTreeNode*& currentNode,
                                    ExprTreeNode* otherCurrent )
{
  // create a copy of the current node in *this
  currentNode = new ExprTreeNode( otherCurrent->dataItem, NULL, NULL );

  // continue to follow the tree to its leaves, copying down each branch
  // case: other's left branch exists
  if( otherCurrent->left != NULL )
  {
    // attempt to copy the sub-tree
    clone_sub( currentNode->left, otherCurrent->left );
  }
  // case: other's right branch exists
  if( otherCurrent->right != NULL )
  {
    // attempt to copy the sub-tree
    clone_sub( currentNode->right, otherCurrent->right );
  }

  // no return, void
}


/**
build_sub

The private helper function for constructing a tree that contains the tree
specified in the keyboard input

@param currentNode

@pre
-# a tree has been instantiated
-# the user has entered a complete and correct expression into the iostream
-# each recursive call is given a NULL pointer

@post
-# with each recursive call, another element of the expression will be added
to the tree in the current node
-# upon completion of recursive calle, the tree will contain the entire
expression

@detail @bAlgorithm
-# a character representing a part of the expression is read in
-# a node is created with the new element and its left and right pointers are
set to NULL
-# if the new element was a valid operator (and not a digit or operand)
recursive calls are made to continue building the sub-trees

@code
@endcode
*/
template <typename DataType>
void ExprTree<DataType>::build_sub( ExprTreeNode*& currentNode )
{
  // variables
  char newdataItement;

  // read in the next character of the expression
  cin >> newdataItement;

  // create a new node with the read in data
  currentNode = new ExprTreeNode( newdataItement, NULL, NULL );

  // case: the read character was an operator
  if( (newdataItement == '+') ||
      (newdataItement == '-') ||
      (newdataItement == '*') ||
      (newdataItement == '/') )
  {
    // continue to build the tree by creating further sub-trees
    // build the left branch
    build_sub( currentNode->left );

    // build the right branch
    build_sub( currentNode->right );
  }
  // otherwise, there is no more work to do

  // no return - void
}


/**
expression_sub

The private helper function used to display the expression contained in the tree
in fully parenthesised infix form

@param currentNode   a pointer to the node currently being read
@param side   a boolean indicator as to which branch or side (left or right) the
function is tracing 

@pre
-# a tree containing an expression has been created
-# the function was called using an appropriate side indicator value

@post
-# all subtrees have been recursively read to provide the representation of the
expression thus far
-# the expression thus far will be displayed on the screen
-# the tree will remain unchanged

@detail @bAlgorithm
-# if the current node contains an operand and was called for the left side,
then an open parenthesis and the operand are written to the screen
-# if the curren node contains an operator, the operator with surrounding spaces
is written to the screen
-# if the current node contains an operand and was called for the right side,
then the operand and a close parenthesis are written to the screen

@code
@endcode
*/
template <typename DataType>
void ExprTree<DataType>::expression_sub( ExprTreeNode* currentNode,
                                         bool side ) const
{
  // variables
    // none

  // traverse the tree in-order
  // case: the node data item is a left side operand
  if( ((currentNode->dataItem >= '0') && (currentNode->dataItem <= '9' ))
      && (side == LEFT) )
  {
    // print an open parenthesis, the operand, and a space
    cout << '(' << currentNode->dataItem;
  }
  // case: the node dataItement is an operator
  else if( (currentNode->dataItem == '+') ||
           (currentNode->dataItem == '-') ||
           (currentNode->dataItem == '*') ||
           (currentNode->dataItem == '/') )
  {
    // find the left side operand
    expression_sub( currentNode->left, LEFT );

    // print the operator and a space
    cout << ' ' << currentNode->dataItem << ' ';

    // find the right side operand
    expression_sub( currentNode->right, RIGHT );
  }
  // case: the node dataItement is a right side operand
  else
  {
    // print the number and the close parenthisis
    cout << currentNode->dataItem << ')';
  }

  // no return - void
}


/**
evaluate_sub

The private helper function used to evaulate the expression stored in the tree.

@param currentNode   a pointer to the node currently being read included in the
evaluation of the expression

@return result   the result of the evaluated expression for the sub-tree
evaluated thus far

@pre
-# a tree containing an expression has been created

@post
-# all subtrees have been recursively evaluated
-# the result of the evaluation (including this node) will be returned for use
by previous recursive calls or the original evaluate() call
-# the tree will remain unchanged

@detail @bAlgorithm
-# if the current node contains an operand (digit), it is converted to an
equivalent representation in the tree's specified data type an stored in result
-# otherwise, it is assumed that the current node contains a valid operator and
recursive calls are made to lower levels to evaluate the corresponding results
-# appropriate arithmetic is performed and the value is stored to the result
-# the result is returned

@exception logic_error   an exception is thrown if an inappropriate operand has
been detected in the tree

@code
@endcode
*/
template <typename DataType>
DataType ExprTree<DataType>::evaluate_sub( ExprTreeNode* currentNode )
                                           const throw (logic_error)
{
  // variables
  DataType leftResult;
  DataType rightResult;
  DataType result;

  // check for the class of item stored at this node
  // case: operand
  if( currentNode->dataItem >= '0' && currentNode->dataItem <= '9' )
  {
    result =  DataType( currentNode->dataItem - '0' );  
  }
  // case: operator
  else
  {
    // evaluate the expressions stored in the sub-trees
    // get the left and right results
    leftResult = evaluate_sub( currentNode->left );
    rightResult = evaluate_sub( currentNode->right );

    // evaluate based on the operand
    switch( currentNode->dataItem )
    {
      // addition
      case '+':
        // simply add the numbers
        result = leftResult + rightResult;
        break;

      // subtraction
      case '-':
        result = leftResult - rightResult;
        break;

      // multiplication
      case '*':
        result = leftResult * rightResult;
        break;

      // division
      case '/':
        // first check for the forbidden divide by 0 condition  TODO: get this included
        //if( int( rightResult ) == 0 )
        //{
          // throw an exception for this
          //throw logic_error( "ERROR - Dividing by 0 is undefined" );
        //}
 
        // otherwise, operate as normal
        result = leftResult / rightResult;
        break;

      // corrupted data
      default:
        // if something other than an operand or a digit is stored here, bad
        // input was taken in
        throw logic_error( "ERROR - An dataItement is not an operator or operand" );
        break;
    }
  }

  // return result
  return result;
}


/**
clear_sub

The private helper function used to delete all nodes in the tree

@param currentNode   a pointer to the node currently being operated on

@pre
-# the function was called using an appropriate pointer

@post
-# all subtrees have been recursively removed with dynamic memory de-allocated

@detail @bAlgorithm
-# a check is made to see if the given pointer points to NULL to see if action
is required
-# the clear_sub function is called to clear the sub-tree on the right,
recursively
-# the clear-sub function is called to clear the sub-tree on the left,
recursively
-# once sub-trees are cleared, the current node is then deallocated

@code
@endcode
*/
template <typename DataType>
void ExprTree<DataType>::clear_sub( ExprTreeNode*& currentNode )
{
  // case: the parameter pointer actually points to an intantiated node
  if( currentNode != NULL ) 
  {
    // attempt to clear any "child" nodes and their "children"
      // case: right child exists
      if( currentNode->right != NULL )
      {
        // clear the corresponding sub-tree
        clear_sub( currentNode->right );
      }

      // case: left child exists
      if( currentNode->left != NULL )
      {
        // clear the corresponding sub-tree
        clear_sub( currentNode->left );
      }

    // delete the current node
    delete currentNode;
      currentNode = NULL;
  }
  // otherwise, there is nothing to clear

  // no return - void
}


/**
commute_sub

The private helper function used to commute the tree about appropriate
operators. Literally flip-flops the entire tree wherever possible while
maintaining equivalency

@param currentNode   a pointer to the node currently being operated on

@pre
-# a tree containing an expression has been created
-# the function was called with a non NULL pointer as a parameter

@post
-# all subtrees have been recursively been commuted where possible
-# if possible, the tree will be commuted about the current node before the
function resolves

@detail @bAlgorithm
-# if the current node is a leaf, there is nothing to commute
-# if a left child exists, then an attempt to commute that sub-tree is made
-# if a right child exists, then an attempt to cummute that sub-tree is made
-# finally, if the current node contains an appropriately commutable operator,
the left and right pointers have their pointed addresses switched

@code
@endcode
*/
template <typename DataType>
void ExprTree<DataType>::commute_sub( ExprTreeNode*& currentNode )
{
  // variables
  ExprTreeNode* temp;

  // case: the current node is not a leaf
  if( (currentNode->left != NULL) || (currentNode->right != NULL) )
  {
    // case: a left leaf of the tree have not been reached
    if( currentNode->left != NULL )
    {
      // follow the tree down to the leaves if possible
      commute_sub( currentNode->left );
    }

    // case: a right leaf of the tree have not been reached
    if( currentNode->right != NULL )
    {
      // follow the tree down to the leaves if possible
      commute_sub( currentNode->right );
    }

    // recursive calls resolved, sub-commuting complete
    // case: this is a commutable operand  TODO: make this mathematically correct
    if( (currentNode->dataItem == '+') ||
        (currentNode->dataItem == '-') ||
        (currentNode->dataItem == '*') ||
        (currentNode->dataItem == '/') )
    {
      // swap the branch pointers, effectively re-ordering tree
      temp = currentNode->left;
      currentNode->left = currentNode->right;
      currentNode->right = temp;
    }
  }

  // no return - void 
}


/**
isEquivalent_sub

The private helper function used to test for equivalency.

@param currentNode   a pointer to the node currently being evaluated
@param otherCurrent   a pointer to what should be the counterpart of currentNode

@pre
-# a tree containing an expression has been created

@post
-# all subtrees have been recursively evaluated to provide the result of the
equivalency evaluation thus far
-# both trees will remain unchanged

@detail @bAlgorithm
-# if the current node pointers point to NULL, they are equivalent "nodes"
-# if the nodes contain the same value the test continues
-# the branches are then tested for equivalency
-# if this test returns false, then the test is performed again with
commuted branches of *this if *this contains an appropriate operator

@code
@endcode
*/
template <typename DataType>
bool ExprTree<DataType>::isEquivalent_sub( ExprTreeNode* currentNode,
                                           ExprTreeNode* otherCurrent ) const
{
  // variables
  bool result = false;

  // case: either of the pointers point to NULL
  if( ( currentNode == NULL ) || ( otherCurrent == NULL ) )
  {
    // case: both pointers point to null
    if( ( currentNode == NULL ) && ( otherCurrent == NULL ) )
    {
      // the "non-existent" leaves are equivalent
      result = true;
    }
  }
  // case: the nodes contain the same value
  else if( currentNode->dataItem == otherCurrent->dataItem )
  {
    // test for equivalency between right and left branches
    result = ( isEquivalent_sub( currentNode->left, otherCurrent->left )
               && isEquivalent_sub( currentNode->right, otherCurrent->right) );

    // case: equivalency was not found
    if( !result )
    {
      // attempt the test again, but with commuted branches, if possible
      if( (currentNode->dataItem == '+') ||
          (currentNode->dataItem == '*') )
      {
        // attempt the evaluation
        result = ( isEquivalent_sub( currentNode->right, otherCurrent->left )
                 && isEquivalent_sub( currentNode->left, otherCurrent->right) );
      }
    }
  }
 
  // return result
  return result;
}


/**
showHelper

Recursive helper for the showStructure() function. Outputs the
subtree whose root node is pointed to by p. Parameter level is the
level of this node within the expression tree.

PROVIDED BY LAB MANUAL PACKAGE

@param p   the pointer to the current node being operated on
@param level   the level of the tree the node resides in

@pre
-# a tree has been instantiated
-# the data held by the tree's nodes is compatible with operator<<

@post
-# the contents of the tree will be output to the screen
-# the tree will remain unchanged

@detail @bAlgorithm
-# outputs all elements along the right side of the tree recursively
-# using recursion to "back up, the left sides of subtrees are output and then
-# the operation recursively continues down the left and right sides of
sub-trees
-# in this manner the tree is displayed in a "reverse post-order" manner
-# the tree itself remains unchanged

@code
@endcode
*/
template <typename DataType>
void ExprTree<DataType>::showHelper ( ExprTreeNode *p, int level ) const
{
  int j;   // Loop counter

  if ( p != NULL )
  {
  showHelper(p->right,level+1);        // Output right subtree
    for ( j = NULL ; j < level ; j++ )   // Tab over to level
      cout << "\t";
      cout << " " << p->dataItem;        // Output dataItem
      if ( ( p->left != NULL ) &&          // Output "connector"
        ( p->right != NULL ) )
        cout << "<";
      else if ( p->right != NULL )
        cout << "/";
      else if ( p->left != NULL )
        cout << "\\";
        cout << endl;
        showHelper(p->left,level+1);         // Output left subtree
  }
}


