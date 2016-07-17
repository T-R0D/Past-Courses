/**
    @file ExpressionTree.h

    @author Terence Henriod

    Labaratory 8

    @brief Class declarations for the linked implementation of the Expression
           Tree ADT -- including the recursive helpers for the public member
           functions. This class defines a binary tree capable of holding prefix
           notation arithmetic expressions and evaluating them.

    @version Original Code 1.00 (10/18/2013) - T. Henriod
*/


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   PRECOMPILER DIRECTIVES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#ifndef EXPRESSIONTREE_H
#define EXPRESSIONTREE_H


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#include <stdexcept>
#include <iostream>
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
template <typename DataType>
class ExprTree {
 public:
  /*---   Constructor(s) / Destructor   ---*/
  ExprTree();
  ExprTree(const ExprTree<DataType>& source);
  ExprTree& operator=(const ExprTree<DataType>& source);
  ~ExprTree();

  /*---   Mutators   ---*/
  void build();
  void clear();
  void commute();

  /*---   Accessors   ---*/
  DataType evaluate() const throw (logic_error);
  void expression () const;
  bool isEquivalent( const ExprTree& source ) const;
  bool isEmpty() const;
  void showStructure() const;   // Output tree structure for testing/debugging

 private:
  /*---   Forwar Declaration of Inner Class   ---*/
  class ExprTreeNode;

  /*---   Helpers   ---*/
  void clone_sub( ExprTreeNode*& currentNode, ExprTreeNode* otherCurrent );
  void build_sub( ExprTreeNode*& currentNode );
  void clear_sub( ExprTreeNode*& currentNode );
  void commute_sub( ExprTreeNode*& currentNode );
  DataType evaluate_sub( ExprTreeNode* currentNode ) const throw (logic_error);
  void expression_sub( ExprTreeNode* currentNode, bool side ) const;
  bool isEquivalent_sub( ExprTreeNode* currentNode,
                         ExprTreeNode* otherCurrent ) const;
  bool clusterIsEquivalent( ExprTreeNode* currentNode,
                            ExprTreeNode* otherCurrent ) const;
  void showHelper ( ExprTreeNode* p, int level ) const;

  /*---   Data Members   ---*/
  class ExprTreeNode   // inner node class
  {
   public:
    /*---   Constructor   ---*/
    ExprTreeNode ( char elem, ExprTreeNode *leftPtr, ExprTreeNode *rightPtr );

    /*---   Data Members   ---*/
    char dataItem;          // Expression tree data item
    ExprTreeNode* left;     // Pointer to the left child
    ExprTreeNode* right;    // Pointer to the right child
  };

  ExprTreeNode* root;   // Pointer to the root node

};

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   TERMINATING PRECOMPILER DIRECTIVES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#endif		// #ifndef EXPRESSIONTREE_H


