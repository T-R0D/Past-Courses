template <typename DataType, typename KeyType>
bool BSTree<DataType>::remove_sub( const KeyType& deleteKey,
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
        remove_sub( deleteKey, currentNode->left );
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
        removed = remove_sub( searchKey, currentNode->left );
      }
      // case: delete key is greater than the current one
      else
      {
        // follow the right sub-tree
        removed = remove_sub( searchKey, currentNode->right );
      }
    }
  }
  // otherwise we have reached a dead end
    // do nothing

  // return removal result
  return removed;
}







template <typename DataType, typename KeyType>
bool BSTree<DataType, KeyType>::remove_sub( const KeyType& deleteKey,
                                            BSTreeNode* start,
                                            BSTreeNode* predecessor )
{
  // variables
  bool removed = false;
  bool lastDirection; 
    if( predecessor != NULL )
    {
      // initialize the variable for the given data
      lastDirection =
                    ((predecessor->left == start) ? LEFT : RIGHT);
    }
  KeyType currentKey;
    if( start != NULL )
    {
    currentKey = start->dataItem.getKey();
    }
  BSTreeNode* currentNode = start;
  BSTreeNode* previousNode = predecessor;
  BSTreeNode* inorderPredecessor = NULL;

  // attempt to find the data item to be removed
  while( ( currentNode != NULL ) &&
         ( currentKey != deleteKey ) )
  {
    // remember the previous node
    previousNode = currentNode;

    // advance the node to remove pointer based on the keys
    // case: the sought key is less than the current key
    if( deleteKey < currentKey )
    {
      // follow the left subtree
      currentNode = currentNode->left;
      lastDirection = LEFT;
    }
    // case: the sought key is greater than the current key
    else
    {
      // follow the left subtree
      currentNode = currentNode->right;
      lastDirection = RIGHT;
    }

    // case: the currentNode is not pointing to NULL
    if( currentNode != NULL )
    {
      // store the now pointed node's key
      currentKey = currentNode->dataItem.getKey();
    }
  }

  // case: such a key was found
  if( currentNode != NULL )
  {
    // handle the data removal
    // case: the node is a leaf
    if( ( currentNode->left == NULL ) &&
        ( currentNode->right == NULL ) )
    {
      // clean up the predecessor's dangling pointer
      // case: there is a predecessor
      if( previousNode != NULL )
      {
        // case: a left leaf is being deleted
        if( lastDirection == LEFT )
        {
          previousNode->left = NULL;
        }
        // case: a right leaf is being deleted
        else
        {
          previousNode->right = NULL;
        }
      }
      // case: there was not a predecessor
        // the predecessor is NULL only when the root is being deleted
      else
      {
        // clean up dangling root pointer
        root = NULL;
      }

      // delete the leaf
      delete currentNode;
        currentNode = NULL;
    }
    // case: the node has only one child
    else if( ( ( currentNode->left != NULL ) &&
               ( currentNode->right == NULL )   ) ||
             ( ( currentNode->left == NULL ) &&
               ( currentNode->right != NULL )   ) )
    {
      // case: the node has no predecessor
      if( previousNode == NULL )
      {
        // the node must be the root, update accordingly
        // case: the only child is the left one
        if( ( currentNode->left != NULL ) &&
            ( currentNode->right == NULL ) )
        {
          // make the child the new root
          root = currentNode->left;
        }
        // case: the only child is the right one
        else
        {
          // make the child the new root
          root = currentNode->right;
        }
      }
      // case: there is a predecessor
      else
      {
        // re-link the tree appropriately
        // case: the only child is the left one
        if( currentNode->left != NULL )
        {
          // case: a left child is being deleted
          if( lastDirection == LEFT )
          {
            previousNode->left = currentNode->left;
          }
          // case: a right child is being deleted
          else
          {
            previousNode->right = currentNode->left;
          }
        }
        // case: the only child is the right one
        else
        {
          // case: a left child is being deleted
          if( lastDirection == LEFT )
          {
            previousNode->left = currentNode->right;
          }
          // case: a right child is being deleted
          else
          {
            previousNode->right = currentNode->right;
          }
        }
      }

      // delete the node
      delete currentNode;
        currentNode = NULL;
    }
    // case: the node has two children
    else
    {
      // find the node's "in-order" predecessor
      // start at the left child
      previousNode = currentNode;   // can be used for current purpose now, no
                                    // re-linking will occur
      inorderPredecessor = currentNode->left;

      // advance as far right as possible
      while( inorderPredecessor->right != NULL )
      {
        // keep track of the previous node for later performance enhancement
        previousNode = inorderPredecessor;

        // advance the predecessor pointer
        inorderPredecessor = inorderPredecessor->right;
      }

      // copy the data into the current node
      currentNode->dataItem = inorderPredecessor->dataItem;

      // remove the inorder predecessor node using code that was already written
      remove_sub( inorderPredecessor->dataItem.getKey(), inorderPredecessor,
                  previousNode );
    }

    // indicate that removal occurred
    removed = true;
  }
  // case: no such node was found
    // take no further action

  // return the removal flag
  return removed;
}
