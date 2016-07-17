/**
    @file Heap.h

    @author Terence Henriod

    Lab 10: Heap

    @brief Class declaration for the Heap implementation of the Priority Queue
           ADT -- inherits the array implementation of the Heap ADT

    @version Original Code 1.00 (11/8/2013) - T. Henriod
*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   PRECOMPILER DIRECTIVES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#ifndef PRIORITYQUEUE_H
#define PRIORITYQUEUE_H


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#include <stdexcept>
#include <iostream>
#include "Heap.cpp"
using namespace std;


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   GLOBAL CONSTANTS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
const int defMaxQueueSize = 10;


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
================================================================================
                   CLASS DEFINITION(S)
================================================================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
@class PriorityQueue

The Priority Queue ADT. Inherits an array based Heap ADT in order to provide
a priority first functionality. This class is really just a practice for
inheritance and provides new function names to mask the Heap's functionality.
*/
template < typename DataType, typename KeyType = int,
           typename Comparator = Less< KeyType > >
class PriorityQueue : public Heap< DataType, KeyType, Comparator >
{
 public:
 /*---   Constructor(s) / Destructor   ---*/
  PriorityQueue ( int maxNumber = defMaxQueueSize );
  PriorityQueue( const Heap<DataType, KeyType, Comparator>& other );

 /*---   Mutators   ---*/
  void enqueue ( const DataType &newDataItem );
  DataType dequeue ();
};


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   TERMINATING PRECOMPILER DIRECTIVES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#endif   // #ifndef PRIORITYQUEUE_H
