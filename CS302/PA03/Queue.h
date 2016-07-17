//--------------------------------------------------------------------
//
//  Laboratory 7                                               Queue.h
// 
//  Class declaration of the abstract class interface to be used as
//  the basis for implementations of the Queue ADT.
//
//--------------------------------------------------------------------

#ifndef QUEUE_H
#define QUEUE_H

#include <stdexcept>
#include <iostream>

using namespace std;

//--------------------------------------------------------------------

template <typename DataType>
class Queue {
  public:
    static const int MAX_QUEUE_SIZE = 8;

    virtual ~Queue();

    virtual void enqueue(const DataType& newDataItem) throw (logic_error) = 0;
    virtual DataType dequeue() throw (logic_error) = 0;

    virtual void clear() = 0;

    virtual bool isEmpty() const = 0;
    virtual bool isFull() const = 0;

    // The conditional compilation tests below are very important. 
    // Because the functions declared are pure virtual functions, if they 
    // are declared in the base class, then they MUST be implemented in any 
    // derived classes. But they are supposed to be optional implementations.
    // Consequently, they must only be declared here if they are being 
    // implemented in the derived classes.
#if LAB7_TEST2
    virtual void putFront(const DataType& newDataItem) throw (logic_error) = 0;
    virtual DataType getRear() throw (logic_error) = 0;
#endif
#if LAB7_TEST3
    virtual int getLength() const = 0;
#endif

    virtual void showStructure() const = 0;
};

template <typename DataType>
Queue<DataType>::~Queue() 
// Not worth having a separate class implementation file for the destuctor
{}

#endif		// #ifndef QUEUE_H
