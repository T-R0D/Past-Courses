// QueueArray.h

#ifndef QUEUEARRAY_H
#define QUEUEARRAY_H

#include <stdexcept>
#include <iostream>

using namespace std;

#include "Queue.h"

template <typename DataType>
class QueueArray : public Queue<DataType> {
  public:
    QueueArray(int maxNumber = Queue<DataType>::MAX_QUEUE_SIZE);
    QueueArray(const QueueArray& other);
    QueueArray& operator=(const QueueArray& other);
    ~QueueArray();

    void enqueue(const DataType& newDataItem) throw (logic_error);
    DataType dequeue() throw (logic_error);

    void clear();

    bool isEmpty() const;
    bool isFull() const;

    void putFront(const DataType& newDataItem) throw (logic_error);
    DataType getRear() throw (logic_error);
    int getLength() const;

    void showStructure() const;

  private:
    int maxSize;
    int front;
    int back;
    DataType* dataItems;
};

#endif
