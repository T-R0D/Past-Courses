//--------------------------------------------------------------------
//
//  Laboratory 5                                          ListLinked.h
//
//  Class declaration for the linked implementation of the List ADT
//
//--------------------------------------------------------------------

#ifndef LISTLINKED_H
#define LISTLINKED_H

#include <stdexcept>
#include <iostream>

using namespace std;

template <typename DataType>
class List {
  public:
    List(int ignored = 0);
    List(const List& other);
    List& operator=(const List& other);
    ~List();

    void insert(const DataType& newDataItem) throw (logic_error);
    void remove() throw (logic_error);
    void replace(const DataType& newDataItem) throw (logic_error);
    void clear();

    bool isEmpty() const;
    bool isFull() const;

    void gotoBeginning() throw (logic_error);
    void gotoEnd() throw (logic_error);
    bool gotoNext() throw (logic_error);
    bool gotoPrior() throw (logic_error);

    DataType getCursor() const throw (logic_error);

    // Programming exercise 2
    void moveToBeginning () throw (logic_error);

    // Programming exercise 3
    void insertBefore(const DataType& newDataItem) throw (logic_error);
    
    void showStructure() const;

  private:
    class ListNode {
      public:
	ListNode(const DataType& nodeData, ListNode* nextPtr);

	DataType dataItem;
	ListNode* next;
    };

    ListNode* head;
    ListNode* cursor;

};

#endif
