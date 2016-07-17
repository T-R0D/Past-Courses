#ifndef __ARRAYLIST_H__
#define __ARRAYLIST_H__

// TODO: implement delete and associated methods
                                 
#include "Constants.h"

#define INVALID_POS -1
#define DEFAULT_LIST_SIZE 4

typedef struct {
	void** mItems;
	int mHead;
	int mTail;
	int mCursor;
	int mMaxSize;
} ArrayList;

ArrayList* ArrayListFactory() {
	ArrayList* newList = (ArrayList*) malloc(sizeof(ArrayList));

	newList->mMaxSize = DEFAULT_LIST_SIZE;
	newList->mItems = (void*) malloc(DEFAULT_LIST_SIZE * sizeof(void*));
	newList->mHead = INVALID_POS;
	newList->mTail = INVALID_POS;
	newList->mCursor = INVALID_POS;

	return newList;
}


int getArrayListSize(ArrayList* pList);
int isArrayListEmpty(ArrayList* pList);
int isArrayListNotEmpty(ArrayList* pList);
void* getArrayListItem(ArrayList* pList, int pIndex);


void destroyArrayList(ArrayList* pList) {
	if (pList != NULL) {
		if (pList->mItems != NULL) {
			int i;
			for (i = 0; i < getArrayListSize(pList); i++) {
				// make use of a destructor
			}

			free(pList->mItems);
		}

		free(pList);
		pList = NULL;
	}
}


void* getCursoredItem(ArrayList* pList) {
	return pList->mItems[pList->mCursor];
}


int advanceCursor(ArrayList* pList) {
	if (pList->mCursor < pList->mTail) {
		pList->mCursor++;
		return true;
	} else if (pList->mTail < pList->mHead) {
		pList->mCursor++;
		if (pList->mCursor >= pList->mMaxSize) {
			pList->mCursor = 0;
		}
		return true;
	} else {
		return false;
	}
}


int rewindCursor(ArrayList* pList) {
	if (pList->mHead < pList->mCursor) {
		pList->mCursor--;
		return true;
	} else if (pList->mHead < pList->mTail) {
		pList->mCursor--;
		if (pList->mCursor < 0) {
			pList->mCursor = pList->mMaxSize - 1;
		}
	}

	return 0; // TODO
}

#include "Pcb.h"
#include <stdio.h>
#include <string.h>


void* getArrayListItem(ArrayList* pList, int pIndex) {
	void* item;

bug

	if (pList->mHead == INVALID_POS ||
		pIndex >= getArrayListSize(pList)) {
		item = NULL;
	} else if (pIndex + pList->mHead < pList->mTail) {
		item = pList->mItems[pIndex + pList->mHead];
	} else if () {

	} else {
		item = NULL;
	}



	if (pList->mHead == INVALID_POS ||
		pIndex >= getArrayListSize(pList)) {
		item = NULL;

bug

	} else if (pIndex + pList->mHead >= pList->mMaxSize) {
		item = pList->mItems[pIndex + pList->mHead - pList->mMaxSize];
	} else {

bug

		item = pList->mItems[pIndex + pList->mHead];
	}

printTask((Task*) item, stdout);

	return item;
}


void resizeArrayList(ArrayList* pList) {
	int newSize = 2 * getArrayListSize(pList);
	void** temp = (void*) malloc(newSize * sizeof(void*));
	pList->mMaxSize = newSize;

	int i;
	for (i = 0; i < getArrayListSize(pList); i++) {
		temp[i] = getArrayListItem(pList, i);
	}

	pList->mTail = getArrayListSize(pList) - 1;
	pList->mHead = 0;
	free(pList->mItems);
	pList->mItems = temp;
}


int getArrayListSize(ArrayList* pList) {
	if (pList->mHead == INVALID_POS) {
		return 0;
	} else if(pList->mHead <= pList->mTail) {
		return (pList->mTail - pList->mHead + 1);
	} else {
		return pList->mTail + pList->mMaxSize - pList->mHead + 1;
	}
}


int isArrayListEmpty(ArrayList* pList) {
	return getArrayListSize(pList) == 0;
}


int isArrayListNotEmpty(ArrayList* pList) {
	return getArrayListSize(pList) != 0;
}


void addArrayListItem(ArrayList* pList, void* pItem) {
	if (getArrayListSize(pList) >= pList->mMaxSize) {
		resizeArrayList(pList);
	}

	if (pList->mHead == INVALID_POS) {
		pList->mHead = 0;
		pList->mCursor = 0;
		pList->mTail = -1;  // see next code block
	}

	pList->mTail++;
	if (pList->mTail >= pList->mMaxSize) {
		pList->mTail = 0;
	}

	pList->mItems[pList->mTail] = pItem;
}




#endif  // ARRAYLIST_H
