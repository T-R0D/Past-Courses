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
	newList->mCursor = 0;

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
	return 0;
}


int rewindCursor(ArrayList* pList) {
	return 0; // TODO
}


void* getArrayListItem(ArrayList* pList, int pIndex) {
	if (pIndex < pList->mCursor) {
		return pList->mItems[pIndex];
	} else {
		return NULL;
	}
}


void resizeArrayList(ArrayList* pList) {
	int newSize = 2 * getArrayListSize(pList);
	void** temp = (void*) malloc(newSize * sizeof(void*));

	int i;
	for (i = 0; i < pList->mMaxSize; i++) {
		temp[i] = getArrayListItem(pList, i);
	}

	free(pList->mItems);
	pList->mMaxSize = newSize;
	pList->mItems = temp;
}


int getArrayListSize(ArrayList* pList) {
	return pList->mCursor;
}


int isArrayListEmpty(ArrayList* pList) {
	return getArrayListSize(pList) == 0;
}


int isArrayListNotEmpty(ArrayList* pList) {
	return getArrayListSize(pList) != 0;
}


void addArrayListItem(ArrayList* pList, void* pItem) {
	if (pList->mCursor >= pList->mMaxSize) {
		resizeArrayList(pList);
	}

	pList->mItems[pList->mCursor] = pItem;

	pList->mCursor++;
}




#endif  // ARRAYLIST_H
