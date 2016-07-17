#ifndef __SCHEDULING_QUEUE_H__
#define __SCHEDULING_QUEUE_H__

#include "Constants.h"

#define DEFAULT_SCHEDULE_QUEUE_SIZE 64

// Treat this as a min SchedulingQueue

void swap(void** left, void** right) {
	void* temp;
	temp = *left;
	*left = *right;
	*right = temp;
}

typedef void* Item;

typedef struct {
	Item mItem;
	double mArrivalRank;
} ScheduledItem;


typedef int (*ScheduleCompareFunction)(ScheduledItem*, ScheduledItem*);

// default comparison function
int fifoCompare(ScheduledItem* pLeft, ScheduledItem* pRight) {
	return pLeft->mArrivalRank < pRight->mArrivalRank;
}
// NOTE: new compare functions should follow "is left < right" format


typedef void (*ItemDestructorFunction)(void*);

void doNothingDestructor(void* pIgnored) {
	// do literally nothing - application programmer should override
	// if necessary
}


typedef struct {
	ScheduledItem** mItems;
	int mSize;
	int mMaxSize;
	double mNextArrivalRank;   // double for more possibilities
	ScheduleCompareFunction mCompare;
	ItemDestructorFunction mDestructor;
} SchedulingQueue;



int isSqEmpty(SchedulingQueue* pSchedulingQueue);
int isSqNotEmpty(SchedulingQueue* pSchedulingQueue);


SchedulingQueue* SchedulingQueueFactory(
	ScheduleCompareFunction pCompareFunction,
	ItemDestructorFunction pDestructorFunction) {

	SchedulingQueue* newQueue =
		(SchedulingQueue*) malloc(sizeof(SchedulingQueue));
	newQueue->mItems =
		(ScheduledItem**) malloc(DEFAULT_SCHEDULE_QUEUE_SIZE *
								 sizeof(ScheduledItem*));
	newQueue->mMaxSize = DEFAULT_SCHEDULE_QUEUE_SIZE;
	newQueue->mSize = 0;
	newQueue->mNextArrivalRank = 0;
	newQueue->mCompare = pCompareFunction;
	newQueue->mDestructor = pDestructorFunction;

	return newQueue;
}


void resize(SchedulingQueue* pQueue) {
	int newSize = 2 * pQueue->mMaxSize;
	ScheduledItem** temp =
		(ScheduledItem**) malloc(newSize * sizeof(ScheduledItem*));

	int i;
	for (i = 0; i < pQueue->mSize; i++) {
		temp[i] = pQueue->mItems[i];
	}

	pQueue->mItems = temp;
	pQueue->mMaxSize = newSize;
}


void addSqItem(SchedulingQueue* pQueue, Item pNewItem) {
	if (pQueue->mSize >= pQueue->mMaxSize) {
		resize(pQueue);
	}

	ScheduledItem* newSchedItem =
		(ScheduledItem*) malloc(sizeof(ScheduledItem));
	newSchedItem->mItem = pNewItem;
	newSchedItem->mArrivalRank = pQueue->mNextArrivalRank++;

	pQueue->mItems[pQueue->mSize] = newSchedItem;

	int swapMade = true;
	int moverIndex = pQueue->mSize;
	while (swapMade  /*&& moverIndex > 0*/) {
		swapMade = false;
		ScheduledItem* mover = pQueue->mItems[moverIndex];
		ScheduledItem* parent = pQueue->mItems[moverIndex / 2];

		if (pQueue->mCompare(mover, parent)) {
			swap((void**) &parent, (void**) &mover);
			swapMade = true;
			moverIndex /= 2;
		}
	}

	pQueue->mSize++;
}


Item peekAtNextSqItem(SchedulingQueue* pQueue) {
	if (pQueue != NULL && isSqNotEmpty(pQueue)) {
		return pQueue->mItems[0]->mItem;
	} else {
		return NULL;
	}
}


Item popNextSqItem(SchedulingQueue* pQueue) {
	Item returnItem = NULL;

	if (isSqNotEmpty(pQueue)) {
		returnItem = pQueue->mItems[0]->mItem;

		int sinkerIndex  = 0;
		pQueue->mItems[sinkerIndex] = pQueue->mItems[pQueue->mSize - 1];
		pQueue->mSize--;

		int swapMade = false;
		if (pQueue->mSize > 0) {
			swapMade = true;
		}

		while (swapMade) {
			swapMade = false;
			int smallestIndex = sinkerIndex;
			int leftChildIndex = (sinkerIndex * 2) + 1;
			int rightChildIndex = leftChildIndex + 1;

			if (leftChildIndex < pQueue->mSize &&
				pQueue->mCompare(pQueue->mItems[leftChildIndex],
								 pQueue->mItems[smallestIndex])) {
				smallestIndex = leftChildIndex;
			}

			if (rightChildIndex < pQueue->mSize &&
				pQueue->mCompare(pQueue->mItems[rightChildIndex],
								 pQueue->mItems[smallestIndex])) {
				smallestIndex = rightChildIndex;
			}

			if (sinkerIndex != smallestIndex) {
				swap((void**) &(pQueue->mItems[sinkerIndex]),
					 (void**) &(pQueue->mItems[smallestIndex]));
				swapMade = true;
				sinkerIndex = smallestIndex;
			}
		}
	}

	return returnItem;
}


void clear(SchedulingQueue* pQueue) {
	return;
}


int isSqEmpty(SchedulingQueue* pQueue) {
	return pQueue->mSize <= 0;
}


int isSqNotEmpty(SchedulingQueue* pQueue) {
	return !isSqEmpty(pQueue);
}


void printArrivalTimes(SchedulingQueue* pQueue, FILE* pStream) {
	if (isSqEmpty(pQueue)) {
		fprintf(pStream, "The scheduling queue is empty\n");
	} else {
		int i;
		for (i = 0; i < pQueue->mSize; i++) {
			fprintf(pStream, "%.0f\t", pQueue->mItems[i]->mArrivalRank);
		}
		fprintf(pStream, "\n");
	}
}


void destroySchedulingQueue(SchedulingQueue* pQueue) {
	if (pQueue != NULL) {
		int i;
		for (i = 0; i < pQueue->mSize; i++) {
			ScheduledItem** scheduledItem = &(pQueue->mItems[i]);
			if (*scheduledItem != NULL) {
				pQueue->mDestructor((*scheduledItem)->mItem);
				free(*scheduledItem);
			}
		}

		free(pQueue->mItems);

		free(pQueue);
		pQueue = NULL;
	}
}

#endif    //__SCHEDULING_QUEUE_H__