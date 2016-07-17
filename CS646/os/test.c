#include <stdlib.h>
#include <stdio.h>
#include "SchedulingQueue.h"
#include "ArrayList.h"

int main() {
	char buffer[100];

	FILE* fp = fopen("meta_data", "r");

	int i;
	for(i = 0; i < 20; i++) {
		fscanf(fp, " %[^;.];", buffer);
		printf("%i. %s\n", i , buffer);


		if (strcmp("S(end)0", buffer) == 0) {
			puts("donesies!");
			break;
		}
	}

	return 0;
}


// void printListItems(ArrayList* pList) {
// 	int i;
// 	for (i = 0; i < getArrayListSize(pList); i++) {
// 		printf("%d\t", *((int*) getArrayListItem(pList, i)));
// 	}
// 	printf("\n");
// }

// int main() {

// ArrayList* list = ArrayListFactory();
// printf("size of list: %d\n", getArrayListSize(list));
// printListItems(list);

// int i;
// int array[10];
// for (i = 0; i < 3; i++) {
// 	array[i] = i;
// 	addArrayListItem(list, (void*) &(array[i]));
// }
// printf("size of list: %d\n", getArrayListSize(list));
// printListItems(list);
// for (i = 3; i < 10; i++) {
// 	array[i] = i;
// 	addArrayListItem(list, (void*) &(array[i]));
// }
// printf("size of list: %d\n", getArrayListSize(list));
// printListItems(list);

// destroyArrayList(list);

// return 0;

// 	puts("brand new queue:");
// 	SchedulingQueue* queue = schedulingQueueFactory(fifoCompare, doNothingDestructor);
// 	printArrivalTimes(queue, stdout);
// 	puts("");

// 	puts("adding some ints to schedule:");
// 	int seven = 7;
// 	int eight = 8;
// 	int nine = 9;
// 	int ten = 10;
// 	addSqItem(queue, &seven);
// 	addSqItem(queue, &eight);
// 	addSqItem(queue, &nine);
// 	addSqItem(queue, &ten);
// 	int i;
// 	for (i = 0; i < 3; i++) {
// 		addSqItem(queue, &i);
// 	}
// 	printArrivalTimes(queue, stdout);
// 	puts("");

// 	puts("getting items from the scheduler:");
// 	for (i = 0; i < 3; i++) {
// 		seven = *((int*) popNextSqItem(queue)); 
// 		printf("got: %d\n", seven);
// 		printArrivalTimes(queue, stdout);
// 	}
// 	puts("");


// 	for (i = 0; i < 3; i++) {
// 		addSqItem(queue, &i);
// 	}
// 	printArrivalTimes(queue, stdout);
// 	puts("");


// 		puts("getting items from the scheduler:");
// 	for (i = 0; i < 3; i++) {
// 		seven = *((int*) popNextSqItem(queue)); 
// 		printf("got: %d\n", seven);
// 		printArrivalTimes(queue, stdout);
// 	}
// 	puts("");


// 	destroySchedulingQueue(queue);

// 	return 0;
// }

