#ifndef __PCB_H__
#define __PCB_H__

#include "ArrayList.h"

typedef struct {
	char mTaskName[STD_STR_LEN];
	int mTaskType;
	int mTotalCycles;
	int mRemainingCycles;
} Task;


typedef struct {
	ArrayList* mTasks;
	int mCurrentTaskIndex;
	int mRemainingProcessTime;
} TaskList;


typedef struct {
	double mPid;
	int mTimeCreated;
	int mPriority;
	int mTotalProcessTime;
	TaskList* mTaskList;
} Pcb;


void printTask(Task* pTask, FILE* pStream);


Pcb* PcbFactory(double pPid, int pTimeCreated, int pPrioriy, ArrayList* pTasks) {
	Pcb* newPcb = (Pcb*) malloc(sizeof(Pcb));

	newPcb->mPid = pPid;
	newPcb->mTimeCreated = pTimeCreated;
	newPcb->mPriority = pPrioriy;
	newPcb->mTotalProcessTime = 0;

	TaskList* newList = (TaskList*) malloc(sizeof(TaskList));
	newList->mCurrentTaskIndex = 0;
	newList->mTasks = pTasks;
	// newList->mRemainingProcessTime = sumTaskDurations(pTasks);
	newPcb->mTaskList = newList;

	return newPcb;
}


void destroyPcb(Pcb* pPcb) {
	if (pPcb != NULL) {
		if (pPcb->mTaskList != NULL) {
			// destroyList(pPcb->mTaskList->mTasks);
		}

		free(pPcb);
		pPcb = NULL;
	}
}


void destroyPcbInSchedulingQueue(void* pPcb) {
	// TODO: find a way to do type checking?
	destroyPcb((Pcb*) pPcb);
}


Task* getCurrentTask(Pcb* pProcess) {
	return (Task*) getArrayListItem(pProcess->mTaskList->mTasks,
						   pProcess->mTaskList->mCurrentTaskIndex);
}


int advanceToNextProcessTask(Pcb* pProcess) {
	pProcess->mTaskList->mCurrentTaskIndex++;

	if (pProcess->mTaskList->mCurrentTaskIndex <
		getArrayListSize(pProcess->mTaskList->mTasks)) {
		return NO_ERRORS;
	} else {
		return ERROR;
	}
}


int processHasFinished(Pcb* pProcess) {
	return (pProcess->mTaskList->mCurrentTaskIndex >=
		    getArrayListSize(pProcess->mTaskList->mTasks));
}


int processHasNotFinished(Pcb* pProcess) {
	return !processHasFinished(pProcess);
}


int getPriority(Pcb* pProcess) {
	return pProcess->mPriority;
}


void printPcb(Pcb* pPcb, FILE* pStream) {

	fprintf(pStream, "----------------------------\n");
	if (pPcb != NULL) {
		fprintf (pStream, "Pid:                %u\n", (unsigned int) pPcb->mPid);
		fprintf (pStream, "Time Created:       %i\n", pPcb->mTimeCreated);
		fprintf (pStream, "PCB priority value: %i\n", pPcb->mPriority);
		fprintf (pStream, "Total process time: %i\n",
				 pPcb->mTaskList->mRemainingProcessTime);
		fprintf(pStream, "~~~~~~~\n");
		
		int i;
		for (i = 0; i < getArrayListSize(pPcb->mTaskList->mTasks); i++) {
			Task* task = (Task*) getArrayListItem(pPcb->mTaskList->mTasks, i);
			printTask(task, pStream);
		}
	} else {
		fprintf(pStream, "Pcb does not exist.\n");
	}

	fprintf(pStream, "----------------------------\n");
}


void printTask(Task* pTask, FILE* pStream) {
	if (pTask != NULL) {
		fprintf(pStream, "Name:             %s\n", pTask->mTaskName);
		fprintf(pStream, "Type:             %i\n", pTask->mTaskType);
		fprintf(pStream, "Total Cycles:     %i\n", pTask->mTotalCycles);
		fprintf(pStream, "Remaining Cycles: %i\n", pTask->mRemainingCycles);
	}
}

#endif  // define __PCB_H__