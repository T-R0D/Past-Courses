#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "ArrayList.h"
#include "SchedulingQueue.h"
#include "Pcb.h"
#include "ActivityLog.h"
#include "Constants.h"
#include "MyStopwatch.h"

static double sgNextPid = 1;

int readMetaData(SchedulingQueue* pSchedulingQueue, char* pMetaDataFilename, ActivityLog* pLog);

int getTaskType(char* pTaskString);

int parseNumCycles(char* action);

int readMetaData(SchedulingQueue* pSchedulingQueue, char* pMetaDataFilename, ActivityLog* pLog) {
    int schedulerLoadStatus = NO_ERRORS;
    int goodDataSoFar = true;
    char* buffer = (char*) malloc(BIG_STR_LEN);

    FILE* dataFile = fopen(pMetaDataFilename, "r");

    if (dataFile != NULL) {
    	// eat the garbage at the top of the file
    	fscanf(dataFile, "%[^\n]\n\n", buffer);

    	fscanf(dataFile, " %[^;];", buffer);

    	if (strcmp("S(start)0", buffer) == 0) {
    		while (goodDataSoFar && strcmp("S(end)0", buffer) != 0) {

    			fscanf(dataFile, " %[^;.];", buffer);

    			if (strcmp("A(start)0", buffer) == 0) {
    				ArrayList* taskList = ArrayListFactory();

    				// prime read loop with an instruction
    				fscanf(dataFile, " %[^;.];", buffer);

    				while (goodDataSoFar && strcmp("A(end)0", buffer) != 0) {
    					Task* newTask = (Task*) malloc(sizeof(Task));
    					int taskType = getTaskType(buffer);

    					if (taskType != UNKNOWN) {
    						strcpy(newTask->mTaskName, buffer);
    						newTask->mTaskType = taskType;
    						newTask->mTotalCycles = parseNumCycles(buffer);
    						newTask->mRemainingCycles = newTask->mTotalCycles;

    						addArrayListItem(taskList, (void*) newTask);

    						fscanf(dataFile, " %[^;.];", buffer);
    					} else {
    						goodDataSoFar = false;
    					}
    				}

    				Pcb* newProcess = PcbFactory(
    					sgNextPid++,
    					stopwatch('x', GLOBAL_CLOCK),
    					DEFAULT_PRIORITY,
    					taskList
    				);

    				addSqItem(pSchedulingQueue, (Item) newProcess);
                    logEvent(pLog,
                        newProcess->mPid,
                        "Entered system (global system time) ~~~~~~~~~~~~~",
                        stopwatch('x', GLOBAL_CLOCK)
                    );

    			} else if (strcmp("S(end)0", buffer) != 0) {//TODO:makelesshacky
    				goodDataSoFar = false;
    			}
    		}

    		if (goodDataSoFar == false) {
    			schedulerLoadStatus = ERROR;
    		}

    	} else {
    		schedulerLoadStatus = ERROR;
    	}

    } else {
    	schedulerLoadStatus = BAD_FILE_ERROR;
    }

    free(buffer);

	return schedulerLoadStatus;
}

int getTaskType(char* pTaskString)
{
    int taskType = UNKNOWN;

	if(pTaskString[0] == 'P') {
		taskType = PROCESS;
	} else if( pTaskString[0] == 'I' ) {
        if( pTaskString[2] == 'k') {
	    	taskType = KEYBOARD;
    	} else if(pTaskString[0] == 'I' && pTaskString[2] == 'h') {
	    	taskType = HARDDRIVE_READ;
	    }   
    } else if( pTaskString[0] == 'O' ) {
	    if( pTaskString[2] == 'h') {
	    	taskType = HARDDRIVE_WRITE;
    	} else if( pTaskString[2] == 'm') {
	    	taskType = MONITOR;
	    } else if( pTaskString[2] == 'p') {
	    	taskType = PRINTER;
    	}
    } else if (pTaskString[0] == 'A' && pTaskString[2] == 'e') {
	    taskType = PROCESS_END;
	}

    return taskType;
}


int parseNumCycles(char* pAction) {
	char* numStart = null;

	numStart = strstr(pAction, ")");

	if (numStart != NULL) {
		numStart++;
		
		if (isdigit(*numStart)) {
    		return atoi(numStart);
		}
	}

	return ERROR_NUM;
}
