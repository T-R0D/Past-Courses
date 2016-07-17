#ifndef _META_DATA_READIN_H
#define _META_DATA_READIN_H

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include "List.h"
#include "Heap.h"
#include "PCB.h"
#include "readin_config.h"
#include "my_stopwatch.h"

void meta_data_readin()
{
	char strings[STD_STR_LEN];
	char buffer[STD_STR_LEN];
	size_t buffersize = 100;

	//define all necessary objects and initilaize the structs
	Task_ListNode node = task_listNode_intialize();
	Task_List list = task_initialize ();
	PCB process = initialize_PCB ();
	Heap heap = heap_init();
	config value;

	FILE *fp = NULL;
	printf("Enter the meta-data file name to use: ");

	//opening file
	char* filename = malloc(100*sizeof(char));
	scanf("%s", filename);
	fp = fopen(filename, "r");

	//omitting unnecessary initial text and reading in S(start)0;
	fscanf(fp, "%*s %*s %*s %s", strings);
	//if system has started
	if(strcmp(strings, "S(start)0;") == 0)
	{

		// read until the system shut down instruction is found
		while(fscanf(fp, "%s", strings) != EOF)
		{
			printf( "MADE IT INSIDE OUTER LOOP, %d\n", __LINE__ );
			fflush( stdout );

			// if the instruction indicates the start of a process
			if(strcmp(strings, "A(start)0;") == 0)
			{
				printf( "PROCESS STARTED, %d\n", __LINE__ );
				fflush( stdout );

				// read instructions until the process terminates
				while(strcmp(strings, "A(end)0;") != 0)
				{

					// read the instruction
					fscanf(fp, "%s", strings);
					//copy to node member
					(node.task_number) = getTaskType(strings);

					printf("This is the task_number: %d\n", (node.task_number));

					//concatenate in case of "hard drive" read in
					if(strcmp(strings, "I(hard") == 0 || strcmp(strings, "O(hard") == 0)
					{
						fscanf(fp, "%s", buffer);
						strcat(strings, " ");
						strcat(strings, buffer);
					}

					// store the task name in node member
					strcpy((node.task_name), strings);
					printf("This is the task name: %s  " "On line: %d \n", node.task_name, __LINE__);
					//store num of cycles in node member
					node.total_cycles = parseNumCycles(strings);
					printf("total number of cycles: %d \n", node.total_cycles );
					//inserting node into task list
					insert_List_task (&(list), node);
				}
				if(strcmp(strings, "A(end)0;") == 0)
				{
/*
parameter 2 = arrival need to enter live time talk with terence.
*/
 					process = create_PCB ( process.pid, stopwatch('x', GLOBAL_CLOCK), 0, list );

					//insert the pcb into the heap
					insert_heap( &heap,  process, value.cpu_scheduler);

					//debug print the heap 
					printHeap( &heap);
					//increment pid for next process
					process.pid++;
				}
			}
		}
	}

	printf( "MADE IT OUT!, %d\n", __LINE__ );
	fflush( stdout );
	
	free(filename);
	fclose(fp);
}

//function to return task number
int getTaskType(char* strings)
{
	if(strings[0] == 'P')
	{
		return PROCESS;
	}
	else if(strings[0] == 'I' && strings[2] == 'k')
	{
		return KEYBOARD;
	}
	else if(strings[0] == 'I' && strings[2] == 'h')
	{
		return HARDDRIVE_READ;
	}
	else if(strings[0] == 'O' && strings[2] == 'h')
	{
		return HARDDRIVE_WRITE;
	}
	else if(strings[0] == 'O' && strings[2] == 'm')
	{
		return MONITOR;
	}
	else if(strings[0] == 'O' && strings[2] == 'p')
	{
		return PRINTER;
	}
	else if(strings[0] == 'A' && strings[2] == 'e')
	{
		return PROCESS_END;
	}
}

int parseNumCycles( char* action )
{
    // get a pointer for referencing the part of the string
    // that is a number
    char* num_start = action + strlen( action ) - 2;

    // move the pointer back until it points to the first number
    // in the string 
    while((num_start - 1) >= action && isdigit( *(num_start - 1)) )
    {
        // move the pointer back
        num_start--;
    }

    // return the part of a string that is a number as number data
    return atoi( num_start );
}

#endif
