#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "List.h"
#include "Heap.h"
#include "PCB.h"
#include "constants.h"
#include "my_stopwatch.h"


int readMetaData( Heap* heap, char* meta_data_file_name,
                  ConfigData* configurations );

int getTaskType(char* strings);

int parseNumCycles(char* action);

int readMetaData( Heap* heap, char* meta_data_file_name,
                  ConfigData* configurations )
{
    int scheduler_load_success = NO_ERRORS;
	char strings[STD_STR_LEN];
	char buffer[STD_STR_LEN];
    int incoming_pid = 1;

	//define all necessary objects and initilaize the structs
	Task_ListNode node = task_listNode_intialize();
	Task_List list;
        task_initialize( &list );
	PCB process;
        initialize_PCB ( &process );
    heap_init( heap );
	FILE *fp = NULL;

	//opening file
	// char* filename = malloc( 100 * sizeof(char) );
	fp = fopen( meta_data_file_name, "r");

	//omitting unnecessary initial text
	fscanf(fp, "%*s %*s %*s", strings);

	// read the first instruction
	fscanf(fp, "%s", strings);

    // case: the first instruction was to boot the system
	if( strcmp(strings, "S(start)0;") == 0 )
	{
		// read until the system shut down instruction is found or end of file
		while( fscanf(fp, "%s", strings) != EOF)
		{
			// if the instruction indicates the start of a process
			if(strcmp(strings, "A(start)0;") == 0)
			{
		        // read the next instruction
				fscanf(fp, "%s", strings);

				// read instructions until the process terminates
				while(strcmp(strings, "A(end)0;") != 0)
				{
					(node.task_number) = getTaskType(strings);
					//printf("This is the task_number: %d\n", (getTaskType(strings)));
					//printf("This is the task_number: %d\n", (node.task_number));
					if(strcmp(strings, "I(hard") == 0 || strcmp(strings, "O(hard") == 0)
					{
						fscanf(fp, "%s", buffer);
						strcat(strings, " ");
						strcat(strings, buffer);
					}

					// store the task name
					strcpy((node.task_name), strings);
					node.total_cycles = parseNumCycles(strings);
					insert_List_task (  &(list),  node );

					// read the next instruction
					fscanf(fp, "%s", strings);
				}
				if(strcmp(strings, "A(end)0;") == 0)
				{
                    // construct a new process (control block)
 					process = create_PCB ( incoming_pid, stopwatch( 'x', GLOBAL_CLOCK ), DEFAULT_PRIORITY, list );

					//insert the pcb into the heap
					insert_heap( heap,  process,
                                 configurations->cpu_scheduler );

                    // clear the list for the next process
                    task_initialize( &list );

                    // create a new process id for the next one
                    incoming_pid++;
				}
			}
		}
	}

	return scheduler_load_success;
}

int getTaskType(char* strings)
{
    int task_type = UNKOWN;

    // case: the function is a processing type function
	if(strings[0] == 'P')
	{
		task_type = PROCESS;
	}
    // case: the function is destined for an  output type of I/O
    else if( strings[0] == 'I' )
    {
        if( strings[2] == 'k')
    	{
	    	task_type = KEYBOARD;
    	}
    	else if(strings[0] == 'I' && strings[2] == 'h')
    	{
	    	task_type = HARDDRIVE_READ;
	    }   
    }
    else if( strings[0] == 'O' )
    {
	    if( strings[2] == 'h')
    	{
	    	task_type = HARDDRIVE_WRITE;
    	}
	    else if( strings[2] == 'm')
    	{
	    	task_type = MONITOR;
	    }
    	else if( strings[2] == 'p')
	    {
	    	task_type = PRINTER;
    	}
    }
	else if (strings[0] == 'A' && strings[2] == 'e')
	{
	    task_type = PROCESS_END;
	}

    // return the identified task type
    return task_type;
}

int parseNumCycles( char* action )
{
    // get a pointer for referencing the part of the string
    // that is a number
    char* num_start = action + strlen( action ) - 2;

    // move the pointer back until it points to the first number
    // in the string 
    while((num_start - 1) >= action && isdigit( *(num_start - 1) ) )
    {
        // move the pointer back
        num_start--;
    }

    // return the part of a string that is a number as number data
    return atoi( num_start );
}
