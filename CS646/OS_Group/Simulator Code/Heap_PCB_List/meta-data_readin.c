#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include "List.h"
#include "Heap.h"
#include "PCB.h"
#include "preprocessor_directives.h"

//global variables

int getTaskType(char* strings);
int parseNumCycles(char* action);


int main( int argc, char** argv )
{

	printf( "MADE IT HERE, %d\n", __LINE__ );
	fflush( stdout );

	char strings[STD_STR_LEN];
	char buffer[STD_STR_LEN];
	size_t buffersize = 100;

	//define all necessary objects and initilaize the structs
	Task_ListNode node = task_listNode_intialize();
	Task_List list = task_initialize ();
	PCB process = initialize_PCB ();
	Heap heap = heap_init();

	FILE *fp = NULL;
	printf("Enter the file name to use: ");

	//opening file
	char* filename = malloc(100*sizeof(char));
	scanf("%s", filename);
	fp = fopen(filename, "r");

	//omitting unnecessary initial text
	fscanf(fp, "%*s %*s %*s %s", strings);

	if(strcmp(strings, "S(start)0;") == 0)
	{
		printf( "MADE IT HERE, %d\n", __LINE__ );
		fflush( stdout );

		// read until the system shut down instruction is found
		while(fscanf(fp, "%s", strings) != EOF)
		{
			printf( "MADE IT INSIDE OUTER LOOP, %d\n", __LINE__ );
			fflush( stdout );

			// read the next instruction
			//fscanf(fp, "%s", strings);

			printf( "Read in %s on line %d\n", strings, __LINE__ );
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

					printf( "Read in %s on line %d\n", strings, __LINE__ );
					fflush( stdout );

					(node.task_number) = getTaskType(strings);
					printf("This is the task_number: %d\n", (getTaskType(strings)));
					printf("This is the task_number: %d\n", (node.task_number));
					if(strcmp(strings, "I(hard") == 0 || strcmp(strings, "O(hard") == 0)
					{
						printf( "Read in %s on line %d\n", strings, __LINE__ );
						fflush( stdout );
						fscanf(fp, "%s", buffer);
						strcat(strings, " ");
						strcat(strings, buffer);
					}

					printf( "Read in %s on line %d\n", strings, __LINE__ );
					fflush( stdout );


					// store the task name
					strcpy((node.task_name), strings);
					
					printf( "Read in %s on line %d\n", node.task_name, __LINE__ );
					fflush( stdout );
					printf("total numb cycles: %d \n", parseNumCycles(strings) );
					node.total_cycles = parseNumCycles(strings);
					printf("total numb cycles: %d \n", node.total_cycles );
					insert_List_task (  &(list),  node );
					//debug print list
//					print_list (  &list);
					
				}
				if(strcmp(strings, "A(end)0;") == 0)
				{
/*
parameter 1 = pid, speak with terence about what approach
parameter 2 = arrival need to enter live time talk with terence.
*/
 					process = create_PCB ( 0, 0, 0, list );
					//debug, print PCB
//					print_PCB (  &process );

/*

paramter 3 is the schedular type, RR, FIFO, fig out what from config and throw into the function	
*/
					//insert the pcb into the heap
					insert_heap( &heap,  process, 0);

					//debug print the heap 
					printHeap( &heap);				
				}
			}
		}
	}

	printf( "MADE IT OUT!, %d\n", __LINE__ );
	fflush( stdout );
	
	free(filename);
	return 0;
}

int getTaskType(char* strings)
{
	if(strings[0] == 'P')
	{
		return PROCESS;
	}
	if(strings[0] == 'I' && strings[2] == 'k')
	{
		return KEYBOARD;
	}
	if(strings[0] == 'I' && strings[2] == 'h')
	{
		return HARDDRIVE_READ;
	}
	if(strings[0] == 'O' && strings[2] == 'h')
	{
		return HARDDRIVE_WRITE;
	}
	if(strings[0] == 'O' && strings[2] == 'm')
	{
		return MONITOR;
	}
	if(strings[0] == 'O' && strings[2] == 'p')
	{
		return PRINTER;
	}
	if(strings[0] == 'A' && strings[2] == 'e')
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
    while((num_start - 1) >= action && isdigit( *(num_start - 1) ) )
    {
        // move the pointer back
        num_start--;
    }

    // return the part of a string that is a number as number data
    return atoi( num_start );
}
