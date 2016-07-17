#ifndef HEAP_H
#define HEAP_H
#include <stdlib.h>
#include <stdio.h>
#include "PCB.h"
#define HEAP_SIZE 12

//declare the heap struct
typedef   struct{

	PCB dataItems [HEAP_SIZE]; //stores an array of PCBs
	int current_size;			//contains the current heap size
	int iterator;				//contains the location of current pcb ( dataItems)
}Heap;


///////////////////function specifications//////////////////////////////////////
void heap_init( Heap* temp );
void printHeap( Heap *h);
PCB remove_PCB_from_Heap ( Heap* h );
int update_cursor (  Heap *h );
int insert_heap( Heap *h,  PCB item, int scheduler);
void heapify( Heap* h, int method);
void clear_heap (); 
int is_Heap_full ();
int is_Heap_empty();

//////////////////function implementations//////////////////////////////////////
int is_Heap_empty( Heap* h)
{
	int value = 0;

	if ( h -> current_size <= 0)
	{
		value = 1;
	}

	return value;
}


int is_Heap_full ( Heap* h)
{
	int value = 0;

	if ( h -> current_size == HEAP_SIZE )
	{
		value = 1;
	}

	return value;
}
//intializes heap
void heap_init( Heap* temp )
{
	temp->current_size = 0;
	temp->iterator = 0;
}

//sort the heap ure (Per Professor Leverington don't 
//need to heap sort for SJF, STRF or Priority based
void heapify( Heap* h, int method)
{

	//if method is priority based
	if ( method == 2)
	{
		/*didnt implement the code per Professor Leverington said didn't have to
			implement just need stubs to expand if OS needed to handle those jobs
		*/
	}

	//if the method is SFJ
	else if ( method == 3)
	{
		/*didnt implement the code per Professor Leverington said didn't have to
			implement just need stubs to expand if OS needed to handle those jobs
		*/

	}
	//if the method is STRF
	else if ( method == 4 )
	{
		/*didnt implement the code per Professor Leverington said didn't have to
			implement just need stubs to expand if OS needed to handle those jobs
		*/
	} 

	//else must be round robin or FIFO so dont need to sort	

}

//inserts new  PCB into the heap
int insert_heap( Heap *h,  PCB item, int scheduler)
{
	int value = 0;

	if ( is_Heap_full ( h ) == 0 )
	{
		h -> dataItems[ h -> current_size] = item;

		h->current_size ++;

		//if the job scheduling is based on RR or FIFO don't nee to sort at insert
		
		//if the job scheduling is based on priority sort PCBs based on priority
		if ( scheduler == 2 )
		{
			heapify( h, 2);				
		}

		//else if the job scheduling is based on SJF sort PCBs based on SJF
		else if ( scheduler == 3 )
		{
			heapify( h, 3);	
		}

		//else the job scheculing is based on STRF sort PCBs based on STRF
		else
		{
			heapify( h, 4);	
		}	
		
		value = 1;
	}

	else 
	{
		puts ("Full Heap! Please free some memory before inserting!");

	}

	return value;
}

//update the cursor position ( allows us to navigate the heap)
int update_cursor (  Heap *h )
{
	if ( ( is_Heap_empty( h ) ) == 0 )
	{
		//if the iterator is between the front and end move forward and return pos
		if ( h -> iterator < ( h-> current_size) )
		{
			( h -> iterator ) ++;
		}

		//if the cursor reaches the end but the heap isn't empty then wrap around
		else if ( (h -> iterator) == ( h -> current_size) )
		{
			( h -> iterator ) = 0;
		}
	}

	else 
	{
		puts( "Empty Heap!");
	}
	
	return (h -> iterator);
}

PCB remove_PCB_from_Heap ( Heap* h )
{
    PCB temp;
	int j;

	if ( is_Heap_empty( h) == 1 )
	{
		puts("Empty Heap!");
	}
	else
	{
        // get the PCB that is about to be removed
        temp = h -> dataItems [ 0 ];

		//perculate all other PCBs UP
		for ( j = 0; j < h -> current_size - 1; ++j)
		{			
			h -> dataItems [ j ] = h -> dataItems [ j + 1];
		}

        h->current_size--;
	}

    // return the recently removed item
    return temp;
}

//print the heap out so can see data ure
void printHeap( Heap *h)
{

	int i;

	//if the heap is empty print empty heap message
	if ( is_Heap_empty(h) == 1 )
	{
		puts( "Empty Heap" );

	}

	//otherwise print the heap out
	else 
	{
		printf ("Printing Heap: \n ");

		printf ("PCBs in the Heap with jobs: \n");

		for ( i = 0 ; i < h-> current_size; ++i )
		{
			print_PCB ( & ( h-> dataItems [ i ] ) );
		}

	}
}
#endif
