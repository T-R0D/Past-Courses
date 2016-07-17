#ifndef LIST_H
#define LIST_H
#include <stdlib.h>		//for dynamic memory allocation 
#include <stdio.h>		//for io stream in c
#include <string.h>
#include <time.h>

#define MAX_SIZE 1000

// that holds the task name, number, cycle information
typedef  struct
{
	char task_name [20];
	int task_number;
	int total_cycles;
	int cycles_remaining;
	int task_priority;
} Task_ListNode;

// that holds a list of the tasks along with the begin, end, and iterator
typedef  struct
{
	Task_ListNode tasks [MAX_SIZE];
	int iterator;
	int front;
	int end;
} Task_List ;

 Task_ListNode task_listNode_intialize()
{
	//intialize all the members of the  Task
	 Task_ListNode temp;
	
	//set the total cycles to some non-pertinent number
	temp.total_cycles = -1;

	//set the time remaining to some non-pertinent number
	temp.cycles_remaining = -1;

	//set the task value to -1
	temp.task_priority = -1;
	
	//return the intialized object
	return temp;
}

 Task_List task_initialize ()
{

	//create a task 
	 Task_List something;

	//set the iterator of the  equal to negative one
	something.iterator = 0;
	something.front = 0;
	something.end = 0;

	//return the task pointer
	return something;
}

//insert task
 int insert_List_task (  Task_List* someList,  Task_ListNode something )
{

	if ( someList->end + 1 == MAX_SIZE ) //if reached max size of list
	{
		//print out error function
		puts ("Max capacity of list reached please free before trying to insert");

		//return something indicating failiure
		return 0;
	}

	else //insert the value into the list and update all iterators
	{

		//insert the item into the list
		someList->tasks [ someList -> end ] = something;

		//increment the end position
		someList->end ++;


	}
}


 
//remove task from the list by moving the front to the next position
int remove_List_task ( Task_List* someList)
{
	//move the front to the next position
	someList -> front ++;	
}

//get task
 Task_ListNode get_listTask(  Task_List* someList )
{
	//return the current task in the tasklist
	return someList -> tasks [ someList -> iterator ];
}

//update task cycles
int update_task_cycles (  Task_List* someList, int cycles)
{
	someList -> tasks [ someList -> iterator ].cycles_remaining  -= cycles;

}

//get total cycles remaining
int get_task_cycles (  Task_List* someList )
{
	return someList -> tasks [ someList -> iterator ].total_cycles;

}

int get_task_priority (  Task_List* someList )
{
	return someList -> tasks [ someList -> iterator ].task_priority;

}


//get next task item and be able to move to the next position 
 Task_ListNode get_next_task (  Task_List* someList, int move )
{
	//create a temp iterator set to list iterator
	int temp = someList -> iterator;
	
	//if the next task is within the boundies continue
	if ( temp < someList -> end )
	{
		//if user wants to move the iterator move to the next position
		if ( move == 1 )
		{	
			someList -> iterator ++;
		}

		//otherwise leave the iterator at the same place and return the value
		return someList -> tasks [ temp + 1 ];	
	}

	else //else print error message
	{
		puts ("Next task doesn't exits because reached end of list.");
	}
}

//print the task 
void print_node (  Task_ListNode* a )
{
	
	printf ( " task_Name : %s  number: %i total_cycles :%i cycles_remaining: %i priority: %i \n" , (a -> task_name) , a -> task_number, 
			a -> total_cycles, a -> cycles_remaining, a->task_priority );
}

//print the task list
void print_list (  Task_List* b)
{
	int i = 0;

//	printf( "front, iterator and back: %i %i %i \n", b ->front, b->iterator, b->end);

	if ( b->front == b->end)
	{
		puts ("Empty List" );

	}

	else 
	{
		//start from the beginning and print till end reahed
		for ( i = b->front; i < b->end; ++i)
		{
			print_node ( & (b -> tasks [ i ] ));

		}	
	}
}

int increment_iterator ( Task_List* f)
{
	if ( f -> iterator < f -> end )
	{
		f -> iterator ++;
	}

}

int move_list_iterator_to_front ( Task_List* c)
{
	//move the list to the front if the list isn't empty
	if (c -> front != c -> end )
	{
		//set the front equal to the end
		c -> iterator = c -> front;

		//return 1 to indicate true
		return 1;

	}
 
	//else return 0 to indicate false
	return 0;

}
int empty_list ( Task_List* d)
{
	//if the front is equal to end list is empty
	if ( d->front == d->end )
	{
		return 1;
	
	}

	else 
	{
		return 0;

	}
}

int is_iterator_at_end ( Task_List* e)
{
	if ( e -> iterator == e -> end )
	{
		return 1;
	}

	else
	{
		return 0;
	} 		

}

#endif

