#ifndef LIST_H
#define LIST_H
#include <stdlib.h>		//for dynamic memory allocation 
#include <stdio.h>		//for io stream in c
#include <string.h>
#include <time.h>

#define MAX_SIZE 2000

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
}Task_List ;

///////////////////function specifications//////////////////////////////////////

//intialize the node struct
Task_ListNode task_listNode_intialize();

//intializes the list of nodes struct
void task_initialize( Task_List* something );

//inserts a new task into the list
int insert_List_task (  Task_List* someList, Task_ListNode something );

//removes the current task from the list
Task_ListNode remove_List_task ( Task_List* someList); 

//function to return the current task in the task list
Task_ListNode get_listTask(  Task_List* someList );

//function to update the current task cycles ( subtracts the cycles from task time)
int update_task_cycles (  Task_List* someList, int cycles);

//function that returns the total task cycles left for the current task in the list
int get_task_cycles (  Task_List* someList );

//function to retrieve the priority of the current task/job 
int get_task_priority (  Task_List* someList );

//function to get the next task in the list
Task_ListNode get_next_task (  Task_List* someList, int move );

//debug function to print the node struct
void print_node (  Task_ListNode* a );

//debug function to prin the list
void print_list (  Task_List* b);

//function to move the current iterator to the next position
int increment_iterator ( Task_List* f);

//function to move the iterator to the front of the list (helps with wrapping)
int move_list_iterator_to_front ( Task_List* c);

//function that returns info if the iterator is at the end
int is_iterator_at_end ( Task_List* e);

//function to specify if the list is empty
int empty_list ( Task_List* d);

//////////////////function implementations//////////////////////////////////////
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

void task_initialize( Task_List* something )
{
	//set the iterator of the  equal to negative one
	something->iterator = 0;
	something->front = 0;
	something->end = 0;

    // return the list by reference
}

//insert task
int insert_List_task (  Task_List* someList,  Task_ListNode something )
{
    int insert_success = false;

	if ( someList->end + 1 == MAX_SIZE ) //if reached max size of list
	{
		//print out error function
		puts ("Max capacity of list reached please free before trying to insert");
	}

	else //insert the value into the list and update all iterators
	{

		//insert the item into the list
		someList->tasks [ someList -> end ] = something;

		//increment the end position
		someList->end ++;

        // indicate that insertion was successful
        insert_success = true;
	}

    return insert_success;
}

//remove task from the list by moving the front to the next position
Task_ListNode remove_List_task ( Task_List* someList)
{
    // get the item currently at the front of the list
    Task_ListNode temp = someList->tasks[someList->iterator];

	//move the front to the next position
	someList -> front ++;

    // return the just removed item
    return temp;	
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

    return 0; // stub for now
}

//get total cycles remaining
int get_task_cycles (  Task_List* someList )
{
	return someList -> tasks [ someList -> iterator ].total_cycles;

}

//function that get the priority of a task w/i list and returns it
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

        // return some garbage
        return someList -> tasks [ 0 ];
	}
}


//print the task 
void print_node (  Task_ListNode* a )
{
	
	printf ( " task_Name : %s  number: %i total_cycles :%i cycles_remaining: %i priority: %i \n" , 
			(a -> task_name) , a -> task_number, a -> total_cycles,
			 a -> cycles_remaining, a->task_priority );
}

//print the task list
void print_list (  Task_List* b)
{
	int i = 0;

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

//increment the current list iterator to the next node
int increment_iterator ( Task_List* f)
{
    int increment_success = false;

	if ( f -> iterator < f -> end )
	{
		f -> iterator ++;
        increment_success = true;
	}

    return increment_success;
}

//moves the current iterator to the front
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

//checks to see if the list is empty or not (returns 1 for true, 0 for fals)
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

//function returns 1 (true) or 0 (false) if the iterator is at the end
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

