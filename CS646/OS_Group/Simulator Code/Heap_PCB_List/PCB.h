#ifndef PCB_H
#define PCB_H
#include "List.h"

//create a struct to hold all the tasks of a PCB
typedef struct{

	int pid;
	int time_arrived;
	int total_process_time;
	int priority_value;
	 Task_List job_list;
}PCB;

///////////////////function specifications//////////////////////////////////////

//initialize pcb function
PCB initialize_PCB ();

//create a PCB struct, by reading in all the values into the pcb
PCB create_PCB ( int pid, int arrival, int priority,  Task_List something );

//debug function to print out the PCB
void print_PCB (  PCB* a );

//update the current cycles for the pcb task
int update_current_task_cycles (  PCB *object, int time_tick);

//get teh current remaining cycles for the current pcb task
int get_remaining_cycles ( PCB *object);

//get the next pcb job
int get_next_PCB_job (  PCB *object);

//get the priority of the pcb task ( not being used, will use if 
	//implement priority based implementation)
int get_priority_of_PCB_task (  PCB *object );

//get the total time it took to complete the pcb task
int update_total_PCB_time (  PCB *object, int quantz );

//function to check if the pcb is empty
int empty_PCB (  PCB *object );

//get the total time it took to complete the process (all PCB tasks)
int get_PCB_completion_time (  PCB *object );

//////////////////function implementations//////////////////////////////////////

//initialize pcb
PCB initialize_PCB ()
{
	 PCB temp;
	
	temp.pid = 0;
	temp.time_arrived = -1;
	temp.total_process_time = 0;
	temp.priority_value = -1;
	temp.job_list = task_initialize ();
	return temp;

}

//function to create a pcb by inserting all needed member information
PCB create_PCB ( int pid, int arrival, int priority,  Task_List something )
{
	//create pcb 
	 PCB temp = initialize_PCB();

	//update the pid
	temp.pid = pid;

	//update the time of arrival
	temp.time_arrived = arrival;

	//update the process priority
	temp.priority_value = priority;

	//update the pcb list
	temp.job_list= something;

	//return the 
	return temp;
}

//debug function to print a pcb
void print_PCB (  PCB* a )
{
	printf ("PCB pid: %i ", a -> pid);
	printf ("PCB time arrived: %i", a-> time_arrived);
	printf ("PCB total process time: %i", a-> total_process_time);
	printf ("PCB priority value: %i", a-> priority_value);

	//print the job list
	puts("Jobs List: ");
	printf ("\n");
	print_list ( &( a -> job_list) );
	printf ("\n \n");
}

//update the cycles of the current pcb job
int update_current_task_cycles (  PCB *object, int time_tick) 
{
	//update the time_cycles_remaining
	int value = update_task_cycles( &(object -> job_list ), time_tick );

	//the tasks have been completed move it to done list and move to next task
	if ( value <= 0 )
	{
		//move to the next task
		remove_List_task ( &(object -> job_list));
	}

	return value;
}

//get the remaining cycles of the current job
int get_remaining_cycles ( PCB *object)
{
	//access the list object and get its current access time 
	return get_task_cycles ( &(object->job_list) );
	
} 

//get the next job in the PCB
int get_next_PCB_job (  PCB *object)
{
	if ( empty_list ( &(object -> job_list) ) != 1 )
	{	

		//if the iterator is at the end but list not empty move to front
		if (is_iterator_at_end ( &(object -> job_list )) == 1 )
		{
			move_list_iterator_to_front ( & (object -> job_list ) );

			return 1;
		}
	
		//otherwise move the iterator to the next position
		else
		{
			increment_iterator ( &(object -> job_list ) );

			return 0;
		}
	}
}

//get the priority of the curent job
int get_priority_of_PCB_task (  PCB *object )
{
	//if the pcb is not empty return priority of current task
	if ( empty_list ( & (object -> job_list ) ) != 1 )
	{
		return get_task_priority( &( object -> job_list ) );

	}

	else
	{
		puts ("Empty PCB" );
		return 0;
	}
	
}

//keep track of the total time that it takes to complete a PCB
int update_total_PCB_time (  PCB *object, int quantz )
{
	if ( quantz > 0 )
	{
		object -> total_process_time += quantz;
		return 1;
	}

	else
	{
		puts("Please enter the appropriate time quantity.");
		return 0;
	}
}

//retrieve the PCB time
int get_PCB_completion_time (  PCB *object )
{
	return object -> total_process_time;

}

//check to see if the pcb is empty
int empty_PCB (  PCB *object )
{
	int flag = 0;

	if ( empty_list (& (object -> job_list ) ) == 1 )
	{

		flag = 1;
	}

	return flag;
}
#endif
