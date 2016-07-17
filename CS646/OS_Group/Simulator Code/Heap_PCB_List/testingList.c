#include "List.h"

int main()
{
	struct Task_ListNode a = task_listNode_intialize();
	struct Task_List b = task_initialize ();
	struct Task_List c = task_initialize ();

	int i = 0, j = 0;
	char name;

for ( j = 0; j < 3; ++j )
{
	for ( i = 0 ; i < 10; ++i )
	{
		name = 'a' + i ;
		a.task_name [ i ] = name;

	}
		a.task_name [ 10 ] = '\0';
		a.task_number = j;
		a.total_cycles = j * 10;
		a.time_arrived = j * 2; 
		a.cycles_remaining = a.total_cycles - 5;

	//store into the list
	insert_List_task ( &b, a );
	printf ("getting task cycles: %i \n", get_task_cycles (&b) );
	
	//moving to the next task:
 	get_next_task ( &b, 1 );
	move_list_cursor_to_front ( &b);

//	print_node ( &a );
}	



//get a task
insert_List_task (&c, get_listTask (&b) );

print_list(&c);


	print_list (&b);

//remove a task
remove_List_task  ( &c );


	print_list (&c);
return 0;
}	
