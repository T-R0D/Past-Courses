#ifndef __ACTIVITY_LOG_H__
#define __ACTIVITY_LOG_H__


typedef struct
{
    int process_id;
    char task_description[DESCRIPTION_LEN];
    int u_seconds_executed;
} ActivityLogEntry;


typedef struct
{
    ActivityLogEntry entries[MAX_LOG_ITEMS];
    int entry_count;
} ActivityLog;



ActivityLogEntry* makeNewLogEntry( ActivityLog* log );


int logEvent( ActivityLog* activity_log, int process_id, char* description,
               int activity_time_usec );



int logEvent( ActivityLog* activity_log, int process_id, char* description,
               int activity_time_usec )
{
    // variables
    int event_log_success = LOG_FAIL;
    ActivityLogEntry* new_entry = NULL;

    // attempt to create a new entry in the log
    new_entry = makeNewLogEntry( activity_log );

    // case: the log has not been filled, entry created
    if( new_entry != NULL )
    {
        // store the given data
        new_entry->process_id = process_id;
        strcpy( new_entry->task_description, description );
        new_entry->u_seconds_executed = activity_time_usec;

        // indicate that the event logging was successful
        event_log_success = LOG_SUCCESS;
    }
    // case: the log has been filled
    else
    {
        // report the error
        event_log_success = LOG_FAIL;
    }

    // return the success state of the event logging
    return event_log_success;
}


ActivityLogEntry* makeNewLogEntry( ActivityLog* log )
{
    // variables
    ActivityLogEntry* new_entry = NULL;

    // case: the log is not full
    if( log->entry_count < MAX_LOG_ITEMS )
    {
        // return a reference to the first unused log entry
        new_entry = &(log->entries[ log->entry_count ] );

        // "create" a new log entry simply by incrementing the entry count
        log->entry_count++;
    }
    // case: the log is full
    else
    {
        // indicate that no new entry can be created with NULL
        new_entry = NULL;
    }

    // return the result of the entry creation attempt
    return new_entry;
}


int dumpActivityData( ConfigData* configurations, ActivityLog* activity_log )
{
/* Yes, this algorithm is inefficient, but it more clearly defines expected
   behavior */

    // variables
    int i = 0;
    FILE* file = NULL;
	char scheduler [10];
	char memory_type [20];
	char log_type [20];
 
	//convert the scheduler type from stored into to char array	
	if ( configurations -> cpu_scheduler == RR )
	{
		strcpy (scheduler, "RR");
	}
	if ( configurations -> cpu_scheduler == FIFO)
	{
		strcpy ( scheduler, "FIFO" );
	}

	//convert the memory type to char array
	strcpy ( memory_type, "Fixed" );

	//convert the log type to char array

	if ( configurations -> log == LOG_TO_FILE)
	{
		strcpy ( log_type, "Log to File" );

	}

	if ( configurations -> log == LOG_TO_MONITOR )
	{

		strcpy ( log_type, "Log to Monitor" );
	}
	
	if ( configurations -> log == BOTH )
	{

		strcpy ( log_type, "Log to Both" );
	}

    // case: the data should be dumped to a file
    if( configurations->log == LOG_TO_FILE ||
        configurations->log == BOTH )
    {
        // open an output file for writing
        file = fopen( "SYSTEM_RUN_ACTIVITY.txt", "w" ); // arbitrary file name

		//print the configuration information
		fprintf(file, "Start Simulator Configuration \n" 
						"Version: %s\n" 
						"File path: %s\n" 
						"Quantum (cycles) %d\n" 
						"Processor Scheduling: %s\n" 
						"Processor cycle time (msec): %d\n" 
						"Monitor display time (msec): %d\n" 
						"Hard drive cycle time (msec): %d\n" 
						"Printer Cycle time (msec): %d\n" 
						"Keyboard cycle time (msec): %d\n" 
						"Memory type: %s\n" 
						"Log: %s\n" 
						"End Simulator Configuration File",  
						configurations->version, 
						configurations->file_path,
						configurations->cycles, 
						scheduler, 
						configurations -> cpu_cycle_time, 
						configurations -> monitor_display_time,
						configurations -> hd_cycle_time, 
						configurations -> printer_cycle_time, 
						configurations -> keyboard_cycle_time, 
						memory_type,
						log_type);

		fprintf (file, "\n \n \n");

        // visit every item in the activity log
        for( i = 0; i < activity_log->entry_count; i++ )
        {
            // print the output data to the file
            if( activity_log->entries[i].process_id == SYSTEM )
            {

                // print the log message
                fprintf( file, "SYSTEM - %s (%d msec)\n",
                         activity_log->entries[i].task_description,
                         activity_log->entries[i].u_seconds_executed / 1000 );
                                                 // convert to msecs ^^^ 
            }
            else
            {

                // print the log message
                fprintf( file, "PID %d - %s (%d msec)\n",
                         activity_log->entries[i].process_id,
                         activity_log->entries[i].task_description,
                         activity_log->entries[i].u_seconds_executed / 1000 );
                                                 // convert to msecs ^^^
            }
        }

        // close the file
        fclose( file );
    }

    // case: the data should be dumped to the screen
    if( configurations->log == LOG_TO_MONITOR ||
        configurations->log == BOTH  )
    {
		printf("Start Simulator Configuration \n" 
						"Version: %s \n" 
						"File path: %s \n" 
						"Quantum (cycles) %d\n" 
						"Processor Scheduling: %s\n" 
						"Processor cycle time (msec): %d\n" 
						"Monitor display time (msec): %d\n" 
						"Hard drive cycle time (msec): %d\n" 
						"Printer Cycle time (msec): %d\n" 
						"Keyboard cycle time (msec): %d\n" 
						"Memory type: %s\n" 
						"Log: %s\n" 
						"End Simulator Configuration File",  
						configurations->version, 
						configurations->file_path,
						configurations->cycles, 
						scheduler, 
						configurations -> cpu_cycle_time, 
						configurations -> monitor_display_time,
						configurations -> hd_cycle_time, 
						configurations -> printer_cycle_time, 
						configurations -> keyboard_cycle_time, 
						memory_type,
						log_type);

		printf ("\n \n \n"	);

        // visit every item in the activity log
        for( i = 0; i < activity_log->entry_count; i++ )
        {
            // print the output data to the file
            if( activity_log->entries[i].process_id == SYSTEM )
            {
                // print the log message
                printf( "SYSTEM - %s (%d msec)\n",
                        activity_log->entries[i].task_description,
                        activity_log->entries[i].u_seconds_executed / 1000 );
                                                 // convert to msecs ^^^
            }
            else
            {
                // print the log message
                printf( "PID %d - %s (%d msec)\n",
                        activity_log->entries[i].process_id,
                        activity_log->entries[i].task_description,
                        activity_log->entries[i].u_seconds_executed / 1000 );
                                                 // convert to msecs ^^^
            }
        }
    }

    // return the number of sources that the output was dumped to
    return 0;
}


#endif // #ifndef __ACTIVITY_LOG_H__

