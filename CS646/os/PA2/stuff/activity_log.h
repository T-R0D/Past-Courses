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

    // case: the data should be dumped to a file
    if( configurations->log == LOG_TO_FILE ||
        configurations->log == BOTH )
    {
        // open an output file for writing
        file = fopen( "SYSTEM_RUN_LOG.txt", "w" ); // arbitrary file name

        // visit every item in the activity log
        for( i = 0; i < activity_log->entry_count; i++ )
        {
            // print the output data to the file
            if( activity_log->entries[i].process_id == SYSTEM )
            {
                // print the log message
                fprintf( file, "SYSTEM - %s (%f msec)\n",
                         activity_log->entries[i].task_description,
                         (float)activity_log->entries[i].u_seconds_executed / 1000 );
            }
            else
            {
                // print the log message
                fprintf( file, "PID %d - %s (%f msec)\n",
                         activity_log->entries[i].process_id,
                         activity_log->entries[i].task_description,
                         (float)activity_log->entries[i].u_seconds_executed / 1000 );
            }
        }

        // close the file
        fclose( file );
    }

    // case: the data should be dumped to the screen
    if( configurations->log == LOG_TO_MONITOR ||
        configurations->log == BOTH  )
    {
        // visit every item in the activity log
        for( i = 0; i < activity_log->entry_count; i++ )
        {
            // print the output data to the file
            if( activity_log->entries[i].process_id == SYSTEM )
            {
                // print the log message
                printf( "SYSTEM - %s (%f msec)\n",
                        activity_log->entries[i].task_description,
                        (float) activity_log->entries[i].u_seconds_executed / 1000 );
            }
            else
            {
                // print the log message
                printf( "PID %d - %s (%f msec)\n",
                        activity_log->entries[i].process_id,
                        activity_log->entries[i].task_description,
                        (float)activity_log->entries[i].u_seconds_executed / 1000 );
            }
        }
    }

    // return the number of sources that the output was dumped to
    return 0;
}


#endif // #ifndef __ACTIVITY_LOG_H__

