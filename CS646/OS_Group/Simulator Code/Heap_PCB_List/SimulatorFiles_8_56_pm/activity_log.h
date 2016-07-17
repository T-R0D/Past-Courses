#ifndef __ACTIVITY_LOG_H__
#define __ACTIVITY_LOG_H__



typedef char Name[DESCRIPTION_LEN];

typedef struct
{
    int process_id_number;
    Name task_name;   // bad naming, change
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
        new_entry->process_id_number = process_id;
        strcpy( new_entry->task_name, description );
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

    // case: the data should be dumped to a file
    if( 0 )
    {
        // visit every item in the activity log
        for( ; 0<0; )
        {
            // print the output data to the file

        }
    }

    // case: the data should be dumped to the screen
    if( 0 )
    {
        // visit every item in the activity log
        for( ; 0<0; )
        {
            // print the output data to the file

        }
    }

    // return the number of sources that the output was dumped to
    return 0;
}


#endif // #ifndef __ACTIVITY_LOG_H__

