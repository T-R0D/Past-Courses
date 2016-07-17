#ifndef __ACTIVITYLOG_H__
#define __ACTIVITYLOG_H__


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



ActivityLogEntry* makeNewLogEntry(ActivityLog* log);

void printLogEntry(ActivityLogEntry* pEntry, FILE* pStream);


int logEvent(ActivityLog* activity_log, int process_id, char* description,
               int activity_time_usec);



int logEvent(ActivityLog* activity_log, int process_id, char* description,
               int activity_time_usec) {

    int event_log_success = LOG_FAIL;
    ActivityLogEntry* new_entry = NULL;

    new_entry = makeNewLogEntry(activity_log);


    if (new_entry != NULL) {
        // store the given data
        new_entry->process_id = process_id;
        strcpy( new_entry->task_description, description );
        new_entry->u_seconds_executed = activity_time_usec;

        // indicate that the event logging was successful
        event_log_success = LOG_SUCCESS;

printLogEntry(new_entry, stdout);

    } else {
        event_log_success = LOG_FAIL;
    }

    return event_log_success;
}


ActivityLogEntry* makeNewLogEntry(ActivityLog* log) {
    ActivityLogEntry* new_entry = NULL;

    if (log->entry_count < MAX_LOG_ITEMS) {
        new_entry = &(log->entries[log->entry_count]);
        log->entry_count++;
    } else {
        new_entry = NULL;
    }

    return new_entry;
}


int dumpActivityData(ActivityLog* activity_log, FILE* pStream,  ConfigData* configurations) {
    printConfigurations(configurations, pStream);

    int i;
    for(i = 0; i < activity_log->entry_count; i++) {
        ActivityLogEntry entry = activity_log->entries[i];
        printLogEntry(&entry, pStream);
    }

    return 0;
}


void printLogEntry(ActivityLogEntry* pEntry, FILE* pStream) {
    if (pEntry->process_id == SYSTEM) {
        fprintf(pStream, "SYSTEM");
    } else {
        fprintf(pStream, "Process %i", pEntry->process_id);
    }

    fprintf( pStream, " - %s (%d usec)\n",
        pEntry->task_description,
        pEntry->u_seconds_executed
    );
}

#endif // #ifndef __ACTIVITYLOG_H__
