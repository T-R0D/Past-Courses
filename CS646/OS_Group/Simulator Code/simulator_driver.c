/*==============================================================================
    HEADER FILES
==============================================================================*/


#include "my_stopwatch.h" 
#include "simulator_io_thread.h"
#include "activity_log.h"


/*==============================================================================
    GLOBAL CONSTANTS
==============================================================================*/
#define NO_ERRORS 0
#define WORD_LEN 100
#define MAX_LOG_ITEMS 100000
#define IO_MANAGEMENT_TIME 200

// timers
#define GLOBAL_CLOCK 0
#define SYSTEM_ACTION_CLOCK 1
#define IO_TIMER 2

// device enumeration
#define MAX_IO_DEVICES 5
#define HARDDRIVE_READ 0
#define HARDDRIVE_WRITE 1
#define KEYBOARD 2
#define MONITOR 3
#define PRINTER 4
#define CPU 99             // these two are same thing
#define PROCESS 99         // have both for readability (hopefully)

/*==============================================================================
    USER DEFINED TYPES
==============================================================================*/

typedef struct
{
    char name[WORD_LEN];
    PcbList waiting_queue;
    PCB* working_process;
    IoData data;
} Device;

typedef struct
{
    Device harddrive_write;
    Device harddrive_read;
    Device keyboard;
    Device monitor;
    Device printer;
} DeviceSet;


/*==============================================================================
    GLOBAL VARIABLES
==============================================================================*/



/*==============================================================================
    FUNCTION PROTOTYPES
==============================================================================*/
int executeSimulation( Heap* ready_queue, DeviceSet* io_queues,
                       DataLogEntry** activity_log,
                       ConfigData* configurations );

int manageIoTasks( DeviceSet* io_queues, ActivityLog* activity_log );

int manageIoDevice( Device* device, PcbList* main_cheduler,
                    ActivityLog* activity_log );

int manageContextSwitching( PCB* currently_running_process, Heap* ready_queue,
                            DeviceSet* io_queues, ActivityLog* activity_log );

int dumpActivityData( ConfigData* configurations, ActivityLog* activity_log );



/*==============================================================================
    DRIVER PROGRAM
==============================================================================*/
int main( int argc, char** argv )
{
    // start the timer to track the system bootup time
    stopwatch( 's', GLOBAL_CLOCK );

    // variables
    int system_status = NO_ERRORS;
    ConfigData configurations;
    Heap ready_queue;
    DeviceSet io_queues;
    ActivityLog activity_log;

    // perform system bootup actions (which does actually include the
    // inittialization of the above the items)
    system_status = bootSystem( &configurations, &ready_queue, &activity_log );

    // store the time for starting up the system in the activity log
    logAction( &activity_log, SYSTEM, stopwatch( 'x', GLOBAL_CLOCK );

    // case: the system booted successfully
    if( system_status = NO_ERRORS )
    {
        // execute the simulation
        system_status = executeSimulation( &configurations, &ready_queue,
                                           &activity_log );
    }

    // dump the activity log to the specified locations
    dumpActivityData( &configurations, &activity_log );

    // return the program run status to end the program
    return system_status;
}


/*==============================================================================
    FUNCTION IMPLEMENTATIONS
==============================================================================*/
int executeSimulation( Heap* ready_queue, DeviceSet* io_queues,
                       DataLogEntry** activity_log,
                       ConfigData* configurations );
{
    // variables
    PCB* currently_running_process = get_next_item( ready_queue );
    int cycles_remaining = configurations->cycles;  // awkward naming

    // run simulator until there are not more jobs to process
    while( currently_running_process != NULL || !isEmpty( ready_queue ) )
    {
        // manage any interrupts I/O if there are any
        manageIoTasks( ready_queue, io_queues, activity_log );

        // perform any necessary context switching
        manageContextSwitching( currently_running_process, ready_queue,
                                io_queues, activity_log );

        // simulate the running process being given some CPU time
        runProcess( currently_running_process, configurarations );

        // count a CPU cylce for the running process
        cycle++;
    }

    // return simulation execution status
    return 0;
}


int manageIoTasks( DeviceSet* devices, Heap* main_scheduler,
                   ActivityLog* activity_log )
{
    // handle each type of I/O device
    manageIoDevice( &(devices->harddrive_write), main_scheduler, activity_log );
    manageIoDevice( &(devices->harddrive_read), main_scheduler, activity_log );
    manageIoDevice( &(devices->keyboard), main_scheduler, activity_log );
    manageIoDevice( &(devices->monitor), main_scheduler, activity_log );
    manageIoDevice( &(devices->printer), main_scheduler, activity_log );

  return 0;  // stub for later improved functionality - status reporting
}


int manageIoDevice( Device* device, PcbList* main_scheduler,
                    ActivityLog* activity_log )
{
    // variables
    char action_message[WORD_LEN];

    // case: an I/O completion interrupt has been raised
    if( device->data.interrupt_flag )
    {
        // time this action
        stopwatch( 's', IO_TIMER );

        // kill the I/O simulating thread
        pthread_join( device->data.thread_id, NULL );

        // release the device from the process, load it into the main
        // scheduler
        device->data.interrupt_flag = false;
        device->data.in_use = false;
        heapInsert( main_scheduler, decive->data->working_process );
        
        // log this action
        sprintf( action_message, "%s action completed for process %d - process
                                 "placed back in main scheduler",
                 device->name, device->working_process.pid );
        working_process = NULL;
        logAction( activity_log, SYSTEM, action_message,
                   stopwatch( 'x', IO_TIMER ) );
    }
    // case: an I/O processed is running and needs to be managed
    else if( device->data.in_use )
    {
        // log the maintenance of the process (simulate with a short wait)
        stopwatch( 's', IO_TIMER );
        usleep( IO_MANAGEMENT_TIME );
        sprintf( action_message,
                 "I/O maintnenance: %s still in use by process %d",
                 device->name, device->working_process.pid );
        logAction( activity_log, SYSTEM, action_message,
                   stopwatch( 'x', IO_TIMER ) );
    }

    // case: an I/O process is wating to run and the I/O device is free
    //       (it is possible that the device was previously freed previously)
    if( !(device->data.device_in_use) && !isEmpty( device->waiting_queue ) )
    {
        // start the I/O process on this device
        stopwatch( 's', IO_TIMER );
        device->working_process = ListPop( device->waiting_queue );

        // startup the independent I/O action (simulated of course)
        pthread_attr_init( &(device->data.attribute) );
        pthread_create( &(device->data.thread_id), &(device->data.attribute),
                        conductIoProcess, (void*) &(device->data) );

        // log the action
        sprintf( action_message, "Starting %s for process %d",
                 device->name, device->working_process.pid );
    }
}


int manageContextSwitching( PCB* currently_running_process, Heap* ready_queue,
                            DeviceSet* io_queues, ActivityLog* activity_log )
{
    // case: the currently running process has used up its turn to run
    if( cycles_remaining <= 0 ||
        get_remaining_cycles( currently_running_process ) <= 0 )
    {
        // place the process back in the ready_queue
        HeapInsert( Heap, currently_running_process );
        currently_running_process = NULL;

        // perform context switches until we get a processor destined
        // for the CPU or we are out of processeses to get
        while( ( getTaskType( currently_running_process->job_list ) ==
                 PROCESS ) ||
               !isEmpty( ready_queue ) )
        {
            // place the currently running process in the back of the queue
            // (if we have one that has not already been placed somewhere
            // else, like an I/O queue)
            if( currently_running_process != NULL )
            {
                // get one from the queue if possible
                currently_running_process = HeapPop( ready_queue );
            }

            // case: the next item is not destined for processing time
            //       (i.e. an I/O process)
            if( !(get_task_type( currently_running_process->job_list ) ==
                  PROCESS)  )
            {
                // send the process to the I/O queue it is destined for
                sendToIoQueue( currently_running_process, io_queues,
                               activity_log );
            }
            // otherwise, the process can be left in the running state
        }
    }
}


int runProcess( PCB* currently_running_process, ConfigData* configurations,
                ActivityLog* activity_log )
{
    // variables
    int cycles_remaining = configurations->cycles;  // a time quantum
    int time_ran = 0;
    char action_message[WORD_LEN];

    // "run" the process for the time specified by the time quantum
    while(  ( cycles_remaining > 0 ) &&
            ( currently_running_process->job_list->tasks[ currently_running_process->job_list->iterator ].cycles_remaining > 0 ) )
                            // ^^^ make a function for this mess ^^^
    {
        // decrement the number of cycles the taks needs to complete
        (currently_running_process->job_list->tasks[ currently_running_process->job_list->iterator ].cycles_remaining)--

        // decrement the number of cycles the task has left to run
        cycles_remaining--
    }

    // compute the "actual" time that the process was allowed to run for
    time_ran = ( configurations->cycles - cycles_remaining ) *
               configurations->cpu_cycle_time;

    // log the event
    sprintf( action_message, "Process %d: ran on CPU for %d cycles",
             currently_running_process->pid,
             ( configurations->cycles - cycles_remaining ) );
    logAction( activity_log, currently_running_process->pid, action_message,
               time_ran );
}


