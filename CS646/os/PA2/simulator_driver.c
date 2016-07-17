/*
    This is a "High-Level" Simulator.

    Please note that if you see something that looks like an additional
    feature, it may just be the preparations for an additional feature, so
    you may not find the feature itself.

    When running this program, please note that it accepts 2 command line
    arguments: the first one being the configurations file name, and the
    second being the meta-data file name.
    Ex: ./a.out my_config_file my_meta_data_file

    Also please note that this program uses real time for timing, so please
    use meta-data according to how long a program run should take.
*/

/*==============================================================================
    HEADER FILES
==============================================================================*/
// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

// constants
#include "constants.h"

// special functions
#include "readin_config.h"
#include "meta_data_readin.h"
#include "simulator_io.h"
#include "my_stopwatch.h"

// containers
#include "List.h"
#include "Heap.h"
#include "activity_log.h"

// objects
#include "PCB.h"


/*==============================================================================
    GLOBAL CONSTANTS
==============================================================================*/


/*==============================================================================
    USER DEFINED TYPES
==============================================================================*/

typedef struct
{
    int io_still_running;
    Device device_number[MAX_IO_DEVICES]; // strange naming - may be less weird
                                          // in actual code use
/* from constants.h:
#define HARDDRIVE_READ 0
#define HARDDRIVE_WRITE 1
#define KEYBOARD 2
#define MONITOR 3
#define PRINTER 4
#define MAX_IO_DEVICES 5
*/
} DeviceSet;

typedef struct
{
    int process_is_running;
    PCB currently_running_process;
} RunningState;


/*==============================================================================
    GLOBAL VARIABLES
==============================================================================*/



/*==============================================================================
    FUNCTION PROTOTYPES
==============================================================================*/
int bootSystem( ConfigData* configurations, Heap* ready_queue,
                DeviceSet* devices, char* config_file_name,
                char* meta_data_file_name, ActivityLog* activity_log );

int executeSimulation( ConfigData* configurations, Heap* ready_queue,
                       DeviceSet* device_set, ActivityLog* activity_log );

int manageIoTasks( Heap* ready_queue, DeviceSet* devices,
                   ConfigData* configurations, ActivityLog* activity_log );

int manageIoDevice( Device* device, Heap* ready_queue,
                    ConfigData* configurations, ActivityLog* activity_log );

int manageContextSwitching( RunningState* run_state, Heap* ready_queue,
                            DeviceSet* device_set, ConfigData* configurations,
                            ActivityLog* activity_log );

int sendToIoQueue( PCB* process, DeviceSet* device_set,
                   ActivityLog* activity_log );

int runProcess( RunningState* run_state, ConfigData* configurations,
                ActivityLog* activity_log );

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
    DeviceSet device_set;
    ActivityLog activity_log;

    // perform system bootup actions (which does actually include the
    // inittialization of the above the items)
    system_status = bootSystem( &configurations, &ready_queue, &device_set,
                                argv[CONFIG_FILE], argv[2],
                                &activity_log );

    // case: the system booted successfully
    if( system_status == NO_ERRORS )
    {
        // execute the simulation
        system_status = executeSimulation( &configurations, &ready_queue,
                                           &device_set, &activity_log );
    }


    // dump the activity log to the specified locations
    dumpActivityData( &configurations, &activity_log );

    // return the program run status to end the program
    return system_status;
}


/*==============================================================================
    FUNCTION IMPLEMENTATIONS
==============================================================================*/
int bootSystem( ConfigData* configurations, Heap* ready_queue,
                DeviceSet* devices, char* config_file_name,
                char* meta_data_file_name, ActivityLog* activity_log )
{
    // start the system boot timer
    stopwatch( 's', BOOT_CLOCK );

    // variables
    int system_boot_success = NO_ERRORS;
    int device_port_num = 0;

    // read the configuration data
    system_boot_success = readConfigurationData( configurations,
                                                 config_file_name ); 

    logEvent( activity_log, SYSTEM, "System configured",
               stopwatch( 'x', BOOT_CLOCK ) ); 

    // case: configuration was successful
    if( system_boot_success == NO_ERRORS )
    {
        // time this
        stopwatch( 's', BOOT_CLOCK );

        // initialize the I/O devices
        for( device_port_num = 0; device_port_num < MAX_IO_DEVICES;
             device_port_num++ )
        {
            // initialize a device
            system_boot_success =
                connectToDevice( &(devices->device_number[device_port_num]),
                                 device_port_num );
        }
        logEvent( activity_log, SYSTEM, "Peripheral devices initialized",
                  stopwatch( 'x', BOOT_CLOCK ) );
    }

    // case: device location was successful
    if( system_boot_success == NO_ERRORS )
    {
        // read in the meta data and load the main scheduler
        system_boot_success = readMetaData( ready_queue, meta_data_file_name,
                                            configurations );

        logEvent( activity_log, SYSTEM, "Meta-data read, batch schedule loaded",
                  stopwatch( 'x', BOOT_CLOCK ) );

        // store the time for starting up the system in the activity log
        logEvent( activity_log, SYSTEM, "System fully booted",
                  stopwatch( 'x', GLOBAL_CLOCK ) );
    }

    // case: the system did not boot successfully
    if( system_boot_success != NO_ERRORS )
    {
        // log that the booting failed
        logEvent( activity_log, SYSTEM, "System boot failed",
                  stopwatch( 'x', GLOBAL_CLOCK ) );
    }

    // return the booting success status
    return system_boot_success;
}


int executeSimulation( ConfigData* configurations, Heap* ready_queue,
                       DeviceSet* device_set, ActivityLog* activity_log )
{
    // variables
    RunningState run_state;
        run_state.process_is_running = false;
        //run_state.currently_running_process = remove_PCB_from_Heap( ready_queue );
    int cycles_remaining_in_quantum = configurations->cycles;  // awkward naming
    int processing_finished = true;  // since nothing has run yet
                                     // processing is finished and a
                                     // context switch is necessary
    device_set->io_still_running = false;

    // run simulator until there are not more jobs to process
    while( run_state.process_is_running ||
           !is_Heap_empty( ready_queue ) ||
           device_set->io_still_running )
    {
        // manage any interrupts I/O if there are any
        manageIoTasks( ready_queue, device_set, configurations, activity_log );

        // case: the time quantum has passed or the process doesn't need any
        //       more CPU time
        if( cycles_remaining_in_quantum <= 0 || processing_finished )
        {
            // perform any necessary context switching
            manageContextSwitching( &(run_state), ready_queue, device_set,
                                    configurations, activity_log );

            // start a new time quantum
            cycles_remaining_in_quantum = configurations->cycles;
            processing_finished = false;
        }

        // simulate the running process being given some CPU time
        processing_finished = runProcess( &(run_state), configurations,
                                          activity_log );

        // count a processing time cycle as passed
        cycles_remaining_in_quantum--;
    }

    // return simulation execution status
    return 0;
}


int manageIoTasks( Heap* ready_queue, DeviceSet* devices,
                   ConfigData* configurations, ActivityLog* activity_log )
{
    // variables
    int device_management_status = NO_ERRORS; // for future improvement
    int i = 0;
    int io_still_running_flags[MAX_IO_DEVICES];
    int num_still_running = 0;

    // handle each type of I/O device
    for( i = 0; i < MAX_IO_DEVICES; i++ )
    {
        // manage that I/O device
        manageIoDevice( &(devices->device_number[i]), ready_queue,
                        configurations, activity_log );

        // keep track of if it has a process using the device
        io_still_running_flags[i] = devices->device_number[i].data.in_use;
    }

    // check for I/O is still running
    for( i = 0; i < MAX_IO_DEVICES; i++ )
    {
        if( io_still_running_flags[i] )
        {
            // count the flags that indicate I/O is running
            num_still_running++;
        }
    }

    // case: we have I/O still running
    if( num_still_running > 0 )
    {
        // report this to the system
        devices->io_still_running = true;
    }
    // case: we have no I/O still running
    else
    {
        // report this to the system
        devices->io_still_running = false;
    }

    // return the device management status
    return device_management_status;
}


int manageIoDevice( Device* device, Heap* ready_queue,
                    ConfigData* configurations, ActivityLog* activity_log )
{
    // variables
    int device_management_success = NO_ERRORS;
    char action_message[DESCRIPTION_LEN];

    // case: an I/O completion interrupt has been raised
    if( device->data.interrupt_flag )
    {
        // time this action
        stopwatch( 's', IO_TIMER );

        // kill the I/O simulating thread
        pthread_join( device->data.thread_id, NULL );

        // release the device from the process, move on to the next task,
        // load it into the main scheduler
        device->data.interrupt_flag = false;
        device->data.in_use = false;
        device->working_process.job_list.iterator++;
        insert_heap( ready_queue, device->working_process, FIFO_HEAP );

        // log this action
        sprintf( action_message, "%s action completed - "
                                 "process placed back in main scheduler",
                 device->name );

        logEvent( activity_log, device->working_process.pid, action_message,
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
        logEvent( activity_log, SYSTEM, action_message,
                   stopwatch( 'x', IO_TIMER ) );
    }

    // case: an I/O process is wating to run and the I/O device is free
    //       (it is possible that the device was previously freed previously)
    if( !(device->data.in_use) && !is_Heap_empty( &(device->waiting_queue ) ) )
    {
        // start the I/O process on this device
        stopwatch( 's', IO_TIMER ); 
        device->working_process =
            remove_PCB_from_Heap( &(device->waiting_queue) );

        // startup the independent I/O action (simulated of course)
        device->data.in_use = true;
        device->data.interrupt_flag = false;
        device->data.usec_to_run =
                
        pthread_attr_init( &(device->data.attribute) );
        pthread_create( &(device->data.thread_id), &(device->data.attribute),
                        conductIoProcess, (void*) &(device->data) );

        // log the action
        sprintf( action_message, "Starting action on %s", device->name );
        logEvent( activity_log, device->working_process.pid, action_message,
                  stopwatch( 'x', IO_TIMER ) );
    }

    // return the success of managing the device
    return device_management_success;
}


int computeIoRunningTime( PCB* process, ConfigData* configurations )
{
    // variables
    int cycles_required_for_task =
        (process->job_list.tasks[ process->job_list.iterator ].cycles_remaining);
    int task_type =
        ( process->job_list.tasks[process->job_list.iterator].task_number );
    int time_per_cycle = 0;

    // find the time per cycle based on device type
    if( task_type == HARDDRIVE_READ || task_type == HARDDRIVE_WRITE )
    {
        // assign the appropriate time per cycle
        time_per_cycle = configurations->hd_cycle_time;
    }
    else if( task_type == KEYBOARD )
    {
        // assign the appropriate time per cycle
        time_per_cycle = configurations->keyboard_cycle_time;
    }
    else if( task_type == MONITOR )
    {
        // assign the appropriate time per cycle
        time_per_cycle = configurations->monitor_display_time;
    }
    else if( task_type == PRINTER )
    {
        // assign the appropriate time per cycle
        time_per_cycle = configurations->printer_cycle_time;
    }
    else  // for custom, unrecognized hardware
    {
        // assign an arbitrary time per cycle
        time_per_cycle = 200;
    }

    // return the time the process will take
    return ( cycles_required_for_task * time_per_cycle );
}


int manageContextSwitching( RunningState* run_state, Heap* ready_queue,
                            DeviceSet* device_set, ConfigData* configurations,
                            ActivityLog* activity_log )
{
    // variables
    int context_switch_made = false;
    char action_message[DESCRIPTION_LEN];

    // case: we have a process that was running
    if( run_state->process_is_running )
    {
        // time this action
        stopwatch( 's', SYSTEM_ACTION_CLOCK );

        // indicate that a context switch was made
        context_switch_made = true;

        // place the process in the back of the ready queue
        run_state->process_is_running = false;

        // log the event
        logEvent( activity_log, run_state->currently_running_process.pid,
                  "Completed processing quantum and was put back in "
                  "the ready_queue",
                  stopwatch( 'x', PROCESSING_CLOCK ) );

        // case: the process has not completed all of its code/tasks/actions
        if( !is_iterator_at_end(
                 &(run_state->currently_running_process.job_list) ) )
        {
            insert_heap( ready_queue, run_state->currently_running_process,
                         configurations->cpu_scheduler );

        }
        // case: the process completed all instructions
        else
        {
            // log that the process finished processing
            logEvent( activity_log, run_state->currently_running_process.pid,
                      "Ran to completion (time is global system time required"
                      "to complete)",
                      stopwatch( 'x', GLOBAL_CLOCK ) );         
        }
    }

    // time the retrieval of a new process
    stopwatch( 's', SYSTEM_ACTION_CLOCK );

    // get one from the queue if possible
    if( !is_Heap_empty( ready_queue ) )
    {
        // pop the item out of the queue
        run_state->currently_running_process =
            remove_PCB_from_Heap( ready_queue );
        run_state->process_is_running = true;
    }
    // case: there were no items to remove from the queue
        // leave the run state empty, with no process running

    // perform context switches until we get a processor destined
    // for the CPU or we are out of processeses to get
    while( ( get_listTask( &(run_state->currently_running_process.job_list) ).task_number != PROCESS ) &&
           !is_Heap_empty( ready_queue ) )
    {
        // send the process to the I/O queue it is destined for
        sendToIoQueue( &(run_state->currently_running_process), device_set,
                       activity_log );

        // get one from the queue if possible
        if( !is_Heap_empty( ready_queue ) )
        {
            // pop the item out of the queue
            run_state->currently_running_process =
                remove_PCB_from_Heap( ready_queue );
        }
        // case: there are no items to remove from the queue
            // leave the run state empty, with no process running
    }

    // case: it was possible to get another process destined for the CPU
    if( run_state->process_is_running )
    {
        // log the event
        logEvent( activity_log, run_state->currently_running_process.pid,
                  "Moved into run state",
                  stopwatch( 'x', SYSTEM_ACTION_CLOCK ) );

        // start the processing time clock
        stopwatch( 's', PROCESSING_CLOCK );
    }
    // case: the scheduler is empty
    else
    {
        // log the event
        logEvent( activity_log, SYSTEM, "No processes to run found",
                  stopwatch( 'x', SYSTEM_ACTION_CLOCK ) );
    }

    // return whether or not a context switch was made
    return context_switch_made;
}


int sendToIoQueue( PCB* process, DeviceSet* device_set,
                   ActivityLog* activity_log )
{
    // variables
    int io_enqueue_success = NO_ERRORS;  // may use in future
    int io_task_type =
        (get_listTask( &(process->job_list)  ) ).task_number;
    Device* requested_device = NULL;

    // assert a precondition that the process is not a processing one
    assert( io_task_type != PROCESS );

    // pick which device the process needs to use
    requested_device = &(device_set->device_number[io_task_type]);

    // add the device to the appropriate I/O queue where it will be handled
    // in I/O management
     insert_heap( &(requested_device->waiting_queue),
                  *process, FIFO_HEAP );

    // return the status of the operation
    return io_enqueue_success;
}


int runProcess( RunningState* run_state, ConfigData* configurations,
                ActivityLog* activity_log )
{
    // variables
    int processing_completed = false;

    // case: we have a process running on the CPU
    if( run_state->process_is_running )
    {
        // "run" the process for a cycle
        run_state->currently_running_process.job_list.tasks[ run_state->currently_running_process.job_list.iterator ].cycles_remaining--;

        // case: the running process finished its current task
        if( run_state->currently_running_process.job_list.tasks[ run_state->currently_running_process.job_list.iterator ].cycles_remaining <= 0 )
        {
            // set the flag to indicate that the current spurt of processing
            // completed
            processing_completed = true;

            // move the process to its next task
            run_state->currently_running_process.job_list.iterator++;
        }

        // simulate the process actually running with a time delay
        usleep( configurations->cpu_cycle_time );

        // update the total time the entire process has run for
        run_state->currently_running_process.total_process_time +=
            configurations->cpu_cycle_time;
    }

    // return the processing completed flag
    return processing_completed;
}


