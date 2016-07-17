/*==============================================================================
    HEADER FILES
==============================================================================*/
// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <pthread.h>

// constants
#include "Constants.h"

// special functions
#include "ConfigurationUtils.h"
#include "MetaDataUtils.h"
#include "SimulatorIo.h"
#include "MyStopwatch.h"

// containers
#include "ArrayList.h"
#include "SchedulingQueue.h"
#include "ActivityLog.h"

// objects
#include "Pcb.h"


/*==============================================================================
    GLOBAL CONSTANTS
==============================================================================*/


/*==============================================================================
    USER DEFINED TYPES
==============================================================================*/


// This could have been done just as an array, but putting it in a struct
// will be helpful in future expansion of the simulator.
typedef struct {
    int mIoStillRunning;
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
    int mProcessIsRunning;
    Pcb* mCurrentlyRunningProcess;
} RunningState;


/*==============================================================================
    GLOBAL VARIABLES
==============================================================================*/

ConfigData* gSystemConfigurations = NULL;
ActivityLog* gSystemLog = NULL;

/*==============================================================================
    FUNCTION PROTOTYPES
==============================================================================*/
int bootSystem( ConfigData** pSystemConfigurations, DeviceSet* pDevices,
    SchedulingQueue** pReadyQueue, char* pConfigFilename, char* pMetadataFilename
);

int executeSimulation(SchedulingQueue* pReadyQueue, DeviceSet* pDeviceSet);

int manageIoInterrupts(SchedulingQueue* pReadyQueue, DeviceSet* pDeviceSet);

int manageIoDevice(Device* pDevice, SchedulingQueue* pReadyQue);

int computeIoRunningTime(Task* pTask);

int manageContextSwitching(
    RunningState* pRunState, SchedulingQueue* pReadyQueue,
    DeviceSet* pDeviceSet
);

int sendToIoQueue(Pcb* pProcess, DeviceSet* pDeviceSet);

int runProcess(RunningState* pRunState, int* pActualTimeProcessed);

void shutdownSystem(int pSystemStatus, ConfigData* pConfigurations,
    SchedulingQueue* pSchedulingQueue);


/*==============================================================================
    DRIVER PROGRAM
==============================================================================*/
int main(int argc, char** argv) {
    stopwatch('s', GLOBAL_CLOCK);

    int systemStatus = NO_ERRORS;
    SchedulingQueue* readyQueue = NULL;
    DeviceSet deviceSet;

    // perform system bootup actions (which does actually include the
    // initialization of the above the items)
    systemStatus = bootSystem(&gSystemConfigurations, &deviceSet, &readyQueue, argv[1],
                                argv[2]);

    if (systemStatus != NO_ERRORS) {
        char errorMessage[BIG_STR_LEN];
        sprintf(errorMessage, "System boot failed, error code %d",
                systemStatus);
        logEvent(gSystemLog, SYSTEM, errorMessage,
                 stopwatch('x', GLOBAL_CLOCK));
    } else {
        systemStatus = executeSimulation(readyQueue, &deviceSet);
    }

    shutdownSystem(systemStatus, gSystemConfigurations, readyQueue);

    return systemStatus;
}


/*==============================================================================
    FUNCTION IMPLEMENTATIONS
==============================================================================*/
int bootSystem( ConfigData** pConfigurations, DeviceSet* pDevices, SchedulingQueue** pReadyQueue,
                char* pConfigFilename, char* pMetadataFilename) { 

    int systemBootStatus = NO_ERRORS;
    stopwatch('s', BOOT_CLOCK);

    gSystemLog = (ActivityLog*) malloc(sizeof(ActivityLog));

    logEvent(gSystemLog, SYSTEM, "System starting...",
                 stopwatch('x', BOOT_CLOCK));

    systemBootStatus = configureSystem(pConfigurations,
                                                 pConfigFilename);

    if (systemBootStatus != NO_ERRORS) {
        logEvent(gSystemLog, SYSTEM, "Unable to configure system.",
                 stopwatch('x', BOOT_CLOCK));
        return systemBootStatus;
    }

    logEvent(gSystemLog, SYSTEM, "System configured",
             stopwatch('x', BOOT_CLOCK));

    int device_port_num;
    for( device_port_num = 0; device_port_num < MAX_IO_DEVICES;
             device_port_num++ ) {
        systemBootStatus =
        connectToDevice(&(pDevices->device_number[device_port_num]),
                                 device_port_num);
    }

    if (systemBootStatus != NO_ERRORS) {
        logEvent(gSystemLog, SYSTEM, "Error initializing peripheral devices.",
                 stopwatch('x', BOOT_CLOCK));
        return systemBootStatus;
    }

    logEvent(gSystemLog, SYSTEM, "Peripheral devices initialized",
              stopwatch( 'x', BOOT_CLOCK ) );


    *pReadyQueue = SchedulingQueueFactory(
        fifoCompare,
        destroyPcbInSchedulingQueue
    );

    systemBootStatus = readMetaData(*pReadyQueue, pMetadataFilename, gSystemLog);

    if (systemBootStatus != NO_ERRORS) {
        logEvent(gSystemLog, SYSTEM, "Unable to process meta-data.",
                 stopwatch('x', BOOT_CLOCK));
        return systemBootStatus;
    }

    logEvent(gSystemLog, SYSTEM, "System fully booted",
             stopwatch('x', BOOT_CLOCK));

    return systemBootStatus;
}


int executeSimulation(SchedulingQueue* pReadyQueue, DeviceSet* pDeviceSet) {
    int executionStatus = NO_ERRORS;
    RunningState runState;
        runState.mProcessIsRunning = false;  // just for initial context switch
        runState.mCurrentlyRunningProcess = NULL;
    int cyclesRemainingInQuantum = gSystemConfigurations->mCyclesPerQuantum;
    int processingFinished = true;  // since nothing has run yet
                                     // processing is "finished" and a
                                     // context switch is necessary
    pDeviceSet->mIoStillRunning = false;

    while (runState.mProcessIsRunning ||
           isSqNotEmpty(pReadyQueue) ||
           pDeviceSet->mIoStillRunning ) {

puts("              ---start of quantum cycle---");

        manageIoInterrupts(pReadyQueue, pDeviceSet);

        if(cyclesRemainingInQuantum <= 0 || processingFinished) {
            manageContextSwitching( &runState, pReadyQueue, pDeviceSet);

            cyclesRemainingInQuantum = gSystemConfigurations->mCyclesPerQuantum;
            processingFinished = false;
        }

        if (runState.mProcessIsRunning) {
            processingFinished = runProcess(&runState, NULL);
        }

        cyclesRemainingInQuantum--;

// if (runState.mProcessIsRunning) {
// printTask(getCurrentTask(runState.mCurrentlyRunningProcess), stdout);
// }

    }

    return executionStatus;
}


int manageIoInterrupts( SchedulingQueue* pReadyQueue, DeviceSet* pDevices) {
    int device_management_status = NO_ERRORS; // for future improvement
    int ioRunningFlags[MAX_IO_DEVICES];
    int ioIsStillRunning = false;

    int i;
    for (i = 0; i < MAX_IO_DEVICES; i++) {
        Device* currentDevice = &(pDevices->device_number[i]);
        manageIoDevice(currentDevice, pReadyQueue);
        ioRunningFlags[i] = currentDevice->data.in_use;
    }

    for (i = 0; i < MAX_IO_DEVICES; i++) {
        if (ioRunningFlags[i]) {
            ioIsStillRunning = true;
            break;
        }
    }

    if (ioIsStillRunning) {
        pDevices->mIoStillRunning = true;
    } else {
        pDevices->mIoStillRunning = false;
    }

    return device_management_status;
}


int manageIoDevice(Device* pDevice, SchedulingQueue* pReadyQueue) {
    int device_management_success = NO_ERRORS;
    char action_message[DESCRIPTION_LEN];

    if (pDevice->data.interrupt_flag) {
        stopwatch( 's', IO_TIMER );

        pthread_join(pDevice->data.thread_id, NULL );

        pDevice->data.interrupt_flag = false;
        pDevice->data.in_use = false;

        advanceToNextProcessTask(pDevice->working_process);

        if (processHasNotFinished(pDevice->working_process)) {
            addSqItem(pReadyQueue, pDevice->working_process);

            sprintf(action_message,
                "%s action completed - process placed back in main scheduler",
                pDevice->name
            );
            logEvent(gSystemLog,
                pDevice->working_process->mPid,
                action_message,
                stopwatch('x', IO_TIMER)
            );
        } else {
            sprintf(action_message,
                "%s - process completed",
                pDevice->name
            );
            destroyPcb(pDevice->working_process);
            logEvent(gSystemLog,
                pDevice->working_process->mPid,
                action_message,
                stopwatch( 'x', GLOBAL_CLOCK )
            );
        }

    } else if(pDevice->data.in_use) {
        stopwatch( 's', IO_TIMER );

        usleep(IO_MANAGEMENT_TIME);

        sprintf(action_message,
            "I/O maintnenance: %s still in use by process %d",
            pDevice->name, (int) pDevice->working_process->mPid
        );
        logEvent(gSystemLog,
            SYSTEM,
            action_message,
            stopwatch('x', IO_TIMER)
        );
    }

    // case: an I/O process is wating to run and the I/O device is free
    //       (it is possible that the device was freed previously)
    if(!(pDevice->data.in_use) && isSqNotEmpty(pDevice->waiting_queue)) {
        stopwatch( 's', IO_TIMER ); 

        pDevice->working_process = (Pcb*) popNextSqItem(pDevice->waiting_queue);
        Task* currentTask = getCurrentTask(pDevice->working_process);

        // startup the independent I/O action (simulated of course)
        pDevice->data.usec_to_run =  computeIoRunningTime(currentTask);
        pDevice->data.interrupt_flag = false;
        pDevice->data.in_use = true;


printf("adding process %i for %i I/O, for %i cycles/%i time units\n",
    (int) pDevice->working_process->mPid,
    currentTask->mTaskType,
    currentTask->mRemainingCycles,
    computeIoRunningTime(currentTask)
);

        pthread_attr_init( &(pDevice->data.attribute) );
        pthread_create( &(pDevice->data.thread_id), &(pDevice->data.attribute),
                        conductIoProcess, (void*) &(pDevice->data) );

        sprintf(action_message, "Starting action on %s", pDevice->name);
        logEvent(gSystemLog,
            pDevice->working_process->mPid,
            action_message,
            stopwatch( 'x', IO_TIMER )
        );
    }

    return device_management_success;
}


int computeIoRunningTime(Task* pTask) {
    int cycles_required_for_task = pTask->mRemainingCycles;
    int time_per_cycle = 0;

    switch (pTask->mTaskType) {
        case HARDDRIVE_READ:
        case HARDDRIVE_WRITE:
            time_per_cycle = gSystemConfigurations->mHardDriveCycleTime;
            break;
        case KEYBOARD:
            time_per_cycle = gSystemConfigurations->mKeyboardCycleTime;
            break;
        case MONITOR:
            time_per_cycle = gSystemConfigurations->mMonitorDisplayTime;
            break;
        case PRINTER:
            time_per_cycle = gSystemConfigurations->mPrinterCycleTime;
            break;
        default: // for custom or unrecognized hardware
            time_per_cycle = 200;
            break;
    }

    return (cycles_required_for_task * time_per_cycle);
}


int manageContextSwitching(RunningState* pRunState, SchedulingQueue* pReadyQueue,
                            DeviceSet* pDeviceSet) {

    int contextSwitchStatus = NO_ERRORS;

    if (pRunState->mProcessIsRunning) {
        stopwatch('s', SYSTEM_ACTION_CLOCK );

        if (processHasNotFinished(pRunState->mCurrentlyRunningProcess)) {

            addSqItem(pReadyQueue, pRunState->mCurrentlyRunningProcess);
            pRunState->mProcessIsRunning = false;

            logEvent(gSystemLog,
                pRunState->mCurrentlyRunningProcess->mPid,
                "Completed processing quantum and process placed in ready queue",
                stopwatch('x', SYSTEM_ACTION_CLOCK)
            );

        } else {
            logEvent(gSystemLog,
                pRunState->mCurrentlyRunningProcess->mPid,
                "Ran to completion (time is global system time required)~~~",
                stopwatch('x', GLOBAL_CLOCK)
            );
            destroyPcb(pRunState->mCurrentlyRunningProcess);
            pRunState->mProcessIsRunning = false;
        }
    }

    stopwatch('s', SYSTEM_ACTION_CLOCK);

    if (isSqNotEmpty(pReadyQueue)) {

        pRunState->mCurrentlyRunningProcess = (Pcb*) popNextSqItem(pReadyQueue);
        pRunState->mProcessIsRunning = true;

        while (pRunState->mProcessIsRunning &&  // HEEEEEEEEEEEEEEERRRRRRRRRRRRREEEEEEEEEEEEEEEE
               (getCurrentTask(pRunState->mCurrentlyRunningProcess)->mTaskType != PROCESS)) {

            pRunState->mProcessIsRunning = false;

            sendToIoQueue(pRunState->mCurrentlyRunningProcess, pDeviceSet);

            if (isSqNotEmpty(pReadyQueue)) {
                pRunState->mCurrentlyRunningProcess = (Pcb*) popNextSqItem(pReadyQueue);
                pRunState->mProcessIsRunning = true;
            }
        }
    }

    if (pRunState->mProcessIsRunning) {
        logEvent(gSystemLog,
            pRunState->mCurrentlyRunningProcess->mPid,
            "Moved into run state",
            stopwatch('x', SYSTEM_ACTION_CLOCK)
        );

        stopwatch( 's', PROCESSING_CLOCK );
    } else {
        logEvent(gSystemLog,
            SYSTEM,
            "No processes to run found",
            stopwatch('x', SYSTEM_ACTION_CLOCK)
        );
    }

    return contextSwitchStatus;
}


int sendToIoQueue(Pcb* pProcess, DeviceSet* pDeviceSet) {

    int ioTransferStatus = NO_ERRORS;
    int io_task_type = getCurrentTask(pProcess)->mTaskType;
    
    if (io_task_type == PROCESS) {
puts("trying to IO a PROCESS task");
        exit(ERROR);
    }

    Device* requestedDevice = &(pDeviceSet->device_number[io_task_type]);
    addSqItem(requestedDevice->waiting_queue, (void*) pProcess);

    return ioTransferStatus;
}


int runProcess(RunningState* pRunState, int* pActualTimeProcessed) {
    int processingCompleted = false;

    if (pRunState->mProcessIsRunning) {
        Pcb* runningProcess = pRunState->mCurrentlyRunningProcess;
        Task* currentTask = getCurrentTask(runningProcess);

        if (currentTask->mTaskType != PROCESS) {
puts("tried to run non-PROCESS task");
printTask(currentTask, stdout);

            exit(0);
        }

        currentTask->mRemainingCycles--;

        if (currentTask->mRemainingCycles <= 0) {
            processingCompleted = true;
            advanceToNextProcessTask(runningProcess);

            if (processHasFinished(runningProcess)) {   // HHHHHEEEEEEEEEEEEEEEEEEERRRRRRRRRRRREEEEEEEEEEEE
                logEvent(gSystemLog,
                    pRunState->mCurrentlyRunningProcess->mPid,
                    "Ran to completion (time is global system time required)~~~",
                    stopwatch('x', GLOBAL_CLOCK)
                );

                pRunState->mProcessIsRunning = false;
                destroyPcb(pRunState->mCurrentlyRunningProcess);
            }
        }

        runningProcess->mTotalProcessTime +=
            gSystemConfigurations->mCpuCycleTime;
    }

    return processingCompleted;
}


void shutdownSystem(int pSystemStatus, ConfigData* pConfigurations,
    SchedulingQueue* pSchedulingQueue) {

    if (pSystemStatus == NO_ERRORS) {
        logEvent(gSystemLog,
            SYSTEM,
            "All processes completed with no errors, shutting down...",
            stopwatch('x', GLOBAL_CLOCK)
        );
    } else {
        logEvent(gSystemLog,
            SYSTEM,
            "Error encountered, shutting down to prevent damage...",
            stopwatch('x', GLOBAL_CLOCK)
        );

    }

    if (pConfigurations->log == BOTH ||
        pConfigurations->log == LOG_TO_FILE) {
        FILE* logFile = fopen("SYSTEM_RUN_ACTIVITY.txt", "w"); // arbitrary file name
        dumpActivityData(gSystemLog, logFile, gSystemConfigurations);
        fclose(logFile);
    }

    if (pConfigurations->log == BOTH ||
        pConfigurations->log == LOG_TO_MONITOR) {
        dumpActivityData(gSystemLog, stdout, gSystemConfigurations);
    }    




    dumpActivityData(gSystemLog, stdout, gSystemConfigurations);

    destroyConfigurations(pConfigurations);
    destroySchedulingQueue(pSchedulingQueue);

    // TODO: deconstruct everything

}