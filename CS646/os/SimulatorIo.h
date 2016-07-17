/**
    @file simulator_io_thread.h

    @author Terence Henriod

    @brief Defines a thread function that can be used to simulate a wait for
           I/O completion, followed by an interrupt.   

    @version Original Code 1.00 (3/29/2014) - T. Henriod

    UNOFFICIALLY:
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef __SIMULATORIO_H__
#define __SIMULATORIO_H__

/*==============================================================================
=======     HEADER FILES     ===================================================
==============================================================================*/
#include <assert.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include "Pcb.h"
#include "ArrayList.h"
#include "SchedulingQueue.h"


/*==============================================================================
=======     CONSTANTS / MACROS     =============================================
==============================================================================*/



/*==============================================================================
=======     USER-DEFINED TYPES     =============================================
==============================================================================*/
/**
IoData

Contains all of the data required to simulate an independent I/O process and
interrupt.

@var interrupt_flag   The flag that will indicate to the main driver if an
                      interrupt has occurred. (By reference to work across the
                      threads)
@var usec_to_run      The number of micro-seconds that the I/O process is
                      intended to run for.
*/
typedef struct
{
    int interrupt_flag;
    int in_use;
    int usec_to_run;
    pthread_t thread_id;
    pthread_attr_t attribute;
} IoData;


typedef struct
{
    char name[STD_STR_LEN];
    SchedulingQueue* waiting_queue;
    Pcb* working_process;
    IoData data;
} Device;


/*==============================================================================
=======     FUNCTION PROTOTYPES     ============================================
==============================================================================*/

/**
conductIoProcess

This funciton pointer (intended for use in a thread) carries out a delay and
a flag setting operation in order to
simulate an I/O interrupt in the CS646 Operating System Simulator.

@param flag_and_time   Despite being a void pointer, this parameter shall be
                       point to a valid IoData struct that contains the time
                       the operation should run for and the flag that will
                       simulate the interrupt.

@pre
-# The given parameter must point to a valid IoData struct, otherwise, behavior
   is undefined.

@post
-# The interrupt flag in the given struct will be set to true to indicate
   that the I/O process has completed.

@detail @bAlgorithm
-# The thread function simply sleeps for the specified amount of time before
   setting the flag to indicate the "interrupt"

@code
  // variables
  IoData test_data;
    test_data.interrupt_flag = true;
    test_data.usec_to_run = 5000000;
  pthread_t thread_id;
  pthread_attr_t attribute;

  // run the io thread
  pthread_attr_init( &attribute );
  pthread_create( &thread_id, &attribute, conductIoProcess,
                  (void*) &test_data );

  // give a periodic report to show that the thread has not set the flag
  while( flag != true )
  {
  printf( "In main process: flag = %d\r\n", flag );
  usleep( 3000000 );
  }

  // join the thread
  pthread_join( thread_id, NULL );
@endcode
*/

int connectToDevice(Device* device, int which_device);

void* conductIoProcess( void* flag_and_time );

/*==============================================================================
=======     FUNCTION IMPLEMENTATIONS     =======================================
==============================================================================*/

int connectToDevice(Device* device, int which_device) {
    // variables
    int connection_success = NO_ERRORS; // unused for now, here for future
                                        // expansion

    // initialize the device so that it is on-line but not performing any
    // actions
    device->waiting_queue = SchedulingQueueFactory(
        fifoCompare,
        destroyPcbInSchedulingQueue
    );
    device->data.interrupt_flag = false;
    device->data.in_use = false;
    device->data.usec_to_run = 0;

    switch (which_device) {
    case HARDDRIVE_READ:
        strcpy(device->name, "Hard Drive Read" );
        break;
    case HARDDRIVE_WRITE:
        strcpy( device->name, "Hard Drive Write" );
        break;
    case KEYBOARD:
        strcpy( device->name, "Keyboard" );
        break;
    case MONITOR:
        strcpy( device->name, "Monitor" );
        break;
    case PRINTER:
        strcpy( device->name, "Printer" );
        break;
    default: // an unknown device (we could either do it this way or throw error,
         // I thought this was better to simulate a "modified" computer
         // where the owner installed their own hardware)
        strcpy( device->name, "UNKNOWN" );
        break;
    }

    // return device by reference, return device connection success
    return connection_success;
}

void* conductIoProcess( void* flag_and_time )
{
  // variables
  IoData* io_process = (IoData*) flag_and_time;

  // ensure interrupt flag is down and that it is known that device is in use
  io_process->interrupt_flag = false;
  io_process->in_use = true;

  // wait for the I/O time to run its course
  usleep( io_process->usec_to_run );

  // raise the interrupt flag, device will be released elsewhere
  io_process->interrupt_flag = true;

  // kill the thread
  pthread_exit(0);

  // no return - void
}

#endif
