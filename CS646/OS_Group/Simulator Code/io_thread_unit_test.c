#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include "simulator_io_thread.h"

int main( int argc, char** argv )
{
  // variables
  IoData test_data;
    test_data.interrupt_flag = true;
    test_data.usec_to_run = 5000000;
  pthread_t thread_id;
  pthread_attr_t attribute;

printf( "In main process: flag = %d\r\n", (test_data.interrupt_flag) );

  // run the io thread
  pthread_attr_init( &attribute );
  pthread_create( &thread_id, &attribute, conductIoProcess,
                  (void*) &test_data );

usleep( 1000000 );
printf( "In main process: flag = %d\r\n", (test_data.interrupt_flag) );


while( test_data.interrupt_flag != true )
{
printf( "Hey!\n" );
printf( "In main process: flag = %d\n", test_data.interrupt_flag );

usleep( 3000000 );
}

printf( "In main process: flag = %d\r\n", test_data.interrupt_flag );

  // join the thread
  pthread_join( thread_id, NULL );

printf( "In main process: flag = %d\r\n", test_data.interrupt_flag );

  return 0;
}

