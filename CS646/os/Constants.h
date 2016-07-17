/*

The purpose of this file is to have all pre-processor directives,
macros, constants, etc. in one place to prevent re-definition.

*/


#define bug printf("line: %i, file: %s\n", __LINE__, __FILE__);
#define bp char breakPoint;printf("stopped @ line: %i, in file: %s\n", __LINE__, __FILE__);scanf("%c", &breakPoint);


/*==============================================================================
    GLOBAL CONSTANTS
==============================================================================*/

// standard constants
#define false 0
#define true 1
#define null NULL
#define NO_ERRORS 0
#define ERROR 1
#define BAD_FILE_ERROR 5
#define NON_DESCRIPT_ERROR 10
#define STD_STR_LEN 55
#define BIG_STR_LEN 100
#define ERROR_NUM -666

// timers
#define GLOBAL_CLOCK 0
#define SYSTEM_ACTION_CLOCK 1
#define IO_TIMER 2
#define BOOT_CLOCK 3
#define PROCESSING_CLOCK 4

// device enumeration
#define HARDDRIVE_READ 0
#define HARDDRIVE_WRITE 1
#define KEYBOARD 2
#define MONITOR 3
#define PRINTER 4
#define UNKNOWN 6
#define MAX_IO_DEVICES 5
#define CPU 99             // these two are same thing
#define PROCESS 99         // have both for readability (hopefully)
#define PROCESS_END 100

// I/O related constants
#define IO_MANAGEMENT_TIME 200

// configurations
#define RR 0
#define FIFO 1
#define FIXED 2
#define BOTH 3
#define LOG_TO_MONITOR 4
#define LOG_TO_FILE 5

// other important things
#define SYSTEM 0

// SchedulingQueue related constants
#define FIFO_SchedulingQueue 0

// priorities
#define DEFAULT_PRIORITY 1

// ActivityLog related constants
#define MAX_LOG_ITEMS 10000
#define DESCRIPTION_LEN 100
#define LOG_SUCCESS 0
#define LOG_FAIL 1

// logging constants (which destination)
#define BOTH 3
#define LOG_TO_MONITOR 4
#define LOG_TO_FILE 5
