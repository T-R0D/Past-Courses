// standard header
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>

// special functions
#include "simulator_io.h"
#include "my_stopwatch.h"
#include "List.h"
#include "Heap.h"
#include "PCB.h"

// data structures
#include "activity_log.h"

//global variables
#define STD_STR_LEN 50

//struct to store all variables read in from config file
typedef struct
{
	double version;
	int cycles;
	char cpu_scheduler[STD_STR_LEN];
	char file_path[STD_STR_LEN];
	int cpu_cycle_time;
	int monitor_display_time;
	int hd_cycle_time;
	int printer_cycle_time;
	int keyboard_cycle_time;
	char memory_type[STD_STR_LEN];
	char log[STD_STR_LEN];
} ConfigData;


int configureSystem( ConfigData* configurations, Heap* main_scheduler,
                     ActivityLog* activity_log )
{
	//buffer
	char buffer[STD_STR_LEN];
    ConfigData configurations;

	FILE *fp;
	printf("Enter the file name to use: ");
	scanf("%s", buffer);
	fp = fopen(buffer, "r");

	//read in data to struct members
	fscanf(fp, "%[^\n]\n", buffer);
	printf("%s\n", buffer);
	fscanf(fp, "%*s %lf", &configurations.version);

	fscanf(fp, "%*s %d", &configurations.cycles);
	
	fscanf(fp, "%*s %*s %s", configurations.cpu_scheduler);
	
	fscanf(fp, "%*s %*s %s", configurations.file_path);

	fscanf(fp, "%*s %*s %*s %*s %d", &configurations.cpu_cycle_time);

	fscanf(fp, "%*s %*s %*s %*s %d", &configurations.monitor_display_time);

	fscanf(fp, "%*s %*s %*s %*s %*s %d", &configurations.hd_cycle_time);

	fscanf(fp, "%*s %*s %*s %*s %d", &configurations.printer_cycle_time);

	fscanf(fp, "%*s %*s %*s %*s %d", &configurations.keyboard_cycle_time);

	fscanf(fp, "%*s %*s %s", configurations.memory_type);

	fscanf(fp, "%*s %[^\n]\n", configurations.log);
	
	//print to screen just to ensure correct storage of variables - can be removed later.
	printf("%lf\n" "%d\n" "%s\n" "%s\n" "%d\n" "%d\n" "%d\n" "%d\n" "%d\n" "%s\n" "%s\n", configurations.version, configurations.cycles, configurations.cpu_scheduler, configurations.file_path, configurations.cpu_cycle_time, configurations.monitor_display_time, configurations.hd_cycle_time, configurations.printer_cycle_time, configurations.keyboard_cycle_time, configurations.memory_type, configurations.log);

	fscanf(fp, "%[^\n]\n", buffer);
	printf("%s\n", buffer);
};
/*
void writeConfigurationData( ConfigData* configurations, FILE* log_file )
{
    // variables
    char readable_config_info[200];

    // prepare the configuration output
    sprintf( readable_config_info,
        "Version: %s\n"
        "Quantum (cycles): %d\n"
        "Processor Scheduling: %s\n"
        "File Path: %s\n"
        "Processor Cycle Time (usec): %d\n"
        "Monitor/Display Cycle Time (usec): %d\n"
        "Hard Drive Cycle Time (usec): %d\n"
        "Printer Cycle Time (usec): %d\n"
        "Keyboard Cycle Time (usec): %d\n"
        "Memory Type: %s\n"
        "Log Destination: %s\n\n",
        configurations->version,
        configurations->cycles,
        configurations->cpu_scheduler,
        configurations->file_path,
        configurations->cpu_cycle_time,
        configurations->monitor_display_time,
        configurations->hd_cycle_time,
        configurations->printer_cycle_time,
        configurations->keyboard_cycle_time,
        configurations->memory_type,
        configurations->log );

    // case: we are logging to a file
    if( configurations->log == file || both )  // <---------------!!!!!!!
    {
        // write to the file
        fprintf( log_file, readable_config_info );
    }

    // case: we are logging to the screen
    if( configurations->log == screen || both )  // <---------------!!!!!!!
    {
        // write to the file
        printf( readable_config_info );
    }
}*/
