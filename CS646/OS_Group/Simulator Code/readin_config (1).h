#ifndef _READIN_CONFIG_H_
#define _READIN_CONFIG_H_

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include "preprocessor_directives.h"

#define RR 0
#define FIFO 1
#define Fixed 2
#define BOTH 3
#define LOG_TO_MONITOR 4
#define LOG_TO_FILE 5

//struct to store all variables read in from config file
typedef struct
{
	char version[STD_STR_LEN];
	int cycles;
	int cpu_scheduler;
	char file_path[STD_STR_LEN];
	int cpu_cycle_time;
	int monitor_display_time;
	int hd_cycle_time;
	int printer_cycle_time;
	int keyboard_cycle_time;
	int memory_type;
	int log;
} config;

//program execution
void readin_config()
{
	config values;
	//buffer
	char buffer[STD_STR_LEN];

	FILE *fp;
	printf("Enter the configuration file name to use: ");
	scanf("%s", buffer);
	fp = fopen(buffer, "r");

	//read in data to struct members
	fscanf(fp, "%[^\n]\n", buffer);
	printf("%s\n", buffer);
	fscanf(fp, "%*s %s", values.version);

	fscanf(fp, "%*s %d", &values.cycles);
	
	fscanf(fp, "%*s %*s %s", buffer);
	if(!strcmp(buffer, "RR"))
	{
		values.cpu_scheduler = RR;
	}
	if(!strcmp(buffer, "FIFO"))
	{
		values.cpu_scheduler = FIFO;
	}
	
	fscanf(fp, "%*s %*s %s", values.file_path);

	fscanf(fp, "%*s %*s %*s %*s %d", &values.cpu_cycle_time);

	fscanf(fp, "%*s %*s %*s %*s %d", &values.monitor_display_time);

	fscanf(fp, "%*s %*s %*s %*s %*s %d", &values.hd_cycle_time);

	fscanf(fp, "%*s %*s %*s %*s %d", &values.printer_cycle_time);

	fscanf(fp, "%*s %*s %*s %*s %d", &values.keyboard_cycle_time);

	fscanf(fp, "%*s %*s %s", buffer);
	if(!strcmp(buffer, "Fixed"))
	{
		values.memory_type = Fixed;
	}

	fscanf(fp, "%*s %[^\n]\n", buffer);
	if(!strcmp(buffer, "Log to Both"))
	{
		values.log = BOTH;
	}
	
	//print to screen just to ensure correct storage of variables - can be removed later.
	printf("%s\n" "%d\n" "%d\n" "%s\n" "%d\n" "%d\n" "%d\n" "%d\n" "%d\n" "%d\n" "%d\n", values.version, values.cycles, values.cpu_scheduler, values.file_path, values.cpu_cycle_time, values.monitor_display_time, values.hd_cycle_time, values.printer_cycle_time, values.keyboard_cycle_time, values.memory_type, values.log);

	fscanf(fp, "%[^\n]\n", buffer);
	printf("%s\n", buffer);
	fclose(fp);
}

#endif
