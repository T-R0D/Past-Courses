#ifndef _CONFIGURATIONUTILS_H_
#define _CONFIGURATIONUTILS_H_

/* Certainly the read statements could be modified to be more flexible,
   but having them read set values helps to explicitly define a "contract." */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Constants.h"

#define RR 0
#define FIFO 1
#define Fixed 2
#define BOTH 3
#define LOG_TO_MONITOR 4
#define LOG_TO_FILE 5

//struct to store all variables read in from configuration file
typedef struct
{
	char* version;
	int mCyclesPerQuantum;
	int mCpuScheduleMode;
	char* mFilePath;
	int mCpuCycleTime;
	int mMonitorDisplayTime;
	int mHardDriveCycleTime;
	int mPrinterCycleTime;
	int mKeyboardCycleTime;
	int mMemoryType;
	int log;
} ConfigData;


int configureSystem(
	ConfigData** pConfigurations, char* pConfigurationsFilename) {

    int configuration_success = NO_ERRORS;
	char buffer[100];
	FILE *fp;
	fp = fopen( pConfigurationsFilename, "r");

	if (fp != NULL) {
		*pConfigurations = (ConfigData*) malloc(sizeof(ConfigData));

		fscanf(fp, "%[^\n]\n", buffer);

		(*pConfigurations)->version = (char*) malloc(STD_STR_LEN * sizeof(char));
		fscanf(fp, "Version: %s\n", (*pConfigurations)->version);

		fscanf(fp, "Quantum(cycles): %d\n", &((*pConfigurations)->mCyclesPerQuantum));
		
		fscanf(fp, "Processor Scheduling: %s\n", buffer);
		if(strcmp(buffer, "RR") == 0) { 
			(*pConfigurations)->mCpuScheduleMode = RR;
		} else if (!strcmp(buffer, "FIFO") == 0) {
			(*pConfigurations)->mCpuScheduleMode = FIFO;
		}

		(*pConfigurations)->mFilePath = (char*) malloc(BIG_STR_LEN * sizeof(char));
		fscanf(fp, "File Path: %s\n", (*pConfigurations)->mFilePath);

		fscanf(fp, "Processor cycle time (msec): %d\n", &((*pConfigurations)->mCpuCycleTime));

		fscanf(fp, "Monitor display time (msec): %d\n", &((*pConfigurations)->mMonitorDisplayTime));

		fscanf(fp, "Hard drive cycle time (msec): %d\n", &((*pConfigurations)->mHardDriveCycleTime));

		fscanf(fp, "Printer cycle time (msec): %d", &((*pConfigurations)->mPrinterCycleTime));

		fscanf(fp, "Keyboard cycle time (msec): %d", &((*pConfigurations)->mKeyboardCycleTime));

		fscanf(fp, "Memory type: %s", buffer);
		if(!strcmp(buffer, "Fixed")) {
			(*pConfigurations)->mMemoryType = Fixed;
		}


		fscanf(fp, "Log: %[^\n]\n", buffer);
		if( strcmp(buffer, "Log to File") == 0 ) {
			(*pConfigurations)->log = LOG_TO_FILE;
		} else if( strcmp(buffer, "Log to Both") == 0 ) {
			(*pConfigurations)->log = BOTH;
	    } else {
			(*pConfigurations)->log = LOG_TO_MONITOR;
	    }


		fclose(fp);
	} else {
		configuration_success = BAD_FILE_ERROR;
	}

    return configuration_success;
}


void destroyConfigurations(ConfigData* pConfigurations) {
	if (pConfigurations != NULL) {
		if (pConfigurations->version != NULL) {
			free(pConfigurations->version);
		}

		if (pConfigurations->mFilePath != NULL) {
			free(pConfigurations->mFilePath);
		}

		free(pConfigurations);
		pConfigurations = NULL;
	}
}


void printConfigurations(ConfigData* configurations, FILE* pStream) {
	if (configurations != NULL) {
		char scheduler [10];
		char mMemoryType [20];
		char log_type [20];
	 
		//convert the scheduler type from stored into to char array	
		if (configurations->mCpuScheduleMode == RR) {
			strcpy (scheduler, "RR");
		}

		if (configurations->mCpuScheduleMode == FIFO) {
			strcpy ( scheduler, "FIFO" );
		}

		strcpy ( mMemoryType, "Fixed" );

		if (configurations->log == LOG_TO_FILE) {
			strcpy ( log_type, "Log to File" );
		}

		if (configurations->log == LOG_TO_MONITOR) {
			strcpy ( log_type, "Log to Monitor" );
		}
		
		if (configurations->log == BOTH) {
			strcpy ( log_type, "Log to Both" );
		}

		fprintf(pStream,
		    "==============================\n"
            "  System Configurations\n"
            "==============================\n"
		    "Version: %s\n" 
			"File path: %s\n" 
			"Quantum (mCyclesPerQuantum) %d\n" 
			"Processor Scheduling: %s\n" 
			"Processor cycle time (msec): %d\n" 
			"Monitor display time (msec): %d\n" 
			"Hard drive cycle time (msec): %d\n" 
			"Printer Cycle time (msec): %d\n" 
			"Keyboard cycle time (msec): %d\n" 
			"Memory type: %s\n" 
			"Log: %s\n" 
			"==============================\n\n",  
			configurations->version, 
			configurations->mFilePath,
			configurations->mCyclesPerQuantum, 
			scheduler, 
			configurations->mCpuCycleTime, 
			configurations->mMonitorDisplayTime,
			configurations->mHardDriveCycleTime, 
			configurations->mPrinterCycleTime, 
			configurations->mKeyboardCycleTime, 
			mMemoryType,
			log_type
        );
	}
}


// void printConfigurations(ConfigData* pConfigurations, FILE* pStream) {
// 	if (pConfigurations != NULL) {
// 		fprintf(pStream,
// 			"%s\n"
// 			"%d\n"
// 			"%d\n"
// 			"%s\n"
// 			"%d\n"
// 			"%d\n"
// 			"%d\n"
// 			"%d\n"
// 			"%d\n"
// 			"%d\n"
// 			"%d\n",
// 			pConfigurations->version,
// 			pConfigurations->mCyclesPerQuantum,
// 			pConfigurations->mCpuScheduleMode,
// 			pConfigurations->mFilePath,
// 			pConfigurations->mCpuCycleTime,
// 			pConfigurations->mMonitorDisplayTime,
// 			pConfigurations->mHardDriveCycleTime,
// 			pConfigurations->mPrinterCycleTime,
// 			pConfigurations->mKeyboardCycleTime,
// 			pConfigurations->mMemoryType,
// 			pConfigurations->log
// 		);
// 	} else {
// 		fprintf(pStream, "Unconfigured...\n");
// 	}
// }

#endif     // #ifndef _CONFIGURATIONUTILS_H_
