#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define null 0
#define STD_STR_LEN 20

#define BODIES_ARG "-bodies"
#define TIME_STEP_ARG "-timestep"
#define NUM_STEPS_ARG "-steps"
#define DATA_FILE_NAME_ARG "-datafile"
#define OUTPUT_FILE_NAME_ARG "-outfile"

#define G 1.0
#define epsilon 0.000001

typedef struct
{
	int num_bodies;
	int num_time_steps;
	float timestep_magnitude;
	char* data_file_name;
	char* outfile_name;
} ProgramParams;

typedef struct
{
	float val[3];
} Vector3;

typedef struct
{
	float mass;
	Vector3 position;
	Vector3 velocity;
} Body;



void parseArgs(int argc, char** argv);

Body* readDataFile();

Body readBody(FILE* _data_file);

void conductExperiment(Body* _bodies);

void displayBody(Body* _body);

Vector3 calcForceOnBody(Body* _reference_body, Body* _other_body);

void accelerateBody(Body* _body, Vector3* force);

void moveBody(Body* _body);

void recordSnapshot(FILE* _outfile, int timestep, Body* _bodies);


ProgramParams gProgramParams;


int main(int argc, char** argv) {
	parseArgs(argc, argv);

	Body* bodies = readDataFile();

	if (bodies != null) {
		conductExperiment(bodies);
	}


	return 0;
}


void parseArgs(int argc, char** argv) {
	char* arg;
	char* argname = malloc(STD_STR_LEN);
	char* value;

	puts("RUNNING PROGRAM WITH:");

	int i;
	for (i = 1; i < argc; i++) {
		arg = argv[i];
		strncpy(argname, arg, strcspn(arg, "="));
		argname[strcspn(arg, "=")] = '\0';
		value = strrchr(arg, '=') + 1;
		printf("%s=%s\n", argname, value);

		if (strcmp(argname, BODIES_ARG) == 0) {
			gProgramParams.num_bodies = atoi(value);
		} else if (strcmp(argname, TIME_STEP_ARG) == 0) {
			gProgramParams.timestep_magnitude = atof(value);
		} else if (strcmp(argname, NUM_STEPS_ARG) == 0) {
			gProgramParams.num_time_steps = atoi(value);
		} else if (strcmp(argname, DATA_FILE_NAME_ARG) == 0) {
			gProgramParams.data_file_name = value;
		} else if (strcmp(argname, OUTPUT_FILE_NAME_ARG) == 0) {
			gProgramParams.outfile_name = value;
		}
	}

	free(argname);
}


Body* readDataFile() {
	Body* bodies = null;
	FILE* data_file = fopen(gProgramParams.data_file_name, "r");

	if (data_file != null) {
		bodies = malloc(gProgramParams.num_bodies * sizeof(Body));

		int i;
		for (i = 0; i < gProgramParams.num_bodies; i++) {
			bodies[i] = readBody(data_file);
		}

		fclose(data_file);
	} else {
		puts("BAD FILE THERE, GUY!");
	}

	return bodies;
}


Body readBody(FILE* _data_file) {
	Body new_body;

	if (_data_file != null) {
		fscanf(_data_file, " mass: %f", &(new_body.mass));

		fscanf(
			_data_file,
			" position: %f %f %f",
			&(new_body.position.val[0]),
			&(new_body.position.val[1]),
			&(new_body.position.val[2])
		);

		fscanf(
			_data_file,
			" velocity: %f %f %f",
			&(new_body.velocity.val[0]),
			&(new_body.velocity.val[1]),
			&(new_body.velocity.val[2])
		);
	}

	return new_body;
}


void conductExperiment(Body* _bodies) {
	FILE* outfile = fopen(gProgramParams.outfile_name, "w");

	int t;
	int i;
	int j;
	Vector3 force;
	for (t = 0; t < gProgramParams.num_time_steps; t++) {

		recordSnapshot(outfile, t, _bodies);

		for (i = 0; i < gProgramParams.num_bodies; i++) {
			for (j = 0; j < gProgramParams.num_bodies; j++) {
				if (i != j) {
					force = calcForceOnBody(&_bodies[i], &_bodies[j]);
					accelerateBody(&_bodies[i], &force);

					// printf("%d on %d\n", i, j);
				}
			}
		}

		for (i = 0; i < gProgramParams.num_bodies; i++) {
			moveBody(&_bodies[i]);
		}
	}

	fclose(outfile);
}


void displayBody(Body* _body) {
	if (_body != null) {
		puts("");
		printf("Mass:    \t%f\n", _body->mass);
		printf(
			"Position:\t%f\t%f\t%f\n",
			_body->position.val[0],
			_body->position.val[1],
			_body->position.val[2]
		);
		printf(
			"Velocity:\t%f\t%f\t%f\n",
			_body->velocity.val[0],
			_body->velocity.val[1],
			_body->velocity.val[2]
		);
		puts("");
	} else {
		puts("THERE'S NO BODY HERE!");
	}
}


Vector3 calcForceOnBody(Body* _reference_body, Body* _other_body) {
	Vector3 _attractive_force;
	float radius;

	int i;
	for (i = 0; i < 3; i++) {
		radius =
			_other_body->position.val[i] -
			_reference_body->position.val[i]; 

		if (abs(radius) > epsilon) {
			_attractive_force.val[i] =
				(G * _reference_body->mass * _other_body->mass) /
				(radius * radius);
			_attractive_force.val[i] *= radius > 0 ? 1 : -1; 
		} else {
			// force is too large, don't make bodies come together more,
			// ignore it
			_attractive_force.val[i] = 0;
		}
	}

	return _attractive_force;
}


void accelerateBody(Body* _body, Vector3* force) {
	int i;
	for (i = 0; i < 3; i++) {
		_body->velocity.val[i] +=
			(force->val[i] / _body->mass) *
			gProgramParams.timestep_magnitude;
	}
}


void moveBody(Body* _body) {
	int i;
	for (i = 0; i < 3; i++) {
		_body->position.val[i] +=
			_body->velocity.val[i] * gProgramParams.timestep_magnitude;
	}
}


void recordSnapshot(FILE* _outfile, int timestep, Body* _bodies) {
	fprintf(_outfile, "========\n");
	fprintf(_outfile, "TIME %d:\n", timestep);
	fprintf(_outfile, "========\n\n");

	int i;
	for (i = 0; i < gProgramParams.num_bodies; i++) {
		if (&_bodies[i] != null) {
			fprintf(_outfile, "Body %d\n", i);
			fprintf(_outfile, "---------\n");
			fprintf(_outfile, "Mass:    \t%f\n", _bodies[i].mass);
			fprintf(
				_outfile,
				"Position:\t%f\t%f\t%f\n",
				_bodies[i].position.val[0],
				_bodies[i].position.val[1],
				_bodies[i].position.val[2]
			);
			fprintf(
				_outfile,
				"Velocity:\t%f\t%f\t%f\n",
				_bodies[i].velocity.val[0],
				_bodies[i].velocity.val[1],
				_bodies[i].velocity.val[2]
			);
			fprintf(_outfile, "\n");
		} else {
			puts("THERE'S NO BODY HERE!");
		}
	}

	fprintf(_outfile, "============================\n\n");
}