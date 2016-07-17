#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

#define null 0
#define STD_STR_LEN 20
#define MASTER 0

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
	int my_rank;
	int total_nodes;
	int* start_indices;
	int* shares_of_bodies;
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



void CreateMPIDatatype();

void initProgram(int argc, char** argv);

void parseArgs(int argc, char** argv);

void computeShares();

Body* readDataFile();

Body readBody(FILE* _data_file);

void conductExperiment(Body* _bodies);

void displayBody(Body* _body);

Vector3 calcForceOnBody(Body* _reference_body, Body* _other_body);

void accelerateBody(Body* _body, Vector3* force);

void moveBody(Body* _body);

void recordSnapshot(FILE* _outfile, int timestep, Body* _bodies);

void bodiesToFloat(Body* _bodies, float* buffer);

void floatsToBodies(float* buffer, Body* _bodies);



ProgramParams gProgramParams;
MPI_Datatype MPI_Body;


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &(gProgramParams.total_nodes));
    MPI_Comm_rank(MPI_COMM_WORLD, &(gProgramParams.my_rank));
    CreateMPIDatatype();

    if (gProgramParams.my_rank == MASTER) {
    	puts("Initializing program...");
	}
    MPI_Barrier(MPI_COMM_WORLD);

	initProgram(argc, argv);

	Body* bodies = NULL;

	if (gProgramParams.my_rank == MASTER) {
		puts("Reading data file...");
	}

	bodies = readDataFile();

	if (gProgramParams.my_rank == MASTER) {
		puts("Conducting experiment...");
	}
	MPI_Barrier(MPI_COMM_WORLD);
	conductExperiment(bodies);

	if (gProgramParams.my_rank == MASTER) {
		puts("Writing results...");
	}

	MPI_Finalize();

	return 0;
}



void CreateMPIDatatype() {
/*
    int blocks = 3;
    
    MPI_Type_struct( 3,
// count  // array of blocklengths // array of displacements  // array of types  // newtype )
*/

    // Array of datatypes
    MPI_Datatype type[3] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};

    // Number of each type
    int block_len[3] = {1, 3, 3};
    
    // Displacements
    MPI_Aint disp[3] = {0, sizeof(MPI_FLOAT), (4 * sizeof(MPI_FLOAT))};

    // MPI Body struct
    MPI_Type_create_struct(3, block_len, disp, type, &MPI_Body);

    MPI_Type_commit(&MPI_Body);
}


void initProgram(int argc, char** argv) {
	parseArgs(argc, argv);

	computeShares();
}


void computeShares() {
	int minimum_share = 0;
    int bigger_share = 0;
    int remainder = 0;
    int i = 0;

    gProgramParams.start_indices =
    	(int*) calloc(gProgramParams.total_nodes, sizeof(int));
    gProgramParams.shares_of_bodies =
    	(int*) calloc(gProgramParams.total_nodes, sizeof(int));

    minimum_share = gProgramParams.num_bodies / gProgramParams.total_nodes;
    remainder = gProgramParams.num_bodies % gProgramParams.total_nodes;
    bigger_share = minimum_share + 1;

    for (i = 0; i < remainder; i++) {
      gProgramParams.start_indices[i] = i * bigger_share;
      gProgramParams.shares_of_bodies[i] = bigger_share;
    }

    for (/* i = whatever it was from last loop */;
         i < gProgramParams.total_nodes;
         i++) {
      gProgramParams.start_indices[i] = ( i * minimum_share ) + remainder;
      gProgramParams.shares_of_bodies[i] = minimum_share;
    }
}


void parseArgs(int argc, char** argv) {
	char* arg;
	char* argname = malloc(STD_STR_LEN);
	char* value;

	int i;
	for (i = 1; i < argc; i++) {
		arg = argv[i];
		strncpy(argname, arg, strcspn(arg, "="));
		argname[strcspn(arg, "=")] = '\0';
		value = strrchr(arg, '=') + 1;

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
	//FILE* outfile = fopen(gProgramParams.outfile_name, "w");
	float* buffer =
		(float*) malloc(gProgramParams.num_bodies * 7 * sizeof(float));
	int* starts = (int*) malloc(gProgramParams.num_bodies);
	int* counts = (int*) malloc(gProgramParams.num_bodies);
	int q;
	for (q = 0; q < gProgramParams.num_bodies; q++) {
		starts[q] = gProgramParams.start_indices[q] * 7;
		counts[q] = gProgramParams.shares_of_bodies[q] * 7;
	}

	bodiesToFloat(_bodies, buffer);
	MPI_Bcast(
		buffer,
		gProgramParams.num_bodies * 7,
		MPI_FLOAT,
		MASTER,
		MPI_COMM_WORLD
	);
	floatsToBodies(buffer, _bodies);

	int t;
	int i;
	int j;
	int my_rank = gProgramParams.my_rank;
	int start_index = gProgramParams.start_indices[my_rank];
	int stop_index = start_index + gProgramParams.shares_of_bodies[my_rank];
	Vector3 force;
	for (t = 0; t < gProgramParams.num_time_steps; t++) {
		//recordSnapshot(stdout, t, _bodies);

		for (i = start_index; i < stop_index; i++) {
			for (j = 0; j < gProgramParams.num_bodies; j++) {
				if (i != j) {
					force = calcForceOnBody(&_bodies[i], &_bodies[j]);
					accelerateBody(&_bodies[i], &force);
				}
			}
		}

		for (i = start_index; i < stop_index; i++) {
			moveBody(&_bodies[i]);
		}

		bodiesToFloat(_bodies, buffer);
	    MPI_Allgatherv(
	    	buffer + (start_index * 7), // send buffer
	    	gProgramParams.shares_of_bodies[my_rank] * 7, // send count
	    	MPI_FLOAT,                           // send type
	    	buffer,                           // receive buffer
	    	counts,                  // receive counts
	    	starts,                // displacements
	    	MPI_FLOAT,                           // receive type
	    	MPI_COMM_WORLD
	    );
	    floatsToBodies(buffer, _bodies);
	}

	free(buffer);
	buffer = NULL;
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


void bodiesToFloat(Body* _bodies, float* buffer) {
	int i;
	for (i = 0; i < (gProgramParams.num_bodies * 7); i += 7) {
		buffer[i] = _bodies[i / 7].mass;
		buffer[i + 1] = _bodies[i / 7].position.val[0];
		buffer[i + 2] = _bodies[i / 7].position.val[1];
		buffer[i + 3] = _bodies[i / 7].position.val[2];
		buffer[i + 4] = _bodies[i / 7].velocity.val[0];
		buffer[i + 5] = _bodies[i / 7].velocity.val[1];
		buffer[i + 6] = _bodies[i / 7].velocity.val[2];
	}
}


void floatsToBodies(float* buffer, Body* _bodies) {
	int i;
	for (i = 0; i < (gProgramParams.num_bodies * 7); i += 7) {
		_bodies[i / 7].mass                = buffer[i];
		_bodies[i / 7].position.val[0] = buffer[i + 1];
		_bodies[i / 7].position.val[1] = buffer[i + 2];
		_bodies[i / 7].position.val[2] = buffer[i + 3];
		_bodies[i / 7].velocity.val[0] = buffer[i + 4];
		_bodies[i / 7].velocity.val[1] = buffer[i + 5];
		_bodies[i / 7].velocity.val[2] = buffer[i + 6];
	}
}