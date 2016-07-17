#include <iostream>
#include <cstring>
#include <string>
#include <random>

#include <Eigen/Dense>

#include "sim_node_network.hpp"
#include "sim_node.hpp"

const char* DATA_FILE_ARG = "data";
const char* NUM_NODES_ARG = "nodes";
const char* SENSING_RANGE_ARG = "sense_range";
const char* COMMUNICATION_RANGE_ARG = "comm_range";
const char* FIELD_X_ARG = "x";
const char* FIELD_Y_ARG = "y";
const char* FIELD_CELLS_ARG = "cells";

const int SEED = 1;

class Configurations{
 public:
  std::string data_file_name;
  int number_of_sensor_nodes;
  double sensing_range;
  double communication_range;
  double field_x_size;
  double field_y_size;
  int field_cells;

  Configurations() {
    data_file_name = "data_files_should_have_names";
    number_of_sensor_nodes = 10;
    sensing_range = 1.6;
    communication_range = 1.5;
    field_x_size = 4.0;
    field_y_size = 4.0;
    field_cells = 25;
  };
};




Configurations process_command_line_arguments(int pArgc, char** pArguments);

SimNode::CoordinatePair createRandomLocation(
  std::default_random_engine& p_generator,
  std::normal_distribution<double>& p_x_distribution,
  std::normal_distribution<double>& p_y_distribution);


int main(int num_arguments, char** arguments) {
  puts("Parsing arguments...");
  Configurations configurations = process_command_line_arguments(
    num_arguments,
    arguments
  );

  puts("Preparing random number generator...");
  std::default_random_engine random_engine(SEED);
  std::normal_distribution<double> x_location_distribution(
    0.0,
    configurations.field_x_size / 2.0
  );
  std::normal_distribution<double> y_location_distribution(
    0.0,
    configurations.field_y_size / 2.0);
  double sample = x_location_distribution(random_engine);


  puts("Generating target field...");
  std::vector<SimNode::Target> target_field;
  target_field.push_back({50.0, {0.0, 0.0}});


  puts("Generating sensor node network...");
  SimNodeNetwork sensorNodeNetwork(SimNodeNetwork::MaxDegreeConsensus, 0.001);
  for (int i = 0; i < configurations.number_of_sensor_nodes; ++i) {
    SimNode::CoordinatePair location = createRandomLocation(
      random_engine,
      x_location_distribution,
      y_location_distribution
    );

std::cout << location.x << ' ' << location.y << std::endl;

    sensorNodeNetwork.AddNode(
      configurations.sensing_range,
      configurations.communication_range,
      location
    );
  }


  puts("Performing consensus filter...");
  std::vector<SimNodeNetwork::EstimateLog> logs =
    sensorNodeNetwork.BuildMapOfField(target_field);
  // record initial readings for output



  // output results

  return 0;
}


Configurations
process_command_line_arguments(int pArgc, char** pArguments) {
  Configurations configurations;

  for (int i = 1; i < pArgc; ++i) {
    char* arg_value_pair = pArguments[i];
    std::string argument = strtok(arg_value_pair, "-=");
    char* value = strtok(NULL, "=");

    if (argument == DATA_FILE_ARG) {
      configurations.data_file_name = value;
      printf(
        "Data File: %s\n",
        configurations.data_file_name.c_str()
      );
    } else if (argument == NUM_NODES_ARG) {
      configurations.number_of_sensor_nodes = atoi(value);
      printf(
        "Number of senor nodes: %i\n",
        configurations.number_of_sensor_nodes
      );
    } else if (argument == SENSING_RANGE_ARG) {
      configurations.sensing_range = strtod(value,NULL);
      printf(
        "Node sensing range: %f\n",
        configurations.sensing_range
      );
    } else if (argument == COMMUNICATION_RANGE_ARG) {
      configurations.communication_range = strtod(value, NULL);
      printf(
        "Node communication range: %f\n",
        configurations.communication_range
      );
    } else if (argument == FIELD_X_ARG) {
      configurations.field_x_size = strtod(value, NULL);
      printf(
        "Field X size: %f\n",
        configurations.field_x_size
      );
    } else if (argument == FIELD_Y_ARG) {
      configurations.field_y_size = strtod(value, NULL);
      printf(
        "Field Y size: %f\n",
        configurations.field_y_size
      );
    } else if (argument == FIELD_CELLS_ARG) {
      configurations.field_cells = atoi(value);
      printf(
        "Cells per side of the field: %i\n",
        configurations.field_cells
      );
    }else {
      printf(
        "%s is an unrecognized argument. Program terminating.\n",
        argument.c_str()
      );
      throw std::exception();
    }
  }

  return configurations;
}


SimNode::CoordinatePair
createRandomLocation(
  std::default_random_engine& p_generator,
  std::normal_distribution<double>& p_x_distribution,
  std::normal_distribution<double>& p_y_distribution) {

  return {p_x_distribution(p_generator), p_y_distribution(p_generator)};
}