#include <utility>
#include <random>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <fstream>

#include <Eigen/Dense>

typedef std::pair<double, double> CoordinatePair;
typedef struct {
  double value;
  CoordinatePair location;
} Target;

const std::string ln = "\r\n";
const std::string del = " ";

const char* DATA_FILE_ARG = "data";
const char* NUM_NODES_ARG = "nodes";
const char* SENSING_RANGE_ARG = "sense_range";
const char* COMMUNICATION_RANGE_ARG = "comm_range";
const char* FIELD_X_ARG = "x";
const char* FIELD_Y_ARG = "y";
const char* FIELD_CELLS_ARG = "cells";

const double C_W_CONSTANT = 0.01;

const int SEED = 5;
const double NO_READING = -9999.9999;

const double ERROR_THRESHOLD = 0.0001;

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


Configurations ProcessCommandLineArguments(int pArgc, char** pArguments);

CoordinatePair GenerateNewCoordinatePair(
  std::default_random_engine& random_generator,
  const Configurations& configurations);

double GenerateNoisyReading(
  Target target,
  double constant,
  CoordinatePair sensor_node_location,
  CoordinatePair average_sensor_location,
  double sensing_range,
  std::default_random_engine generator);

double ComputeDistance(
  CoordinatePair node_location,
  CoordinatePair reckoning_point);

double ComputeNoiseCovariance(
  const CoordinatePair& node_coordinates,
  const CoordinatePair& reckoning_point,
  double p_weight_constant,
  double p_node_sensing_range);

int CountNeighbors(
  double communication_range,
  CoordinatePair source,
  std::vector<CoordinatePair> all_nodes);

int CountNeighborsWhoSenseTarget(
  double communication_range,
  CoordinatePair source,
  std::vector<CoordinatePair> all_nodes,
  Eigen::VectorXd estimates);

double ComputeAverageEstimate(
  Eigen::VectorXd estimates,
  Eigen::MatrixXd weights,
  std::function<double(Eigen::VectorXd, Eigen::MatrixXd)> averaging_method);

double ComputeMaxDegreeWeight(
  int source_node_index,
  double source_node_measurement,
  int neighbor_index,
  double neighbor_measurement,
  double communication_range,
  const std::vector<CoordinatePair>& sensor_nodes,
  const Eigen::MatrixXd weights);

double ComputeMetropolisWeight(
  int source_node_index,
  double source_node_measurement,
  int neighbor_index,
  double neighbor_measurement,
  double communication_range,
  const std::vector<CoordinatePair>& sensor_nodes,
  const Eigen::MatrixXd weights);

bool EmptyEstimatePresent(Eigen::VectorXd estimates);

bool SomeNodeNotConverged(
  Eigen::VectorXd estimates,
  double average_estimate,
  double error_threshold,
  bool method);

void DumpResultsToFile(
  Configurations configurations,
  std::string filename,
  std::vector<CoordinatePair> nodes,
  std::vector<Eigen::VectorXd> estimates);

std::vector<Eigen::VectorXd> MaxDegreeAnalysis(
  Configurations configurations,
  Target target,
  std::vector<CoordinatePair> sensor_nodes,
  CoordinatePair average_node_location,
  std::default_random_engine random_generator,
  Eigen::VectorXd initial_estimates);

std::vector<Eigen::VectorXd> MetropolisAnalysis(
  Configurations configurations,
  Target target,
  std::vector<CoordinatePair> sensor_nodes,
  CoordinatePair average_node_location,
  std::default_random_engine random_generator,
  Eigen::VectorXd initial_estimates);

std::vector<Eigen::VectorXd> WeightDesign1Analysis(
  Configurations configurations,
  Target target,
  std::vector<CoordinatePair> sensor_nodes,
  CoordinatePair average_node_location,
  std::default_random_engine random_generator,
  Eigen::VectorXd initial_estimates);

std::vector<Eigen::VectorXd> WeightDesign2Analysis(
  Configurations configurations,
  Target target,
  std::vector<CoordinatePair> sensor_nodes,
  CoordinatePair average_node_location,
  std::default_random_engine random_generator,
  Eigen::VectorXd initial_estimates);

double ComputeWeightDesign1Weight(
  int source_node_index,
  double source_node_measurement,
  int neighbor_index,
  double neighbor_measurement,
  double communication_range,
  double sensing_range,
  CoordinatePair target_location,
  const std::vector<CoordinatePair>& sensor_nodes,
  const Eigen::MatrixXd weights);

double ComputeWeightDesign2Weight(
  int source_node_index,
  double source_node_measurement,
  int neighbor_index,
  double neighbor_measurement,
  double communication_range,
  double sensing_range,
  CoordinatePair target_location,
  const std::vector<CoordinatePair>& sensor_nodes,
  const Eigen::MatrixXd weights,
  Eigen::VectorXd estimates);

std::vector<Target> ReadInData(Configurations configurations, std::string filename);

void DumpFieldDataToFile(
  Configurations configurations,
  std::string filename,
  std::vector<CoordinatePair> nodes,
  std::vector<double> estimates);

int
main(int arg_count, char** arg_values) {
  Configurations configurations = ProcessCommandLineArguments(
    arg_count,
    arg_values
  );

  // setup (RNGs 'n' stuff)
  std::default_random_engine random_generator(1);

  // generate targets
  std::vector<Target> targets;
  targets.push_back({50.0, {0.0, 0.0}});

  // generate node coordinates
  std::vector<CoordinatePair> sensor_nodes;
  CoordinatePair average_node_location = {0.0, 0.0};
  for (int i = 0; i < configurations.number_of_sensor_nodes; ++i) {
    sensor_nodes.push_back(
      GenerateNewCoordinatePair(random_generator, configurations)
    );

    std::cout << "x: " << sensor_nodes[i].first << " y: " << sensor_nodes[i].second << std::endl;

    average_node_location.first += sensor_nodes[i].first;
      average_node_location.second += sensor_nodes[i].second;
  }
  average_node_location.first /= (double) configurations.number_of_sensor_nodes;
  average_node_location.second /=
    (double) configurations.number_of_sensor_nodes;

  std::cout << "x: " << average_node_location.first << " y: " << average_node_location.second << std::endl;

  // for each target
  for (Target target : targets) {

    Eigen::VectorXd estimates(configurations.number_of_sensor_nodes);
    for (int i = 0; i < configurations.number_of_sensor_nodes; ++i) {
      estimates(i) = GenerateNoisyReading(
        target,
        0.01,
        sensor_nodes[i],
        average_node_location,
        configurations.sensing_range,
        random_generator
      );

      std::cout << "estimate: " << estimates(i) << std::endl;
    }

    std::vector<Eigen::VectorXd> results = MaxDegreeAnalysis(
      configurations,
      target,
      sensor_nodes,
      average_node_location,
      random_generator,
      estimates
    );
    DumpResultsToFile(configurations, "MaxDegreeResults.txt", sensor_nodes, results);

    results = MetropolisAnalysis(
      configurations,
      target,
      sensor_nodes,
      average_node_location,
      random_generator,
      estimates
    );
    DumpResultsToFile(configurations, "MetropolisResults.txt", sensor_nodes, results);

    results = WeightDesign1Analysis(
      configurations,
      target,
      sensor_nodes,
      average_node_location,
      random_generator,
      estimates
    );
    DumpResultsToFile(configurations, "WeightDesign1Results.txt", sensor_nodes, results);

    results = WeightDesign2Analysis(
      configurations,
      target,
      sensor_nodes,
      average_node_location,
      random_generator,
      estimates
    );
    DumpResultsToFile(configurations, "WeightDesign2Results.txt", sensor_nodes, results);
  }

  // part 2/////////////////////////////////////////////////////////////////////////////////
  configurations.field_x_size = 12.0;
  configurations.field_y_size = 12.0;
  configurations.field_cells = 25;
  configurations.number_of_sensor_nodes = 30;
  configurations.sensing_range = 5.0;
  configurations.communication_range = 4.5;


  sensor_nodes.clear();
  average_node_location = {0.0, 0.0};
  for (int i = 0; i < configurations.number_of_sensor_nodes; ++i) {
    sensor_nodes.push_back(
      {0.0, 0.0}
    );

    std::cout << "x: " << sensor_nodes[i].first << " y: " << sensor_nodes[i].second << std::endl;

    average_node_location.first += sensor_nodes[i].first;
      average_node_location.second += sensor_nodes[i].second;
  }
  average_node_location.first /= (double) configurations.number_of_sensor_nodes;
  average_node_location.second /=
    (double) configurations.number_of_sensor_nodes;

  std::cout << "x: " << average_node_location.first << " y: " << average_node_location.second << std::endl;

  std::vector<double> final_cell_estimates_1;
  std::vector<double> final_cell_estimates_2;

  std::vector<Target> new_targets = ReadInData(configurations, "field1.txt");

  // for each target
  for (Target target : new_targets) {

    Eigen::VectorXd estimates(configurations.number_of_sensor_nodes);
    for (int i = 0; i < configurations.number_of_sensor_nodes; ++i) {
      estimates(i) = GenerateNoisyReading(
        target,
        0.01,
        sensor_nodes[i],
        average_node_location,
        configurations.sensing_range,
        random_generator
      );

      std::cout << "estimate: " << estimates(i) << std::endl;
    }

    std::vector<Eigen::VectorXd> wd1_results = WeightDesign1Analysis(
      configurations,
      target,
      sensor_nodes,
      average_node_location,
      random_generator,
      estimates
    );
    std::vector<Eigen::VectorXd> wd2_results = WeightDesign2Analysis(
      configurations,
      target,
      sensor_nodes,
      average_node_location,
      random_generator,
      estimates
    );

    final_cell_estimates_1.push_back(wd1_results.back()(0));
    final_cell_estimates_2.push_back(wd2_results.back()(0));
  }

  DumpFieldDataToFile(configurations, "WeightDesign1_field.txt", sensor_nodes, final_cell_estimates_1);
  DumpFieldDataToFile(configurations, "WeightDesign2_field.txt", sensor_nodes, final_cell_estimates_2);

  return 0;
}


Configurations
ProcessCommandLineArguments(int pArgc, char** pArguments) {
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


CoordinatePair
GenerateNewCoordinatePair(
  std::default_random_engine& random_generator,
  const Configurations& configurations) {

  CoordinatePair new_coordinates;

  new_coordinates.first =
    random_generator() % (int) (configurations.field_x_size * 5.0);
  new_coordinates.first /= 10.0;
  if (random_generator() % 2 == 0) {
    new_coordinates.first *= -1.0;
  }

  new_coordinates.second =
    random_generator() % (int) (configurations.field_y_size * 5.0);
  new_coordinates.second /= 10.0;
  if (random_generator() % 2 == 0) {
    new_coordinates.second *= -1.0;
  }

  return new_coordinates;
}

double GenerateNoisyReading(
  Target target,
  double constant,
  CoordinatePair sensor_node_location,
  CoordinatePair average_sensor_location,
  double sensing_range,
  std::default_random_engine generator) {

  double distance = ComputeDistance(sensor_node_location, target.location);

  if (distance <= sensing_range) {
    std::normal_distribution<double> noise_distribution(
      target.value,
      ComputeNoiseCovariance(
        sensor_node_location,
        average_sensor_location,
        constant,
        sensing_range
      )
    );

    return noise_distribution(generator);
  } else {
    return NO_READING;
  }
}

double ComputeDistance(
  CoordinatePair node_location,
  CoordinatePair reckoning_point) {

  // Euclidean Distance = sqrt((x_0 - x_1)^2 + (y_0 - y_1)^2)

  double x_diff = node_location.first - reckoning_point.first;
  double y_diff = node_location.second - reckoning_point.second;

  return sqrt((x_diff * x_diff) + (y_diff * y_diff));
}

double ComputeNoiseCovariance(
  const CoordinatePair& node_coordinates,
  const CoordinatePair& reckoning_point,
  double constant,
  double sensing_range) {

  double distance = ComputeDistance(
    node_coordinates,
    reckoning_point
  );

  double numerator = (distance * distance) + constant;

  return numerator / (sensing_range * sensing_range);
}

double ComputeAverageEstimate(
  Eigen::VectorXd estimates,
  Eigen::MatrixXd weights,
  std::function<double(Eigen::VectorXd, Eigen::MatrixXd)> averaging_method) {

  double sum = 0.0;
  int n = 0;
  for (int i = 0; i < estimates.size(); ++i) {
    if (estimates(i) != NO_READING) {
      sum += estimates(i);
      n++;
    }
  }

  return sum / (double) n;
}

double ComputeMaxDegreeWeight(
  int source_node_index,
  double source_node_measurement,
  int neighbor_index,
  double neighbor_measurement,
  double communication_range,
  const std::vector<CoordinatePair>& sensor_nodes,
  const Eigen::MatrixXd weights) {

  if (source_node_index != neighbor_index) {
    if (source_node_measurement == NO_READING) {
      return 0.0;
    }


    int num_neighbors = 0;
    for (int i = 0; i < sensor_nodes.size(); ++i) {
      if (i != source_node_index) {
        double separation = ComputeDistance(
          sensor_nodes[source_node_index],
          sensor_nodes[neighbor_index]
        );

        if (separation <= communication_range) {
          num_neighbors++;
        }
      }
    }
    
    if (neighbor_measurement != NO_READING) {
      double separation = ComputeDistance(
        sensor_nodes[source_node_index],
        sensor_nodes[neighbor_index]
      );

      if (separation <= communication_range) {
        return 1.0 / (double) sensor_nodes.size();
      } 
    }

    return 0.0;
  } else {
    if (source_node_measurement == NO_READING) {
      return 1.0;
    } else {
      double neighbor_weights = 0.0;

      for (int j = 0; j < sensor_nodes.size(); ++j) {
        if (j != source_node_index) {
          neighbor_weights += weights(source_node_index, j);
        }
      }

      return 1.0 - neighbor_weights;
    }
  }
}

bool EmptyEstimatePresent(Eigen::VectorXd estimates) {
  for (int i = 0; i < estimates.size(); ++i) {
    if (estimates(i) == NO_READING) {
      return true;
    }
  }

  return false;
}

bool SomeNodeNotConverged(
  Eigen::VectorXd estimates,
  double average_estimate,
  double error_threshold,
  bool method) {

  if (method) {
    for (int i = 0; i < estimates.size(); ++i) {
      if (estimates(i) != NO_READING &&
          fabs(estimates(i) - average_estimate) >= error_threshold) {
        return true;
      }
    }
  } else {
    double estimate = 0.0;
    for (int i = 0; i < estimates.size(); ++i) {
      if (estimate == 0.0 && estimates(i) != NO_READING) {
        estimate = estimates(i);
        break;
      }
    }

    for (int i = 0; i < estimates.size(); ++i) {
      if (estimates(i) != NO_READING && 
          (estimates(i) < (estimate - error_threshold) || (estimate + error_threshold) < estimates(i))) {
        return true;
      }
    }
  }

  return false;
}


void DumpResultsToFile(
  Configurations configurations,
  std::string filename,
  std::vector<CoordinatePair> nodes,
  std::vector<Eigen::VectorXd> estimates) {

  std::ofstream fout;
  fout.open(filename.c_str());

  for (CoordinatePair source : nodes) {
    int num_neighbors = 0;
    for (CoordinatePair neighbor : nodes) {
      if (configurations.communication_range <
          ComputeDistance(source, neighbor)) {
        num_neighbors++;
      }
    }

    fout << source.first << del << source.second << del << num_neighbors
         << ln;
  }

  for (Eigen::VectorXd snapshot : estimates) {
    fout << snapshot.transpose() << ln;
  }

  fout.close();
}


std::vector<Eigen::VectorXd> MaxDegreeAnalysis(
  Configurations configurations,
  Target target,
  std::vector<CoordinatePair> sensor_nodes,
  CoordinatePair average_node_location,
  std::default_random_engine random_generator,
  Eigen::VectorXd initial_estimates) {

  // take initial measurements
  Eigen::VectorXd estimates = initial_estimates;


  // determine weight matrix
  Eigen::MatrixXd weights(
    configurations.number_of_sensor_nodes,
    configurations.number_of_sensor_nodes
  );
  weights.setZero();
  for (int i = 0; i < configurations.number_of_sensor_nodes; ++i) {
    for (int j = 0; j < configurations.number_of_sensor_nodes; ++j) {
      if (i != j) {
        weights(i, j) = ComputeMaxDegreeWeight(
          i,
          estimates(i),
          j,
          estimates(j),
          configurations.communication_range,
          sensor_nodes,
          weights
        );
      }
    }

    weights(i, i) = ComputeMaxDegreeWeight(i, estimates(i), i, estimates(i), configurations.communication_range, sensor_nodes, weights);
  }

  std::cout << "Max-Degree weights" << std::endl << weights << std::endl;

  std::function<double(Eigen::VectorXd, Eigen::MatrixXd)> averaging_method;
  double average_estimate = 0;
  average_estimate = ComputeAverageEstimate(
    estimates,
    weights,
    averaging_method
  );

  std::cout << "average: " << average_estimate << std::endl;

  // iterate til consensus
  std::vector<Eigen::VectorXd> estimate_history;
  estimate_history.push_back(estimates);
  int l = 0;
  while (SomeNodeNotConverged(estimates, average_estimate, ERROR_THRESHOLD, true)) {
    estimates = weights * estimates;
    l++;
    estimate_history.push_back(estimates);
  }


  // update nodes that didn't see the target
  double consensus;
  for (int i = 0; i < estimates.size(); ++i) {
    if (estimates(i) != NO_READING) {
      consensus = estimates(i);
      break;
    }
  }
  for (int i = 0; i < estimates.size(); ++i) {
    if (estimates(i) == NO_READING) {
      estimates(i) = consensus;
    }
  }
  estimate_history.push_back(estimates);

  return estimate_history; 
}

std::vector<Eigen::VectorXd> MetropolisAnalysis(
  Configurations configurations,
  Target target,
  std::vector<CoordinatePair> sensor_nodes,
  CoordinatePair average_node_location,
  std::default_random_engine random_generator,
  Eigen::VectorXd initial_estimates) {

  // take initial measurements
  Eigen::VectorXd estimates = initial_estimates;

  // determine weight matrix
  Eigen::MatrixXd weights(
    configurations.number_of_sensor_nodes,
    configurations.number_of_sensor_nodes
  );
  weights.setZero();
  for (int i = 0; i < configurations.number_of_sensor_nodes; ++i) {
    for (int j = 0; j < configurations.number_of_sensor_nodes; ++j) {
      if (i != j) {
        weights(i, j) = ComputeMetropolisWeight(
          i,
          estimates(i),
          j,
          estimates(j),
          configurations.communication_range,
          sensor_nodes,
          weights
        );
      }
    }

    weights(i, i) = ComputeMetropolisWeight(i, estimates(i), i, estimates(i), configurations.communication_range, sensor_nodes, weights);
  }

  std::cout << "Metropolis Weights: " << std::endl << weights << std::endl;

  double average_estimate = 0.0;
  int num_in_average = 0;
  for (int i = 0; i < estimates.size(); ++i) {
    if (estimates(i) != NO_READING) {
      average_estimate += estimates(i);
      num_in_average++;
    }
  }
  average_estimate /= (double) num_in_average;

  std::cout << "average: " << average_estimate << std::endl;

  // iterate til consensus
  std::vector<Eigen::VectorXd> estimate_history;
  estimate_history.push_back(estimates);
  int l = 0;
  while (SomeNodeNotConverged(estimates, average_estimate, ERROR_THRESHOLD, true)) {
    estimates = weights * estimates;
    l++;
    estimate_history.push_back(estimates);
  }

  // update nodes that didn't see the target
  double consensus;
  for (int i = 0; i < estimates.size(); ++i) {
    if (estimates(i) != NO_READING) {
      consensus = estimates(i);
      break;
    }
  }
  for (int i = 0; i < estimates.size(); ++i) {
    if (estimates(i) == NO_READING) {
      estimates(i) = consensus;
    }
  }
  estimate_history.push_back(estimates);

  return estimate_history; 
}


int CountNeighbors(
  double communication_range,
  CoordinatePair source,
  std::vector<CoordinatePair> all_nodes) {

  int num_neighbors = 0;

  for (CoordinatePair node : all_nodes) {
    if (node != source &&
        ComputeDistance(source, node) <= communication_range) {

      num_neighbors++;
    }
  }

  return num_neighbors;
}


double ComputeMetropolisWeight(
  int source_node_index,
  double source_node_measurement,
  int neighbor_index,
  double neighbor_measurement,
  double communication_range,
  const std::vector<CoordinatePair>& sensor_nodes,
  const Eigen::MatrixXd weights) {

  if (source_node_index != neighbor_index) {
    if (source_node_measurement != NO_READING &&
        neighbor_measurement != NO_READING &&
        ComputeDistance(sensor_nodes[source_node_index], sensor_nodes[neighbor_index]) <= communication_range) {

      return 1.0 / std::max(
        CountNeighbors(
          communication_range,
          sensor_nodes[source_node_index],
          sensor_nodes
        ),
        CountNeighbors(
          communication_range,
          sensor_nodes[neighbor_index],
          sensor_nodes
        )
      );
    }

    return 0.0;
  } else {
    if (source_node_measurement == NO_READING) {
      return 1.0;
    } else {
      double neighbor_weights = 0.0;

      for (int j = 0; j < sensor_nodes.size(); ++j) {
        if (j != source_node_index) {
          neighbor_weights += weights(source_node_index, j);
        }
      }

      return 1.0 - neighbor_weights;
    }
  }
}



std::vector<Eigen::VectorXd> WeightDesign1Analysis(
  Configurations configurations,
  Target target,
  std::vector<CoordinatePair> sensor_nodes,
  CoordinatePair average_node_location,
  std::default_random_engine random_generator,
  Eigen::VectorXd initial_estimates) {

  // take initial measurements
  Eigen::VectorXd estimates = initial_estimates;

  // determine weight matrix
  Eigen::MatrixXd weights(
    configurations.number_of_sensor_nodes,
    configurations.number_of_sensor_nodes
  );
  weights.setZero();
  for (int i = 0; i < configurations.number_of_sensor_nodes; ++i) {
    for (int j = 0; j < configurations.number_of_sensor_nodes; ++j) {
      if (i != j) {
        weights(i, j) = ComputeWeightDesign1Weight(
          i,
          estimates(i),
          j,
          estimates(j),
          configurations.communication_range,
          configurations.sensing_range,
          target.location,
          sensor_nodes,
          weights
        );
      }
    }

    weights(i, i) = ComputeWeightDesign1Weight(i, estimates(i), i, estimates(i),
      configurations.communication_range,
      configurations.sensing_range,
      target.location,
      sensor_nodes,
      weights
    );
  }

  std::cout << "Weight Design 1 Weights:" << std::endl << weights << std::endl;

  double average_estimate = 0.0;
  double total_weight = 0.0;
  for (int i = 0; i < estimates.size(); ++i) {
    if (estimates(i) != NO_READING) {
      average_estimate += estimates(i) * weights(i, i);
      total_weight += weights(i, i);
    }
  }
  average_estimate /= total_weight;

  std::cout << "weighted average: " << average_estimate << std::endl;

  // iterate til consensus
  std::vector<Eigen::VectorXd> estimate_history;
  estimate_history.push_back(estimates);
  int l = 0;
  while (SomeNodeNotConverged(estimates, average_estimate, ERROR_THRESHOLD, false)) {
    estimates = weights * estimates;
    l++;
    estimate_history.push_back(estimates);

// char y;
// std::cin >> y;
// std::cout << estimates.transpose() << std::endl;
  }

  // update nodes that didn't see the target
  double consensus;
  for (int i = 0; i < estimates.size(); ++i) {
    if (estimates(i) != NO_READING) {
      consensus = estimates(i);
      break;
    }
  }
  for (int i = 0; i < estimates.size(); ++i) {
    if (estimates(i) == NO_READING) {
      estimates(i) = consensus;
    }
  }
  estimate_history.push_back(estimates);

  return estimate_history; 
}


double ComputeWeightDesign1Weight(
  int source_node_index,
  double source_node_measurement,
  int neighbor_index,
  double neighbor_measurement,
  double communication_range,
  double sensing_range,
  CoordinatePair target_location,
  const std::vector<CoordinatePair>& sensor_nodes,
  const Eigen::MatrixXd weights) {

  if (source_node_index != neighbor_index) {
    if (source_node_measurement != NO_READING &&
        neighbor_measurement != NO_READING &&
        ComputeDistance(sensor_nodes[source_node_index], sensor_nodes[neighbor_index]) <= communication_range) {


      return C_W_CONSTANT /
        (ComputeNoiseCovariance(
          sensor_nodes[source_node_index],
          target_location,
          C_W_CONSTANT,
          sensing_range
        ) +
        ComputeNoiseCovariance(
          sensor_nodes[source_node_index],
          target_location,
          C_W_CONSTANT,
          sensing_range
        ))
      ;
    }

    return 0.0;
  } else {
    if (source_node_measurement == NO_READING) {
      return 1.0;
    } else {
      double neighbor_weights = 0.0;

      for (int j = 0; j < sensor_nodes.size(); ++j) {
        if (j != source_node_index) {
          neighbor_weights += weights(source_node_index, j);
        }
      }

      return 1.0 - neighbor_weights;
    }
  }
}


std::vector<Eigen::VectorXd> WeightDesign2Analysis(
  Configurations configurations,
  Target target,
  std::vector<CoordinatePair> sensor_nodes,
  CoordinatePair average_node_location,
  std::default_random_engine random_generator,
  Eigen::VectorXd initial_estimates) {

  // take initial measurements
  Eigen::VectorXd estimates = initial_estimates;

  // determine weight matrix
  Eigen::MatrixXd weights(
    configurations.number_of_sensor_nodes,
    configurations.number_of_sensor_nodes
  );
  weights.setZero();
  for (int i = 0; i < configurations.number_of_sensor_nodes; ++i) {
    weights(i, i) = ComputeWeightDesign2Weight(i, estimates(i), i, estimates(i),
      configurations.communication_range,
      configurations.sensing_range,
      target.location,
      sensor_nodes,
      weights,
      estimates
    );

    for (int j = 0; j < configurations.number_of_sensor_nodes; ++j) {
      if (i != j) {
        weights(i, j) = ComputeWeightDesign2Weight(
          i,
          estimates(i),
          j,
          estimates(j),
          configurations.communication_range,
          configurations.sensing_range,
          target.location,
          sensor_nodes,
          weights,
          estimates
        );
      }
    }
  }

  std::cout << "Weight Design 2 Weights:" << std::endl << weights << std::endl;

  double average_estimate = 0.0;
  double total_weight = 0.0;
  for (int i = 0; i < estimates.size(); ++i) {
    if (estimates(i) != NO_READING) {
      average_estimate += estimates(i) * weights(i, i);
      total_weight += weights(i, i);
    }
  }
  average_estimate /= total_weight;

  std::cout << "weighted average: " << average_estimate << std::endl;

  // iterate til consensus
  std::vector<Eigen::VectorXd> estimate_history;
  estimate_history.push_back(estimates);
  int l = 0;
  while (SomeNodeNotConverged(estimates, average_estimate, ERROR_THRESHOLD, false)) {
    estimates = weights * estimates;
    l++;
    estimate_history.push_back(estimates);

// char y;
// std::cin >> y;
// std::cout << estimates.transpose() << std::endl;
  }

  // update nodes that didn't see the target
  double consensus;
  for (int i = 0; i < estimates.size(); ++i) {
    if (estimates(i) != NO_READING) {
      consensus = estimates(i);
      break;
    }
  }
  for (int i = 0; i < estimates.size(); ++i) {
    if (estimates(i) == NO_READING) {
      estimates(i) = consensus;
    }
  }
  estimate_history.push_back(estimates);

  return estimate_history; 
}


double ComputeWeightDesign2Weight(
  int source_node_index,
  double source_node_measurement,
  int neighbor_index,
  double neighbor_measurement,
  double communication_range,
  double sensing_range,
  CoordinatePair target_location,
  const std::vector<CoordinatePair>& sensor_nodes,
  const Eigen::MatrixXd weights,
  Eigen::VectorXd estimates) {

  if (source_node_index != neighbor_index) {
    if (source_node_measurement != NO_READING &&
        neighbor_measurement != NO_READING &&
        ComputeDistance(sensor_nodes[source_node_index], sensor_nodes[neighbor_index]) <= communication_range) {

        return (1.0 - weights(source_node_index, source_node_index)) /
          (double) CountNeighborsWhoSenseTarget(communication_range, sensor_nodes[source_node_index], sensor_nodes, estimates);
    }

    return 0.0;
  } else {
    if (source_node_measurement == NO_READING) {
      return 1.0;
    } else {
      return C_W_CONSTANT / ComputeNoiseCovariance(sensor_nodes[source_node_index], target_location, C_W_CONSTANT, sensing_range);
    }
  }
}


int CountNeighborsWhoSenseTarget(
  double communication_range,
  CoordinatePair source,
  std::vector<CoordinatePair> all_nodes,
  Eigen::VectorXd estimates) {

  int num_neighbors = 0;

  for (int i = 0; i < all_nodes.size(); ++i) {
    CoordinatePair node = all_nodes[i];

    if (node != source &&
        ComputeDistance(source, node) <= communication_range &&
        estimates(i) != NO_READING) {

      num_neighbors++;
    }
  }

  return num_neighbors;
}

std::vector<Target> ReadInData(Configurations configurations, std::string filename) {

  std::ifstream fin;
  fin.open(filename.c_str());

  std::vector<Target> targets;

  char dummy;

  double x;
  double y;

  double increment = 0.5;
  double startval = -6.0;

  y = startval;
  for (int i = 0; i < configurations.field_cells; ++i) {
    x = startval;
    for (int j = 0; j < configurations.field_cells; ++j) {
      Target target;
      fin >> target.value >> dummy;
      target.value *= 1;
      target.location = {x, y};
      targets.push_back(target);

std::cout << target.value << ", ";

      x += increment;
    }

std::cout << std::endl;

    y += increment;
  }

char c;
std::cin >> c;

  fin.close();

  return targets;
}

void DumpFieldDataToFile(
  Configurations configurations,
  std::string filename,
  std::vector<CoordinatePair> nodes,
  std::vector<double> estimates) {

  std::ofstream fout;
  fout.open(filename.c_str());

  for (CoordinatePair source : nodes) {
    int num_neighbors = 0;
    for (CoordinatePair neighbor : nodes) {
      if (configurations.communication_range <
          ComputeDistance(source, neighbor)) {
        num_neighbors++;
      }
    }

    fout << source.first << ", " << source.second << ", " << num_neighbors
         << ln;
  }

  for (int i = 0; i < configurations.field_cells; ++i) {
    for (int j = 0; j < configurations.field_cells; ++j) {
      fout << estimates[(i * configurations.field_cells) + j] << ", ";
    }
    fout << ln;
  }

  fout.close();
}