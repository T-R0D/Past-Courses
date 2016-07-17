#if 0
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>
#include <sys/time.h>

#include "project_apsp.hpp"
#include "sequential_apsp.hpp"

#define beta 16

#define UNKNOWN "unknown"
#define NOT_STARTED "not started"
#define COMPLETE "complete"
#define INC_ERROR "incomplete - error"

typedef struct {
  std::string device;
  std::string method;
  std::string completion_status;
  unsigned graph_size;
  unsigned blocks;
  unsigned threads;
  float compute_time;
  float total_time;

// ApspResult result = {
//   .device = "thing",
//   ...
//   .total_time = 100
// };
} ApspResult;

void InitAspsResult(ApspResult* result) {
    result->device = UNKNOWN;
    result->method = UNKNOWN;
    result->completion_status = NOT_STARTED;
    result->graph_size = 0;
    result->blocks = 1;
    result->threads = 1;
    result->compute_time = 0.0;
    result->total_time = 0.0;
}

std::string
ResultsToString(
  const std::vector<ApspResult>& results,
  const std::string& delimeter) {
/**
 *
 */
  std::stringstream ss;
  const std::string& d = delimeter;

  ss << "Device" << d << "Method" << d << "Status" << d << "|v|" << d
     << "Compute Time" << d <<"Total Time" << std::endl;

  unsigned i;
  for (i = 0; i < results.size(); ++i) {
    ss << results[i].device << d
       << results[i].method << d
       << results[i].completion_status << d
       << results[i].graph_size << d
       << results[i].blocks << d
       << results[i].threads << d
       << results[i].compute_time << d
       << results[i].total_time << std::endl;
  }

  return ss.str();
}

typedef struct {
  std::string results_file_name;

  unsigned average_degree;

  unsigned graph_size_start;
  unsigned graph_size_stop;
  unsigned graph_size_step_factor;

  double jitter_start;
  double jitter_stop;
  double jitter_step_factor;

  unsigned num_trials;

  unsigned block_size_start;
  unsigned block_size_stop;
  unsigned block_size_step_factor;

  unsigned thread_size_start;
  unsigned thread_size_stop;
  unsigned thread_size_step_factor;
} RunParameters;

void InitRunParameters(RunParameters* run_params) {
  run_params->results_file_name = "Results/seq_apsp_results";

  run_params->average_degree = 6;

  run_params->graph_size_start = 512;
  run_params->graph_size_stop = 512;
  run_params->graph_size_step_factor = 2;

  run_params->jitter_start = 1.0;
  run_params->jitter_stop = 1.0;
  run_params->jitter_step_factor = 0.1;

  run_params->num_trials = 1;

  run_params->block_size_start = 1;
  run_params->block_size_stop = 1;
  run_params->block_size_step_factor = 16;

  run_params->thread_size_start = 1;
  run_params->thread_size_stop = 1;
  run_params->thread_size_step_factor = 16;
}

int
ProcessArguments(RunParameters* run_params, int argc, char** argv) {
  InitRunParameters(run_params);

  unsigned i;
  for (i = 1; i < argc; i += 2) {
    if (strcmp(argv[i], "output") == 0) {
      run_params->results_file_name = argv[i + 1];
    } else if (strcmp(argv[i], "degree") == 0) {
      run_params->average_degree = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "trials") == 0) {
      run_params->num_trials = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "size") == 0) {
      char* sequence_param = strtok(argv[i + 1], ",");
      run_params->graph_size_start = atoi(sequence_param);
      sequence_param = strtok(NULL, ",");
      run_params->graph_size_stop = atoi(sequence_param);
      sequence_param = strtok(NULL, ",");
      run_params->graph_size_step_factor = atoi(sequence_param);
    } else if (strcmp(argv[i], "jitter") == 0) {
      char* sequence_param = strtok(argv[i + 1], ",");
      run_params->jitter_start = atof(sequence_param);
      sequence_param = strtok(NULL, ",");
      run_params->jitter_stop = atof(sequence_param);
      sequence_param = strtok(NULL, ",");
      run_params->jitter_step_factor = atof(sequence_param);
    } else if (strcmp(argv[i], "block") == 0) {
      char* sequence_param = strtok(argv[i + 1], ",");
      run_params->block_size_start = atoi(sequence_param);
      sequence_param = strtok(NULL, ",");
      run_params->block_size_stop = atoi(sequence_param);
      sequence_param = strtok(NULL, ",");
      run_params->block_size_step_factor = atoi(sequence_param);
    } else if (strcmp(argv[i], "thread") == 0) {
      char* sequence_param = strtok(argv[i + 1], ",");
      run_params->thread_size_start = atoi(sequence_param);
      sequence_param = strtok(NULL, ",");
      run_params->thread_size_stop = atoi(sequence_param);
      sequence_param = strtok(NULL, ",");
      run_params->thread_size_step_factor = atoi(sequence_param);
    } else if (strcmp(argv[i], "-h") == 0) {
      puts("Usage:");
      puts("output   [str: file name]");
      puts("degree   [int: average degree]");
      puts("trials   [int: trials performed]");
      puts("size     [int: start],[int: stop],[int: step]");
      puts("jitter   [float: start],[float: stop],[float: step]");
      puts("block    [int: start],[int: stop],[int: step]");
      puts("thread   [int: start],[int: stop],[int: step]");
      puts("");
      puts("Example: ./apsp output results.txt trials 3 size 512,1024,2");
      exit(0);
    } else {
      puts("Warning: unrecognized, unused args provided.");
    }
  }
}

int
main(int argc, char** argv) {

  RunParameters run_params;
  ProcessArguments(&run_params, argc, argv);

  std::vector<ApspResult> results;

  unsigned graph_size, trial;
  double jitter;
  for (
    graph_size = run_params.graph_size_start;
    graph_size <= run_params.graph_size_stop;
    graph_size *= run_params.graph_size_step_factor) {

    for (
      jitter = run_params.jitter_start;
      jitter <= run_params.jitter_stop;
      jitter += run_params.jitter_step_factor) {

      for (trial = 0; trial < run_params.num_trials; ++trial) {
        float* graph = NULL;
        unsigned size = (unsigned) ((float) graph_size * jitter);
        GenerateErdosRenyiGraph(&graph, graph_size, run_params.average_degree);

        struct timeval seq_start, seq_finish;

        gettimeofday(&seq_start, NULL);

        NaiveFloydWarshall(graph, graph_size);
        
        gettimeofday(&seq_finish, NULL);
        float elapsed_time = ComputeElapsedSeconds(&seq_start, &seq_finish);

        results.push_back(
          {"CPU", "sequential-FW", COMPLETE, graph_size, 1, 1, elapsed_time,
            elapsed_time}
        );

        free(graph); graph = NULL;
      }
    }
  }

  FILE* out_file = fopen(run_params.results_file_name.c_str(), "w");
  fprintf(out_file, "%s", ResultsToString(results, ", ").c_str());
  fclose(out_file);

  return 0;
}
#endif