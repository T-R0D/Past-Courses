#include <cassert>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <ctime>  // clock() supposedly gives ms precision
                  // clock() gives CPU time on Linux, wall-clock on Windows

#include "cuda_runtime.h"

#include "vector_add.hpp"

const std::string STRIDING = "Striding";
const std::string INTERVAL = "Interval";
const std::string OPERATION_SUCCESS = "Success";
const std::string ALLOCATION_FAILURE = "Allocation Failure";
const std::string HOST_ALLOCATION_FAILURE = "Allocation Failure: host";
const std::string GPU_ALLOCATION_FAILURE = "Allocation Failure: gpu";
const std::string OPERATION_FAILED = "Operation Failed";
const std::string SUM_FAILED = OPERATION_FAILED + std::string(": sum");

const std::string DELIMETER = ",";

/**
 *
 */
typedef struct {
  unsigned int vector_size;
  unsigned int num_gpu_blocks;
  unsigned int num_gpu_threads;
} ExperimentInfo;

/**
 *
 */
typedef struct {
  std::string outcome;
  double run_time;
  int vector_size;
} CpuExperimentResult;

/**
 *
 */
typedef struct {
  std::string outcome;
  float compute_time;
  float total_run_time;
  int vector_size;
  int num_blocks;
  int num_threads;
  std::string work_distribution;
} GpuExperimentResult;

/**
 *
 */
ExperimentInfo
GetExperimentInfoFromUser();

/**
 *
 */
bool
ExperimentInfoIsValid(const ExperimentInfo& experiment_info);

/**
 *
 */
CpuExperimentResult
PerformCpuVectorAdditionExperiment(const size_t vector_size);

/**
 *
 */
GpuExperimentResult
PerformGPUVectorAdditionExperiment(
  const size_t vector_size,
  const int num_gpu_blocks,
  const int num_gpu_threads,
  const bool use_striding);

bool
AllocateHostVectors(
  int** vector_a, int** vector_b, int** vector_c, const int size);

void
FreeHostVectors(int** vector_a, int** vector_b, int** vector_c);

bool
AllocateGpuVectors(
  int** vector_a, int** vector_b, int** vector_c, const int size);

bool
FreeGpuVectors(int* vector_a, int* vector_b, int* vector_c);

bool
VectorsAreSummedProperly(
  const int* vector_a, const int* vector_b, const int* vector_c, const int vector_size);

void
DumpDataToFile(
  const std::string& output_file_name,
  const size_t size_of_vector_element,
  const std::vector<GpuExperimentResult>& gpu_interval_results,
  const std::vector<GpuExperimentResult>& gpu_striding_results,
  const std::vector<CpuExperimentResult>& cpu_results);

std::string
GetGpuProperties(const int device_number);

/**
 *
 */
int
main(int argc, char** argv) {
  bool use_striding = false;
  std::vector<GpuExperimentResult> gpu_interval_results;
  std::vector<GpuExperimentResult> gpu_striding_results;
  std::vector<CpuExperimentResult> cpu_results;
  std::string output_file_name;

  printf("Enter a name for the output file: ");
  std::cin >> output_file_name;

  #define USER_INPUT_MODE 0
  #if USER_INPUT_MODE
    bool keep_going = true;
    while (keep_going) {
      ExperimentInfo experiment_info = GetExperimentInfoFromUser();

      if (ExperimentInfoIsValid(experiment_info)) {
        gpu_striding_results.push_back(
          PerformGPUVectorAdditionExperiment(
            experiment_info.vector_size,
            experiment_info.num_gpu_blocks,
            experiment_info.num_gpu_threads,
            true
          )
        );
        gpu_interval_results.push_back(
          PerformGPUVectorAdditionExperiment(
            experiment_info.vector_size,
            experiment_info.num_gpu_blocks,
            experiment_info.num_gpu_threads,
            false
          )
        );
        cpu_results.push_back(
          PerformCpuVectorAdditionExperiment(experiment_info.vector_size)
        );
      
        printf("Would you like to stop and dump the data to a file? (y/*) ");
        char response;
        std::cin >> response;
        if (response == 'y') {
          keep_going = false;
        }
      } else {
        puts("You provided bad parameters for the experiment. Try again.");
      }
    }
  #else
  // once it fails, it doesn't recover
  //1 000 000 ; 15 000 000; 2 000 000
  for (int vector_size = 1000000; vector_size <= 10000000; vector_size += 2000000) {
    for (int num_gpu_blocks = 1; num_gpu_blocks <= 65535; num_gpu_blocks *= 8) {
      for (int num_gpu_threads = 1; num_gpu_threads <= 1024; num_gpu_threads *= 4) {
        gpu_striding_results.push_back(
          PerformGPUVectorAdditionExperiment(
            vector_size,
            num_gpu_blocks,
            num_gpu_threads,
            true
          )
        );
        gpu_interval_results.push_back(
          PerformGPUVectorAdditionExperiment(
            vector_size,
            num_gpu_blocks,
            num_gpu_threads,
            false
          )
        );
      }
    }

    cpu_results.push_back(PerformCpuVectorAdditionExperiment(vector_size));
  }

  for (int vector_size = 15000000, int num_gpu_blocks = 65535, int num_gpu_threads = 1024;
        vector_size <= 100000000;
        vector_size += 5000000) {
    gpu_striding_results.push_back(
      PerformGPUVectorAdditionExperiment(
        vector_size,
        num_gpu_blocks,
        num_gpu_threads,
        true
      )
    );
    gpu_interval_results.push_back(
      PerformGPUVectorAdditionExperiment(
        vector_size,
        num_gpu_blocks,
        num_gpu_threads,
        false
      )
    );

    cpu_results.push_back(PerformCpuVectorAdditionExperiment(vector_size));
  }

  int vector_size = 10000000;
  for (int num_gpu_blocks = 2; num_gpu_blocks <= 65536; num_gpu_blocks *= 2) {
    for (int num_gpu_threads = 1; num_gpu_threads <= 1024; num_gpu_threads *= 2) {
      gpu_striding_results.push_back(
        PerformGPUVectorAdditionExperiment(
          vector_size,
          num_gpu_blocks - 1,
          num_gpu_threads,
          true
        )
      );
      gpu_interval_results.push_back(
        PerformGPUVectorAdditionExperiment(
          vector_size,
          num_gpu_blocks - 1,
          num_gpu_threads,
          false
        )
      );
    }
  }
  cpu_results.push_back(PerformCpuVectorAdditionExperiment(vector_size));
  #endif //USER_INPUT_MODE

  DumpDataToFile(
    output_file_name,
    sizeof(int),
    gpu_striding_results,
    gpu_interval_results,
    cpu_results
  );

  return 0;
}


ExperimentInfo
GetExperimentInfoFromUser(){
  ExperimentInfo experiment_info;

  puts("Please enter the parameters with which to run a test.");
  printf("vector size (in ints): ");
  std::cin >> experiment_info.vector_size;
  printf("number of gpu blocks:  ");
  std::cin >> experiment_info.num_gpu_blocks;
  printf("number of gpu threads: ");
  std::cin >> experiment_info.num_gpu_threads;

  return experiment_info;
}

bool
ExperimentInfoIsValid(const ExperimentInfo& experiment_info) {
  cudaDeviceProp device;
  cudaError cuda_status = cudaGetDeviceProperties(&device, 0);
  
  if (cuda_status != cudaSuccess) {
    std::cerr <<
      "Device couldn't be access, no matter what parameters you use, "
      "it's a bad experiment." << std::endl;
    return false;
  } else if (experiment_info.vector_size <= 0) {
    std::cerr << "A vector size wasn't really provided..." << std::endl;
    return false;
  } else if (experiment_info.num_gpu_blocks <= 0 ||
              experiment_info.num_gpu_blocks > device.maxGridSize[0]) {

    // since we are only using 1 dimensional grids, its ok for now
    // to only check one dimension

    std::cerr << "An invalid number of blocks was provided." << std::endl;
    return false;
  } else if (experiment_info.num_gpu_threads <= 0 ||
              experiment_info.num_gpu_threads > device.maxThreadsPerBlock) {
    std::cerr << "An invalid number of blocks was provided." << std::endl;
    return false;
  }

  return true;
}

CpuExperimentResult
PerformCpuVectorAdditionExperiment(const size_t vector_size) {
  CpuExperimentResult result = {"", 0.0, vector_size};
  int* vector_a = nullptr;
  int* vector_b = nullptr;
  int* vector_c = nullptr;
  size_t num_bytes = vector_size * sizeof(int);
  bool operation_success;

  if (vector_size <= 0) {
    result.outcome =
      OPERATION_FAILED + std::string(": invalid vector size");
    return result;
  }

  operation_success = AllocateHostVectors(
    &vector_a, &vector_b, &vector_c, vector_size);
  if (operation_success == false) {
    FreeHostVectors(&vector_a, &vector_b, &vector_c);
    result.outcome = HOST_ALLOCATION_FAILURE;
    return result;
  }

  clock_t start, end;
  start = clock();

  for (int i = 0; i < vector_size; ++i) {
    vector_c[i] = vector_a[i] + vector_b[i];
  }

  end = clock();
  result.run_time = ((float) (end - start) / CLOCKS_PER_SEC) * 1000.0;  

  if (VectorsAreSummedProperly(vector_a, vector_b, vector_c, vector_size)) {
    result.outcome = OPERATION_SUCCESS;
  } else {
    result.outcome = SUM_FAILED;
  }

  FreeHostVectors(&vector_a, &vector_b, &vector_c);

  return result;
}

GpuExperimentResult
PerformGPUVectorAdditionExperiment(
  const size_t vector_size,
  const int num_gpu_blocks,
  const int num_gpu_threads,
  const bool use_striding) {

  GpuExperimentResult result = {
    "",
    0.0,
    0.0,
    vector_size,
    num_gpu_blocks,
    num_gpu_threads,
    use_striding ? STRIDING : INTERVAL
  };

  if (vector_size <= 0) {
    result.outcome =
      OPERATION_FAILED + std::string(": invalid vector size");
    return result;
  }
  
  int* vector_a = nullptr;
  int* vector_b = nullptr;
  int* vector_c = nullptr;
  int* dev_vector_a = nullptr;
  int* dev_vector_b = nullptr;
  int* dev_vector_c = nullptr;
  size_t num_bytes = vector_size * sizeof(int);
  bool operation_success;
  
  operation_success = AllocateGpuVectors(
    &dev_vector_a, &dev_vector_b, &dev_vector_c, vector_size);
  if (operation_success == false) {
    FreeGpuVectors(dev_vector_a, dev_vector_b, dev_vector_c);
    result.outcome = GPU_ALLOCATION_FAILURE;
    return result;
  }

  operation_success = AllocateHostVectors(
    &vector_a, &vector_b, &vector_c, vector_size);
  if (operation_success == false) {
    FreeGpuVectors(dev_vector_a, dev_vector_b, dev_vector_c);
    FreeHostVectors(&vector_a, &vector_b, &vector_c);
    result.outcome = HOST_ALLOCATION_FAILURE;
    return result;
  }

  cudaEvent_t start, compute_start, compute_end, end;
  cudaEventCreate(&start);
  cudaEventCreate(&compute_start);
  cudaEventCreate(&compute_end);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  cudaMemcpy(dev_vector_a, vector_a, num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_vector_b, vector_b, num_bytes, cudaMemcpyHostToDevice);

  cudaEventRecord(compute_start, 0);

  // some lame, hacky stuff since if a GPU operation takes longer than 2
  // seconds, Windows TDR will take back the GPU on the assumption that
  // it has frozen or failed or an application is being a hog or...
  if (vector_size <= 10000000) {
    //puts("normal mode");

    if (use_striding) {
      StridingVectorAdd<<<num_gpu_blocks, num_gpu_threads>>>(
        dev_vector_c, dev_vector_a, dev_vector_b, vector_size);
    } else {
      SimpleVectorAdd<<<num_gpu_blocks, num_gpu_threads>>>(
        dev_vector_c, dev_vector_a, dev_vector_b, vector_size);
    }
  } else {
    //puts("HACK MODE!");

    int start_index = 0;
    int end_index = 5000000;

    while (start_index < vector_size) {
      if (use_striding) {
        StridingVectorAdd<<<num_gpu_blocks, num_gpu_threads>>>(
          dev_vector_c, dev_vector_a, dev_vector_b, vector_size);
      } else {
        SimpleVectorAdd<<<num_gpu_blocks, num_gpu_threads>>>(
          dev_vector_c, dev_vector_a, dev_vector_b, vector_size);
      }

      start_index = end_index;
      end_index = (start_index + end_index) >= vector_size ?
                  vector_size :
                  end_index + 5000000
      ;
    }
  }

  cudaEventRecord(compute_end, 0);
  cudaEventSynchronize(compute_end);

  cudaMemcpy(vector_c, dev_vector_c, num_bytes, cudaMemcpyDeviceToHost);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  cudaEventElapsedTime(&(result.compute_time), compute_start, compute_end);
  cudaEventElapsedTime(&(result.total_run_time), start, end);

  if (cudaGetLastError() != cudaSuccess) {
    result.outcome = OPERATION_FAILED + std::string(": kernel failed");
    return result;
  } else if (
    !VectorsAreSummedProperly(vector_a, vector_b, vector_c, vector_size)) {

    result.outcome = SUM_FAILED;
    return result;
  }

  FreeHostVectors(&vector_a, &vector_b, &vector_c);
  FreeGpuVectors(dev_vector_a, dev_vector_b, dev_vector_c);

  result.outcome = OPERATION_SUCCESS;

  return result;
}

bool
AllocateHostVectors(
  int** vector_a, int** vector_b, int** vector_c, const int size) {

  size_t num_bytes = size * sizeof(int);

  *vector_a = (int*) malloc(num_bytes);
  *vector_b = (int*) malloc(num_bytes);
  *vector_c = (int*) malloc(num_bytes);

  if (*vector_a == nullptr || *vector_b == nullptr || *vector_c == nullptr) {
    return false;
  }

  for (int i = 0; i < size; ++i) {
    (*vector_a)[i] = 1;
    (*vector_b)[i] = 1;
    (*vector_c)[i] = 0;
  }

  return true;
}

void
FreeHostVectors(int** vector_a, int** vector_b, int** vector_c) {
  free(*vector_a);
  free(*vector_b);
  free(*vector_c);
  *vector_a = nullptr;
  *vector_b = nullptr;
  *vector_c = nullptr;
}

bool
AllocateGpuVectors(
  int** vector_a, int** vector_b, int** vector_c, const int size) {

  size_t num_bytes = size * sizeof(int);
  cudaError_t status;

  status = cudaMalloc((void**) vector_a, num_bytes);
  if (status != cudaSuccess) {
    return false;
  }

  status = cudaMalloc((void**) vector_b, num_bytes);
  if (status != cudaSuccess) {
    return false;
  }

  status = cudaMalloc((void**) vector_c, num_bytes);
  if (status != cudaSuccess) {
    return false;
  }

  return true;
}

bool
FreeGpuVectors(int* vector_a, int* vector_b, int* vector_c) {
  cudaFree((void*) vector_a);
  cudaFree((void*) vector_b);
  cudaFree((void*) vector_c);
  return true;
}

bool
VectorsAreSummedProperly(
  const int* vector_a, const int* vector_b, const int* vector_c, const int vector_size) {
  for (int i = 0; i < vector_size; i++) {
    if (vector_c[i] != vector_a[i] + vector_b[i]) {
      return false;
    }
  }

  return true;
}

void
DumpDataToFile(
  const std::string& output_file_name,
  const size_t size_of_vector_element,
  const std::vector<GpuExperimentResult>& gpu_striding_results,
  const std::vector<GpuExperimentResult>& gpu_interval_results,
  const std::vector<CpuExperimentResult>& cpu_results) {

  std::ofstream fout;

  fout.clear();
  fout.open(output_file_name.c_str());

  fout << GetGpuProperties(0) << std::endl
       << "Size per element:    " << size_of_vector_element << " bytes"
       << std::endl
       << std::endl;

    for (GpuExperimentResult result : gpu_striding_results) {
    fout << result.vector_size << DELIMETER
         << result.compute_time << DELIMETER
         << result.total_run_time << DELIMETER
         << result.num_blocks << DELIMETER
         << result.num_threads << DELIMETER
         << result.work_distribution << DELIMETER
         << result.outcome << std::endl;
  }
  fout << std::endl << std::endl;

  for (GpuExperimentResult result : gpu_interval_results) {
    fout << result.vector_size << DELIMETER
         << result.compute_time << DELIMETER
         << result.total_run_time << DELIMETER
         << result.num_blocks << DELIMETER
         << result.num_threads << DELIMETER
         << result.work_distribution << DELIMETER
         << result.outcome << std::endl;
  }
  fout << std::endl << std::endl;
 
  for (CpuExperimentResult result : cpu_results) {
    fout << result.vector_size << DELIMETER
         << result.run_time << DELIMETER
         << result.outcome << std::endl;
  }

}

std::string
GetGpuProperties(const int device_number) {
  cudaDeviceProp cudaProperties;
  cudaError_t cudaStatus = cudaGetDeviceProperties(
    &cudaProperties,
    device_number
  );
  if (cudaStatus != cudaSuccess) {
    char error_message[100];
    sprintf(
      error_message,
      "Device properties could for device %i not be retrieved!",
      device_number
    );
    return std::string(error_message);
  }

  char properties[2048];
  sprintf(
    properties,
    "Device Name:         %s\n"
    "Cuda Version:        %i.%i\n"
    "Multiprocessors:     %i\n"
    "Clock Rate:          %i mHz\n"
    "Total Global Memory: %i MB\n"
    "Warp Size:           %i\n"
    "Max Threads/Block:   %i\n"
    "Max Threads-Dim:     %i x %i x %i\n"
    "Max Grid Size:       %i x %i x %i",
    cudaProperties.name,
    cudaProperties.major,
    cudaProperties.minor,
    cudaProperties.multiProcessorCount,
    cudaProperties.clockRate / 1000,
    cudaProperties.totalGlobalMem / 1000000,
    cudaProperties.warpSize,
    cudaProperties.maxThreadsPerBlock,
    cudaProperties.maxThreadsDim[0],
    cudaProperties.maxThreadsDim[1],
    cudaProperties.maxThreadsDim[2],
    cudaProperties.maxGridSize[0],
    cudaProperties.maxGridSize[1],
    cudaProperties.maxGridSize[2]
  );

  return std::string(properties);
}