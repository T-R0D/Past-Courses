/**
 *
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <numeric>
#include "reduce.hpp"

#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <ctime>


/**
 *  A simple structure for easy handling of vector reduction trial results.
 */
typedef struct VectorReduceResult {
  std::string device_type;
  unsigned vector_size;
  float compute_time_sec;
  float data_xfer_time_sec;
  unsigned num_blocks;
  unsigned threads_per_block;
  std::string completion_status;

  VectorReduceResult() {
    device_type = "";
    compute_time_sec = 0.0;
    data_xfer_time_sec = 0.0;
    num_blocks = 1;
    threads_per_block = 1;
    completion_status = "NOT STARTED";
  };
};


const std::string RELATIVE_REPORT_FOLDER_PATH = "../../../Report/";


/**
 * Conducts a trial using the single CPU sequential version of vector reduction.
 */
VectorReduceResult
RunSequentialCpuVectorReduceTrial(unsigned vector_size);

/**
 * Conducts a trial of vector reduce on the GPU, collects the necessary
 * data transfer times, and other pertinent data.
 */
VectorReduceResult
RunGpuVectorReduceTrial(
  unsigned vector_size, unsigned num_blocks, unsigned threads_per_block);

/**
 * Dynamically allocates a vector with the given parameters and fills the
 * values in so that the results of summing can be checked appropriately.
 */
bool
AllocateTestVectorOnHost(int** test_vector, const unsigned size);

/**
 * Allocates a test vector on the GPU and returns it by reference.
 * Return value indicates status of the CUDA operation.
 */
cudaError
AllocateTestVectorMemoryOnGpu(int** dev_vector, const unsigned size);

/**
 * My reimplementation of the std::accumulate. I don't know what optimiztiaons
 * std::accumulate may or may not use, so I don't know if it is fine to use
 * it when calculating speed-up.
 */
int
myaccumulate(const int* input_vector, unsigned size);

/**
 * Writes the results of all of the trials to a plain, large .csv file.
 */
bool
DumpResultsToFile(
  const std::string& file_path,
  const std::string& file_name,
  const std::string& delimeter,
  const std::vector<VectorReduceResult>& results);

/**
 * Gets relevant GPU properties from the installed device and puts them into
 * a formated string.
 */
std::string
GetGpuProperties(const int device_number);

void debugIntermediateVectorSum(int* d_arr, int size) {
  int bytes = size * sizeof(int);
  int* p = (int*) malloc(bytes);
  cudaMemcpy(p, d_arr, bytes, cudaMemcpyDeviceToHost);
  std::cout<<"intermediate sum: "<<std::accumulate(p, p+size,0)<<std::endl;
  free(p);
}

void debugIntermediateVectorArray(int* d_arr, int size) {
  int bytes = size * sizeof(int);
  int* p = (int*) malloc(bytes);
  cudaMemcpy(p, d_arr, bytes, cudaMemcpyDeviceToHost);
  std::cout<<"values:"<<std::endl;
  unsigned i = 0;
  while (i < size) {
    std::cout<<p[i]<<" ";
    i++;
  }
  std::cout<<std::endl;
  free(p);
}


/**
 * A test harness for collecting data on vector reduction using both the CPU
 * and the GPU.
 */
int
main(const int argc, const char** argv) {
  cudaDeviceProp gpu_properties;
  cudaError cudaStatus = cudaGetDeviceProperties(&gpu_properties, 0);

  if (cudaStatus == cudaSuccess) {
    std::vector<VectorReduceResult> results;
    unsigned vector_size;
    unsigned num_blocks;
    unsigned threads_per_block;

    for (vector_size =  21000000;
         vector_size >=  1000000;
         vector_size -=  1000000) {

      std::cout << "Using vector of " << vector_size << " values" << std::endl;

      for (double jitter = 0.9; jitter < 1.2; jitter += 0.1) {

        for (num_blocks = gpu_properties.maxGridSize[0];
             num_blocks >= 1;
             num_blocks = ((num_blocks + 1) / 2) - 1) {

          for (threads_per_block = gpu_properties.maxThreadsPerBlock;
               threads_per_block >= 3; // 1
               threads_per_block /= 2) {

            if (num_blocks != 1 || threads_per_block != 1) {
              results.push_back(
                RunGpuVectorReduceTrial(
                  vector_size * jitter,
                  num_blocks,
                  threads_per_block
                )
              );

              std::cout << ".";
            }
          }
        }

        results.push_back(
          RunSequentialCpuVectorReduceTrial(vector_size * jitter)
         );
        std::cout << "." << std::endl;
      }
    }

    DumpResultsToFile(
      RELATIVE_REPORT_FOLDER_PATH,
      "results.csv",
      ", ",
      results
    );
  } else {
    printf(
      "The GPU doesn't seem to be functioning properly.\r\n"
      "Please check the meaning of error code %i and fix the issue.\r\n",
      cudaStatus
    );
    return 1;
  }

  return 0;
}


VectorReduceResult
RunSequentialCpuVectorReduceTrial(unsigned vector_size) {
  VectorReduceResult result;
  result.device_type = "Windows with i5";
  result.vector_size = vector_size;

  int* test_vector = nullptr;
  bool success = AllocateTestVectorOnHost(&test_vector, vector_size);
  if (!success) {
    result.completion_status = "Failure to allocate host memory";
    return result;
  }

  clock_t compute_start, compute_end;

  compute_start = clock();

  int sum = myaccumulate(test_vector, vector_size);


  compute_end = clock() + 1;

  result.compute_time_sec =
    (double) (((double) compute_end) - (double) compute_start) /
    (double) CLOCKS_PER_SEC;

  result.completion_status = "Complete";

  free(test_vector); test_vector = nullptr;

  return result;
}

VectorReduceResult
RunGpuVectorReduceTrial(
  unsigned vector_size, unsigned num_blocks, unsigned threads_per_block) {

  VectorReduceResult result;
  result.vector_size = vector_size;
  result.num_blocks = num_blocks;
  result.threads_per_block = threads_per_block;

  cudaDeviceProp device_prop;
  cudaError cuda_status = cudaGetDeviceProperties(&device_prop, 0);
  if (cuda_status != cudaSuccess) {
    result.completion_status = "GPU NOT FOUND!";
    return result;
  }
  result.device_type = device_prop.name;

  int* test_vector = nullptr;
  int* d_test_vector = nullptr;
  int* d_intermediate_vector = nullptr;
  int* d_sum = nullptr;
  int sum;

  cuda_status = AllocateTestVectorMemoryOnGpu(&d_test_vector, vector_size);
  if(cuda_status != cudaSuccess) {
    result.completion_status = "GPU ALLOCATION FAILURE!";
    return result;
  }
  cuda_status = AllocateTestVectorMemoryOnGpu(&d_intermediate_vector, num_blocks);
  if(cuda_status != cudaSuccess) {
    cudaFree(d_test_vector);
    result.completion_status = "GPU ALLOCATION FAILURE!";
    return result;
  }
  cuda_status = AllocateTestVectorMemoryOnGpu(&d_sum, 1);
  if(cuda_status != cudaSuccess) {
    cudaFree(d_test_vector);
    cudaFree(d_intermediate_vector);
    result.completion_status = "GPU ALLOCATION FAILURE!";
    return result;
  }
  bool success = AllocateTestVectorOnHost(&test_vector, vector_size);
  if (!success) {
    cudaFree(d_test_vector);
    cudaFree(d_intermediate_vector);
    cudaFree(d_sum);
    result.completion_status = "Failure to allocate host memory";
    return result;
  }

  cudaEvent_t total_start, compute_start, compute_end, total_end, debug_sync;
  cudaEventCreate(&total_start);
  cudaEventCreate(&compute_start);
  cudaEventCreate(&compute_end);
  cudaEventCreate(&total_end);
  cudaEventCreate(&debug_sync);
  cudaEventRecord(total_start, 0);


  cudaMemcpy(
    d_test_vector,
    test_vector,
    vector_size * sizeof(int),
    cudaMemcpyHostToDevice
  );

  cudaEventRecord(compute_start, 0);

  unsigned n_blocks = num_blocks;
  unsigned v_size = vector_size;
  unsigned block_base_share = v_size / n_blocks;
  unsigned remainder = v_size % n_blocks;
  unsigned shared_bytes_needed = (block_base_share + 1) * sizeof(int);

  EasyReduceKernel
    <<<n_blocks, threads_per_block, threads_per_block * sizeof(int)>>>(
    d_intermediate_vector,
    d_test_vector,
    v_size
  );

  while (n_blocks > device_prop.maxThreadsPerBlock) {
    v_size = n_blocks;
    n_blocks /= 2;

    EasyReduceKernel
      <<<n_blocks, threads_per_block, threads_per_block * sizeof(int)>>>(
      d_intermediate_vector,
      d_intermediate_vector,
      v_size
    );
  }

  EasyReduceKernel
    <<<1, n_blocks, n_blocks * sizeof(int)>>>(
    d_sum,
    d_intermediate_vector,
    n_blocks
  );

  cudaEventRecord(compute_end, 0);
  cudaEventSynchronize(compute_end);

  cudaMemcpy(&sum, d_sum, 1 * sizeof(int), cudaMemcpyDeviceToHost);

  if (sum != std::accumulate(test_vector, test_vector + vector_size, 0)) {
    std::cout<<"size: "<<vector_size<<std::endl<<"blocks: "<<num_blocks<<std::endl<<"threads: "<<threads_per_block<<std::endl;
    std::cout<<"sum: "<<sum<<std::endl<<"myacc: "<<std::accumulate(test_vector, test_vector + vector_size, 0)<<std::endl;
    std::cout<<std::endl<<std::endl;
    debugIntermediateVectorSum(d_intermediate_vector, n_blocks);
    debugIntermediateVectorArray(d_intermediate_vector, n_blocks);
    assert(sum == myaccumulate(test_vector, vector_size));
  }

  cudaEventRecord(total_end, 0);
  cudaEventSynchronize(total_end);

  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    result.completion_status = "GPU kernel failed to complete!";
    result.completion_status +=
      " " + std::string(cudaGetErrorName(cuda_status));
    return result;
  } else if (
    sum != std::accumulate(test_vector, test_vector + vector_size, 0)) {
    result.completion_status = "GPU failed to reduce correctly!";
    return result;
  }

  cudaEventElapsedTime(&(result.compute_time_sec), compute_start, compute_end);
  result.compute_time_sec /= 1000.0;

  cudaEventElapsedTime(&(result.data_xfer_time_sec), total_start, total_end);
  result.data_xfer_time_sec /= 1000.0;
  result.data_xfer_time_sec -= result.compute_time_sec;

  result.completion_status = "Complete";

  cudaFree(d_test_vector);
  cudaFree(d_intermediate_vector);
  cudaFree(d_sum);
  free(test_vector); test_vector = nullptr;

  return result;

}

bool
AllocateTestVectorOnHost(int** test_vector, const unsigned size) {
  unsigned num_bytes = size * sizeof(int);

  *test_vector = (int*) malloc(size * sizeof(int));
  if (*test_vector == nullptr) {
std::cout<<"failed to allocate "<<num_bytes<<" on host"<<std::endl;
    return false;
  }

  for (unsigned i = 0; i < size; i++) {
    (*test_vector)[i] = 1;
  }

  return true;
}

cudaError
AllocateTestVectorMemoryOnGpu(int** dev_vector, const unsigned size) {
  size_t num_bytes = size * sizeof(int);
  return cudaMalloc(dev_vector, num_bytes);
}

int
myaccumulate(const int* input_vector, unsigned size) {
  int sum = 0;
  for (unsigned i = 0; i < size; ++i) {
    sum += input_vector[i];
  }
  return sum;
}

bool
DumpResultsToFile(
  const std::string& file_path,
  const std::string& file_name,
  const std::string& delimeter,
  const std::vector<VectorReduceResult>& results) {

  std::ofstream fout;
  fout.clear();
  fout.open(file_path + file_name);

  if (!fout.good()) {
    puts("Invalid output destination!");
    return false;
  }

  fout << "Device Type" << delimeter
       << "Vector Size" << delimeter
       << "Compute Time (s)" << delimeter
       << "Data Transfer Time (s)" << delimeter
       << "Total Time (s)" << delimeter
       << "Compute Throughput (Flops)" << delimeter
       << "Total Throughput (Flops)" << delimeter
       << "Number of GPU Blocks" << delimeter
       << "Number of Threads per Block" << delimeter
       << "Completion Status" << std::endl;

  for (VectorReduceResult result : results) {
    double total_time = result.compute_time_sec + result.data_xfer_time_sec;
    fout << result.device_type << delimeter
         << result.vector_size << delimeter
         << result.compute_time_sec << delimeter
         << result.data_xfer_time_sec << delimeter
         << total_time << delimeter
         << result.vector_size / result.compute_time_sec << delimeter
         << result.vector_size / total_time << delimeter
         << result.num_blocks << delimeter
         << result.threads_per_block << delimeter
         << result.completion_status << std::endl;
  }
  fout.close();

  fout.clear();
  fout.open(file_path + "gpu_properties.csv");
  fout << GetGpuProperties(0) << std:: endl;
  fout.close();
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
    "Device Name,         %s\r\n"
    "Cuda Version,        %i.%i\r\n"
    "Multiprocessors,     %i\r\n"
    "Clock Rate,          %i mHz\r\n"
    "Total Global Memory, %i MB\r\n"
    "Warp Size,           %i\r\n"
    "Max Threads/Block,   %i\r\n"
    "Max Threads-Dim,     %i x %i x %i\r\n"
    "Max Grid Size,       %i x %i x %i\r\n"
    "SharedMem/Block,     %i KB",
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
    cudaProperties.maxGridSize[2],
    cudaProperties.sharedMemPerBlock / 1000
  );

  return std::string(properties);
}