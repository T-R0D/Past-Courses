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
#include <map>

#define l std::cout<<__LINE__<<std::endl;

/**
 *  A simple structure for easy handling of vector reduction trial results.
 */
typedef struct {
  std::string deviceType;
  std::string method;
  unsigned vectorSize;
  float computeTimeSec;
  float totalTimeSec;
  unsigned numBlocks;
  unsigned threadsPerBlock;
  std::string completionStatus;

  void init() {
    deviceType = "";
    method = "";
    computeTimeSec = 0.0;
    totalTimeSec = 0.0;
    numBlocks = 1;
    threadsPerBlock = 1;
    completionStatus = "NOT STARTED";
  };
} VectorReduceResult;


// const std::string RELATIVE_REPORT_FOLDER_PATH = "../../../report/";
const std::string RELATIVE_REPORT_FOLDER_PATH = "../report/";
const std::string CPU_DEVICE = "Intel i5";
const std::string SEQUENTIAL_RUN = "sequential";
const std::string GPU_MULTI_KERNEL_RUN = "gpu with multiple kernel launches";
const std::string GPU_KERNEL_IN_KERNEL_RUN = "gpu kernel launching kernels";
const std::string GPU_WITH_CPU_FINISH = "gpu with cpu final reduction";
const std::string GPU_WITH_THREAD_FENCE = "gpu with thread fencing technique";
const unsigned GPU_MULTI_KERNEL_CODE = 0;
const unsigned GPU_KERNEL_IN_KERNEL_CODE = 1;
const unsigned GPU_WITH_CPU_FINISH_CODE = 2;
const unsigned GPU_WITH_THREAD_FENCE_CODE = 3;


/**
 * Conducts a trial using the single CPU sequential version of vector reduction.
 */
VectorReduceResult
RunSequentialCpuVectorReduceTrial(unsigned vectorSize);

/**
 * Conducts a trial of vector reduce on the GPU, collects the necessary
 * data transfer times, and other pertinent data.
 */
VectorReduceResult
RunGpuVectorReduceTrial(
  unsigned vectorSize,
  unsigned numBlocks,
  unsigned threadsPerBlock,
  unsigned reduceType);

void
GpuMultiKernelReduce(
    int* d_sum,
    int* d_intermediateVector,
    int* d_vector,
    const unsigned vectorSize,
    const unsigned numBlocks,
    const unsigned threadsPerBlock,
    cudaDeviceProp& gpuProperties);

void
GpuKernelInKernelReduce(
    int* d_sum,
    int* d_intermediateVector,
    int* d_vector,
    const unsigned vectorSize,
    const unsigned numBlocks,
    const unsigned threadsPerBlock,
    cudaDeviceProp& gpuProperties);

void
GpuWithCpuFinishKernelReduce(
    int* d_sum,
    int* d_intermediateVector,
    int* d_vector,
    const unsigned vectorSize,
    const unsigned numBlocks,
    const unsigned threadsPerBlock,
    cudaDeviceProp& gpuProperties);

void
GpuThreadFenceReduce();


/**
 * Dynamically allocates a vector with the given parameters and fills the
 * values in so that the results of summing can be checked appropriately.
 */
bool
AllocateTestVectorOnHost(int** vector, const unsigned size);

/**
 * Allocates a test vector on the GPU and returns it by reference.
 * Return value indicates status of the CUDA operation.
 */
cudaError
AllocateTestVectorMemoryOnGpu(int** device_vector, const unsigned size);

/**
 * My reimplementation of the std::accumulate. I don't know what optimiztiaons
 * std::accumulate may or may not use, so I don't know if it is fine to use
 * it when calculating speed-up.
 */
int
MyAccumulate(const int* vector, unsigned size);

/**
 * Writes the results of all of the trials to a .csv file suited for plotting
 * graphs in Excel.
 */
bool
DumpResultsToExcelDataFile(
  const std::string& filePath,
  const std::string& fileName,
  const std::string& delimeter,
  const std::vector<VectorReduceResult>& results);

/**
 * Writes the results of all of the trials to a .csv file suited for plotting
 * graphs in R.
 */
bool
DumpResultsToRDataFile(
  const std::string& filePath,
  const std::string& fileName,
  const std::string& delimeter,
  const std::vector<VectorReduceResult>& results);

/**
 * Gets relevant GPU properties from the installed device and puts them into
 * a formated string for inclusion as a csv table.
 */
std::string
GetGpuProperties(const int deviceNumber);

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
  cudaDeviceProp gpuProperties;
  cudaError cudaStatus = cudaGetDeviceProperties(&gpuProperties, 0);
  //__cpuid(); //see wikipedia for example

  if (cudaStatus == cudaSuccess) {

std::cout<<GetGpuProperties(0)<<std::endl;


    std::vector<VectorReduceResult> results;
    unsigned vectorSize = 1000000;
    double jitter = 1.0;
    unsigned numBlocks = 65535;
    unsigned threadsPerBlock = 1024;

    for (vectorSize =  21000000;
         vectorSize <=  1000000;
         vectorSize -=  1000000) {

      std::cout << "Using vector of " << vectorSize << " values" << std::endl;

      for (jitter = 0.9; jitter < 1.2; jitter += 0.1) {

        for (numBlocks = gpuProperties.maxGridSize[0];
             numBlocks >= 1;
             numBlocks = ((numBlocks + 1) / 2) - 1) {

          for (threadsPerBlock = gpuProperties.maxThreadsPerBlock;
               threadsPerBlock >= 1; // 1
             threadsPerBlock -= (gpuProperties.warpSize / 2)) {

            if (threadsPerBlock >= gpuProperties.sharedMemPerBlock) continue;

            std::cout << "." << std::flush;
            results.push_back(
              RunGpuVectorReduceTrial(
                vectorSize * jitter,
                numBlocks,
                threadsPerBlock,
                GPU_MULTI_KERNEL_CODE
              )
            );

            std::cout << "^" << std::flush;
            results.push_back(
              RunGpuVectorReduceTrial(
                vectorSize * jitter,
                numBlocks,
                threadsPerBlock,
                GPU_WITH_CPU_FINISH_CODE
              )
            );
            
            std::cout << "*" << std::flush;
            results.push_back(
              RunGpuVectorReduceTrial(
                vectorSize * jitter,
                numBlocks,
                threadsPerBlock,
                GPU_KERNEL_IN_KERNEL_CODE
              )
            );
            
            std::cout << "#" << std::flush;
            // results.push_back(
            //   RunGpuVectorReduceTrial(
            //     vectorSize * jitter,
            //     numBlocks,
            //     threadsPerBlock,
            //     GPU_WITH_THREAD_FENCE_CODE
            //   )
            // );
            // std::cout << "/";

// std::cout<<results.back().completionStatus<<std::endl;
          }
        }

        results.push_back(
          RunSequentialCpuVectorReduceTrial(vectorSize * jitter)
         );
        std::cout << "|" << std::endl;
      }
    }

    DumpResultsToExcelDataFile(
      RELATIVE_REPORT_FOLDER_PATH,
      "results_big2_excel.csv",
      ", ",
      results
    );
    DumpResultsToExcelDataFile(
      RELATIVE_REPORT_FOLDER_PATH,
      "results_big2_r.csv",
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
RunSequentialCpuVectorReduceTrial(unsigned vectorSize) {
  VectorReduceResult result; result.init();
  result.deviceType = CPU_DEVICE;
  result.vectorSize = vectorSize;

  int* testVector = NULL;
  bool success = AllocateTestVectorOnHost(&testVector, vectorSize);
  if (!success) {
    result.completionStatus = "Failure to allocate host memory";
    return result;
  }

  clock_t computeStart, computeEnd;

  computeStart = clock();

  int sum = MyAccumulate(testVector, vectorSize);

  computeEnd = clock() + 1; // +1 is smoothing for really fast runs

  result.computeTimeSec =
    (double) (((double) computeEnd) - (double) computeStart) /
    (double) CLOCKS_PER_SEC;
  result.totalTimeSec = result.computeTimeSec;

  result.completionStatus = "Complete";

  free(testVector); testVector = NULL;

  return result;
}

VectorReduceResult
RunGpuVectorReduceTrial(
    unsigned vectorSize,
    unsigned numBlocks,
    unsigned threadsPerBlock,
    unsigned reduceType) {

  VectorReduceResult result; result.init();
  result.vectorSize = vectorSize;
  result.numBlocks = numBlocks;
  result.threadsPerBlock = threadsPerBlock;  cudaDeviceProp gpuProperties;
  cudaError cudaStatus = cudaGetDeviceProperties(&gpuProperties, 0);
  if (cudaStatus != cudaSuccess) {
    result.completionStatus = "GPU NOT FOUND!";
    return result;
  }
  result.deviceType = gpuProperties.name;

  int* testVector = NULL;
  int* d_testVector = NULL;
  int* intermediateVector = NULL;
  int* d_intermediateVector = NULL;
  int* d_sum = NULL;
  int sum;  cudaStatus = AllocateTestVectorMemoryOnGpu(&d_testVector, vectorSize);
  if(cudaStatus != cudaSuccess) {
    result.completionStatus = "GPU ALLOCATION FAILURE!";
    return result;
  }

  cudaStatus = AllocateTestVectorMemoryOnGpu(
    &d_intermediateVector, numBlocks);
  if(cudaStatus != cudaSuccess) {
    cudaFree(d_testVector);
    result.completionStatus = "GPU ALLOCATION FAILURE!";
    return result;
  }

  cudaStatus = AllocateTestVectorMemoryOnGpu(&d_sum, 1);
  if(cudaStatus != cudaSuccess) {
    cudaFree(d_testVector);
    cudaFree(d_intermediateVector);
    result.completionStatus = "GPU ALLOCATION FAILURE!";
    return result;
  }

  bool success = AllocateTestVectorOnHost(&testVector, vectorSize);
  if (!success) {
    cudaFree(d_testVector);
    cudaFree(d_intermediateVector);
    cudaFree(d_sum);
    result.completionStatus = "Failure to allocate host memory";
    return result;
  }
  success = AllocateTestVectorOnHost(&intermediateVector, numBlocks);
  if (!success) {
    cudaFree(d_testVector);
    cudaFree(d_intermediateVector);
    cudaFree(d_sum);
    free(testVector);
    result.completionStatus = "Failure to allocate host memory";
    return result;
  }  cudaEvent_t totalStart, computeStart, computeEnd, totalEnd, debug_sync;
  cudaEventCreate(&totalStart);
  cudaEventCreate(&computeStart);
  cudaEventCreate(&computeEnd);
  cudaEventCreate(&totalEnd);
  cudaEventCreate(&debug_sync);
  cudaEventRecord(totalStart, 0);

  cudaMemcpy(
    d_testVector,
    testVector,
    vectorSize * sizeof(int),
    cudaMemcpyHostToDevice
  );

  cudaEventRecord(computeStart, 0);

  switch (reduceType) {
    case GPU_MULTI_KERNEL_CODE:
      result.method = GPU_MULTI_KERNEL_RUN;
      GpuMultiKernelReduce(
        d_sum,
        d_intermediateVector,
        d_testVector,
        vectorSize,
        numBlocks,
        threadsPerBlock,
        gpuProperties
      );
      break;
    case GPU_KERNEL_IN_KERNEL_CODE:
      result.method = GPU_KERNEL_IN_KERNEL_RUN;
      GpuKernelInKernelReduce(
        d_sum,
        d_intermediateVector,
        d_testVector,
        vectorSize,
        numBlocks,
        threadsPerBlock,
        gpuProperties
      );
      break;
    case GPU_WITH_CPU_FINISH_CODE:
      result.method = GPU_WITH_CPU_FINISH;
      GpuWithCpuFinishKernelReduce(
        d_sum,
        d_intermediateVector,
        d_testVector,
        vectorSize,
        numBlocks,
        threadsPerBlock,
        gpuProperties
      );
    
      cudaEventRecord(computeEnd, 0);
      cudaEventSynchronize(computeEnd);
      result.computeTimeSec = 

      cudaMemcpy(
        intermediateVector, 
        d_intermediateVector,
        numBlocks * sizeof(int),
        cudaMemcpyDeviceToHost
      );

      sum = MyAccumulate(intermediateVector, numBlocks);
      break;
    //case GPU_WITH_THREAD_FENCE_CODE:
    //  result.method = GPU_WITH_THREAD_FENCE;
    //  GpuThreadFenceReduce();
    //  break;
    default:
      result.completionStatus = "Invalid method selected.";
      return result;
      // break;
  }

  if (reduceType != GPU_WITH_CPU_FINISH_CODE) {
  cudaEventRecord(computeEnd, 0);
  cudaEventSynchronize(computeEnd);
    cudaStatus = cudaMemcpy(
      &sum,
      d_sum,
      1 * sizeof(int),
      cudaMemcpyDeviceToHost
    );
  }

  cudaEventRecord(totalEnd, 0);
  cudaEventSynchronize(totalEnd);  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    result.completionStatus = "GPU kernel failed to complete";
    result.completionStatus += ": " + std::string(cudaGetErrorName(cudaStatus));
    return result;
  } else if (
    sum != std::accumulate(testVector, testVector + vectorSize, 0)) {
    result.completionStatus = "GPU failed to reduce correctly!";
    return result;
  }

  cudaEventElapsedTime(&(result.computeTimeSec), computeStart, computeEnd);
  result.computeTimeSec /= 1000.0;

  cudaEventElapsedTime(&(result.totalTimeSec), totalStart, totalEnd);
  result.totalTimeSec /= 1000.0;

  result.completionStatus = "Complete";  cudaFree(d_testVector); d_testVector = NULL;
  cudaFree(d_intermediateVector); d_intermediateVector = NULL;
  cudaFree(d_sum); d_sum = NULL;
  free(testVector); testVector = NULL;

  return result;

}

void
GpuMultiKernelReduce(
    int* d_sum,
    int* d_intermediateVector,
    int* d_vector,
    const unsigned vectorSize,
    const unsigned numBlocks,
    const unsigned threadsPerBlock,
    cudaDeviceProp& gpuProperties) {

  unsigned nBlocks = numBlocks;
  unsigned vSize = vectorSize;
  unsigned sharedBytesNeeded = (threadsPerBlock) * sizeof(int);  PlainReduceKernel
    <<<nBlocks, threadsPerBlock, sharedBytesNeeded>>>(
    d_intermediateVector,
    d_vector,
    vSize
  );

  while (nBlocks > (gpuProperties.maxThreadsPerBlock * 2)) {
    vSize = nBlocks;
    nBlocks /= 2;

    PlainReduceKernel
      <<<nBlocks, threadsPerBlock, sharedBytesNeeded>>>(
      d_intermediateVector,
      d_intermediateVector,
      vSize
    );
  }  PlainReduceKernel // TODO: fewer threads than elements for this call?
    <<<1, nBlocks, nBlocks * sizeof(int)>>>(
    d_sum,
    d_intermediateVector,
    nBlocks
  );
}

void
GpuKernelInKernelReduce(
    int* d_sum,
    int* d_intermediateVector,
    int* d_vector,
    const unsigned vectorSize,
    const unsigned numBlocks,
    const unsigned threadsPerBlock,
    cudaDeviceProp& gpuProperties) {

  KernelInKernelReduceKernel
  <<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(
    d_sum,
    d_intermediateVector,
    d_vector,
    vectorSize
  );
}

void
GpuWithCpuFinishKernelReduce(
    int* d_sum,
    int* d_intermediateVector,
    int* d_vector,
    const unsigned vectorSize,
    const unsigned numBlocks,
    const unsigned threadsPerBlock,
    cudaDeviceProp& gpuProperties) {

  unsigned nBlocks = numBlocks;
  unsigned vSize = vectorSize;
  unsigned sharedBytesNeeded = (threadsPerBlock) * sizeof(int);

  PlainReduceKernel
    <<<nBlocks, threadsPerBlock, sharedBytesNeeded>>>(
    d_intermediateVector,
    d_vector,
    vSize
  );

  // because we are timing different intervals, the cpu reduction will take
  // outside of the function call
  // ... the memcpy and all.
}



void
GpuThreadFenceReduce() {/* TODO: implement */}



bool
AllocateTestVectorOnHost(int** testVector, const unsigned size) {
  unsigned num_bytes = size * sizeof(int);

  *testVector = (int*) malloc(size * sizeof(int));
  if (*testVector == NULL) {
std::cout<<"failed to allocate "<<num_bytes<<" on host"<<std::endl;
    return false;
  }

  for (unsigned i = 0; i < size; i++) {
    (*testVector)[i] = 1;
  }

  return true;
}

cudaError
AllocateTestVectorMemoryOnGpu(int** dev_vector, const unsigned size) {
  size_t num_bytes = size * sizeof(int);
  return cudaMalloc(dev_vector, num_bytes);
}

int
MyAccumulate(const int* input_vector, unsigned size) {
  int sum = 0;
  for (unsigned i = 0; i < size; ++i) {
    sum += input_vector[i];
  }
  return sum;
}

bool
DumpResultsToExcelDataFile(
  const std::string& file_path,
  const std::string& file_name,
  const std::string& delimeter,
  const std::vector<VectorReduceResult>& results) {

  std::ofstream fout;
  fout.clear();
  fout.open((file_path + file_name).c_str());

  if (!fout.good()) {
    puts("Invalid output destination!");
    return false;
  }

  std::map<unsigned, double> seqeuntialTimes;
  for (unsigned i = 0; i < results.size(); ++i) {
    VectorReduceResult result = results[i];
    if (result.deviceType == CPU_DEVICE) {
      seqeuntialTimes[result.vectorSize] = result.computeTimeSec;
    }
  }

  fout << "Device Type" << delimeter
       << "Method" <<delimeter
       << "Vector Size" << delimeter
       << "Compute Time (s)" << delimeter
       << "Total Time (s)" << delimeter
       << "Compute Throughput (Flops)" << delimeter
       << "Total Throughput (Flops)" << delimeter
       << "Compute Speedup" << delimeter
       << "Total Speedup" << delimeter
       << "Number of GPU Blocks" << delimeter
       << "Number of Threads per Block" << delimeter
       << "Completion Status" << std::endl;

  for (unsigned i = 0; i < results.size(); ++i) {
  	VectorReduceResult result = results[i];
    double sequentialTime = seqeuntialTimes[result.vectorSize];

    fout << result.deviceType << delimeter
         << result.method << delimeter
         << result.vectorSize << delimeter
         << result.computeTimeSec << delimeter
         << result.totalTimeSec << delimeter
         << result.vectorSize / result.computeTimeSec << delimeter
         << result.vectorSize / result.totalTimeSec << delimeter
         << sequentialTime / result.computeTimeSec << delimeter
         << sequentialTime / result.totalTimeSec << delimeter
         << result.numBlocks << delimeter
         << result.threadsPerBlock << delimeter
         << result.completionStatus << std::endl;
  }
  fout.close();

  fout.clear();
  fout.open((file_path + "gpu_properties.csv").c_str());
  fout << GetGpuProperties(0) << std:: endl;
  fout.close();

  return true;
}

bool
DumpResultsToRDataFile(
  const std::string& file_path,
  const std::string& file_name,
  const std::string& delimeter,
  const std::vector<VectorReduceResult>& results) {

  std::ofstream fout;
  fout.clear();
  fout.open((file_path + file_name).c_str());

  if (!fout.good()) {
    puts("Invalid output destination!");
    return false;
  }

  std::map<unsigned, double> seqeuntialTimes;
  for (unsigned i = 0; i < results.size(); ++i) {
  	VectorReduceResult result = results[i];
    if (result.deviceType == CPU_DEVICE) {
      seqeuntialTimes[result.vectorSize] = result.computeTimeSec;
    }
  }

  fout << "Device Type" << delimeter
       << "Method" << delimeter
       << "Vector Size" << delimeter
       << "Runtime (s)" << delimeter
       << "Throughput (int/s)" << delimeter
       << "Speedup" << delimeter
       << "GPU Blocks" << delimeter
       << "Threads/Block" << delimeter
       << "Completion Status" << std::endl;

  for (unsigned i = 0; i < results.size(); ++i) {
  	VectorReduceResult result = results[i];
    double sequentialTime = seqeuntialTimes[result.vectorSize];

    fout << result.deviceType << delimeter
         << result.method << delimeter
         << result.vectorSize << delimeter
         << result.computeTimeSec << delimeter
         << result.vectorSize / result.computeTimeSec << delimeter
         << sequentialTime / result.computeTimeSec << delimeter
         << result.numBlocks << delimeter
         << result.threadsPerBlock << delimeter
         << result.completionStatus << std::endl;

    if (result.deviceType != CPU_DEVICE) {
      fout << result.deviceType << delimeter
           << result.method << delimeter
           << result.vectorSize << delimeter
           << result.totalTimeSec << delimeter
           << result.vectorSize / result.totalTimeSec << delimeter
           << sequentialTime / result.totalTimeSec << delimeter
           << result.numBlocks << delimeter
           << result.threadsPerBlock << delimeter
           << result.completionStatus << std::endl;
    }
  }
  fout.close();

  fout.clear();
  fout.open((file_path + "gpu_properties.csv").c_str());
  fout << GetGpuProperties(0) << std:: endl;
  fout.close();

  return true;
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
    "ATTRIBUTE,VALUE\n"
    "Device Name,%s\n"
    "Cuda Version,%i.%i\n"
    "Multiprocessors,%i\n"
    "CUDA Cores,96\n"
    "Clock Rate,%i mHz\n"
    "Total Global Memory, %i MB\n"
    "Warp Size,%i\n"
    "Max Threads/Block,%i\n"
    "Max Threads-Dim,%i x %i x %i\n"
    "Max Grid Size,%i x %i x %i\n"
    "SharedMem/Block,%i KB",
    cudaProperties.name,
    (int) cudaProperties.major,
    (int) cudaProperties.minor,
    (int) cudaProperties.multiProcessorCount,
    (int) cudaProperties.clockRate / 1000,
    (int) cudaProperties.totalGlobalMem / 1000000,
    (int) cudaProperties.warpSize,
    (int) cudaProperties.maxThreadsPerBlock,
    (int) cudaProperties.maxThreadsDim[0],
    (int) cudaProperties.maxThreadsDim[1],
    (int) cudaProperties.maxThreadsDim[2],
    (int) cudaProperties.maxGridSize[0],
    (int) cudaProperties.maxGridSize[1],
    (int) cudaProperties.maxGridSize[2],
    (int) cudaProperties.sharedMemPerBlock / 1000
  );

  return std::string(properties);
}