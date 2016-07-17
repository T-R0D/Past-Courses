/**
 *
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "mandelbrot.hpp"
#include "complex_val.hpp"
#include "my_pgm.h"

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
typedef struct MandelbrotResult {
  std::string deviceType;
  std::string method;
  unsigned iterations;
  unsigned imageSize;
  float computeTimeSec;
  float dataXferTimeSec;
  unsigned numBlocks;
  unsigned threadsPerBlock;
  std::string completionStatus;

  MandelbrotResult() {
    deviceType = "";
    method = "striding";
    iterations = 1024;
    computeTimeSec = 0.0;
    dataXferTimeSec = 0.0;
    numBlocks = 1;
    threadsPerBlock = 1;
    completionStatus = "NOT STARTED";
  };
};


const std::string RELATIVE_REPORT_FOLDER_PATH = "../../../Report/";


/**
 * Conducts a trial using the single CPU sequential version of vector reduction.
 */
MandelbrotResult
RunSequentialCpuMandelbrotTrial(
  const unsigned imageSize,
  const unsigned maxMandelbrotIterations);

/**
 * Conducts a trial of vector reduce on the GPU, collects the necessary
 * data transfer times, and other pertinent data.
 */
MandelbrotResult
RunGpuMandelbrotTrial(
  const unsigned imageSize,
  const unsigned numBlocks,
  const unsigned threadsPerBlock,
  const unsigned maxMandelbrotIterations);


/**
 * Writes the results of all of the trials to a plain, large .csv file.
 */
bool
DumpResultsToFile(
  const std::string& file_path,
  const std::string& file_name,
  const std::string& delimeter,
  const std::vector<MandelbrotResult>& results);

/**
 * Gets relevant GPU properties from the installed device and puts them into
 * a formated string.
 */
std::string
GetGpuProperties(const int device_number);



ComplexValue
ComputeComplexCoordinates(
  const unsigned row,
  const unsigned column,
  const unsigned imageWidth,
  const unsigned imageHeight,
  const int complexMax,
  const int complexMin) {

  ComplexValue complexCoordinate = {0.0, 0.0};
  double realScale = 0;
  double imaginaryScale = 0;

  realScale = (complexMax - complexMin) / (double) imageWidth;
  imaginaryScale = (complexMax - complexMin) / (double) imageWidth;

  complexCoordinate.real = complexMin + ((double) row * realScale );
  complexCoordinate.imaginary = complexMin + ((double) column * imaginaryScale);

  return complexCoordinate;
}

unsigned
CalculateMandelbrotPixel(
  const ComplexValue& complexCoordinate,
  const unsigned maxIterations) {

  int iteration = 0;
  double temp = 0;
  double length_squared = 0;
  ComplexValue z = {0.0, 0.0};

  while((length_squared < 4.0 ) /* instead of a sqare root calculation*/ &&
        (iteration <= maxIterations )) {
    // compute new z_real value (a^2 - b^2 + c)
    temp = (z.real * z.real) - (z.imaginary * z.imaginary);
    temp += complexCoordinate.real;

    // compute z_imaginary (2a_i * b_i + c_i), using the previous z_real
    z.imaginary = (2 * (z.real * z.imaginary)) + complexCoordinate.imaginary;

    z.real = temp;

    length_squared = (z.real * z.real) + (z.imaginary * z.imaginary);
    iteration++;
  }

  return iteration - 1;
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
    std::vector<MandelbrotResult> results;
    unsigned imageSize;
    double jitter;
    unsigned numBlocks;
    unsigned threadsPerBlock;
    unsigned maxMandelbrotIterations = 1024;
    PGMImageData pgmData;

    for (imageSize =   2000;
         imageSize <=  8000;
         imageSize +=  2000) {

      std::cout << "Computing image of " << imageSize << "^2" << std::endl;

      for (jitter = 0.9; jitter < 1.2; jitter += 0.1) {

        for (maxMandelbrotIterations = 1024;
             maxMandelbrotIterations <= 2048;
             maxMandelbrotIterations += 256) {

          for (numBlocks = gpu_properties.maxGridSize[0];
               numBlocks >= 20;
               numBlocks = ((numBlocks + 1) / 2) - 1) {

            for (threadsPerBlock = gpu_properties.maxThreadsPerBlock / 2;
                 threadsPerBlock >= 32; // 1
                 threadsPerBlock /= 2) {

              results.push_back(
                RunGpuMandelbrotTrial(
                  imageSize * jitter,
                  numBlocks,
                  threadsPerBlock,
                  maxMandelbrotIterations
                )
              );

                std::cout << ".";
            }
          }

          results.push_back(
            RunSequentialCpuMandelbrotTrial(imageSize, maxMandelbrotIterations)
          );
          std::cout << "." << std::endl;
        }
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


MandelbrotResult
RunSequentialCpuMandelbrotTrial(
  const unsigned imageSize,
  const unsigned maxMandelbrotIterations) {

  MandelbrotResult result;
  result.deviceType = "Windows with i5";
  result.imageSize = imageSize;

  int* pixelValues = (int*) malloc((imageSize * imageSize) * sizeof(int));
  
  if (pixelValues == nullptr) {
    result.completionStatus = "Failure to allocate host memory";
    return result;
  }

  clock_t compute_start, compute_end;
  compute_start = clock();

  for (unsigned i = 0; i < imageSize; ++i) {
    for (unsigned j = 0; j < imageSize; ++j) {
      ComplexValue complexCoordinates = ComputeComplexCoordinates(
        i,
        j,
        imageSize,
        imageSize,
        2,
        -2
      );


      pixelValues[(i * imageSize) + j] = CalculateMandelbrotPixel(
        complexCoordinates,
        maxMandelbrotIterations
      );
    }
  }

  compute_end = clock();
  result.computeTimeSec = ((float) (compute_end - compute_start) / CLOCKS_PER_SEC);

  result.completionStatus = "Complete";

  //std::cout << "here"<<std::endl;
  //char imageName[1024] = "C:/Users/Terence/Documents/GitHub/CS791v_Spring2014/PA03/Report/seqtestimage.pgm";
  //PGMImageData imageData;
  //strcpy(imageData.name, imageName);
  //imageData.width = imageSize;
  //imageData.height = imageSize;
  //imageData.num_shades = 255;
  //imageData.data = (char*) malloc((imageSize * imageSize) * sizeof(char));
  //for (unsigned i = 0; i < (imageSize * imageSize); ++i) {
  //  imageData.data[i] =
  //    (unsigned char) (
  //      ((double) pixelValues[i] / (double) maxMandelbrotIterations) * 255
  //    );
  //}
  //createPGMimage(&imageData);
  //free(imageData.data);

  free(pixelValues); pixelValues = nullptr;

  return result;
}

MandelbrotResult
RunGpuMandelbrotTrial(
  const unsigned imageSize,
  const unsigned numBlocks,
  const unsigned threadsPerBlock,
  const unsigned maxMandelbrotIterations) {

  MandelbrotResult result;
  result.method = "striding";
  result.imageSize = imageSize;
  result.numBlocks = numBlocks;
  result.threadsPerBlock = threadsPerBlock;

  cudaDeviceProp deviceProp;
  cudaError cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
  if (cudaStatus != cudaSuccess) {
    result.completionStatus = "GPU NOT FOUND!";
    return result;
  }
  result.deviceType = deviceProp.name;

  unsigned* pixelValues = nullptr;
  unsigned* d_pixelValues = nullptr;
  unsigned imageSizeInBytes = imageSize * imageSize * sizeof(unsigned);

  cudaStatus = cudaMalloc((void**) &d_pixelValues, imageSizeInBytes);
  if(cudaStatus != cudaSuccess) {
    result.completionStatus = "GPU ALLOCATION FAILURE!";
    return result;
  }
  pixelValues = (unsigned*) malloc(imageSizeInBytes);
  if (pixelValues == nullptr) {
    cudaFree(d_pixelValues);
    result.completionStatus = "Failure to allocate host memory";
    return result;
  }

  cudaEvent_t total_start, compute_start, compute_end, total_end, debug_sync;
  cudaEventCreate(&total_start);
  cudaEventCreate(&compute_start);
  cudaEventCreate(&compute_end);
  cudaEventCreate(&total_end);
  cudaEventCreate(&debug_sync);
  cudaEventRecord(total_start, 0);

  cudaEventRecord(compute_start, 0);





  NaiveMandelbrotKernel
    <<<numBlocks, threadsPerBlock>>>(
      d_pixelValues,
      imageSize,
      maxMandelbrotIterations
    );


 

  cudaEventRecord(compute_end, 0);
  cudaEventSynchronize(compute_end);

  cudaMemcpy(
    pixelValues,
    d_pixelValues,
    imageSizeInBytes,
    cudaMemcpyDeviceToHost
  );

//std::cout << "here"<<std::endl;
//  char imageName[1024] = "C:/Users/Terence/Documents/GitHub/CS791v_Spring2014/PA03/Report/testimage.pgm";
//  PGMImageData imageData;
//  strcpy(imageData.name, imageName);
//  imageData.width = imageSize;
//  imageData.height = imageSize;
//  imageData.num_shades = 255;
//  imageData.data = (char*) malloc((imageSize * imageSize) * sizeof(char));
//  for (unsigned i = 0; i < (imageSize * imageSize); ++i) {
//    imageData.data[i] =
//      (unsigned char) (
//        ((double) pixelValues[i] / (double) maxMandelbrotIterations) * 255
//      );
//  }
//  createPGMimage(&imageData);
//  free(imageData.data);


  cudaEventRecord(total_end, 0);
  cudaEventSynchronize(total_end);

  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    result.completionStatus = "GPU kernel failed to complete!";
    result.completionStatus +=
      " " + std::string(cudaGetErrorName(cudaStatus));
    return result;
  }

  cudaEventElapsedTime(&(result.computeTimeSec), compute_start, compute_end);
  result.computeTimeSec /= 1000.0;

  cudaEventElapsedTime(&(result.dataXferTimeSec), total_start, total_end);
  result.dataXferTimeSec /= 1000.0;
  result.dataXferTimeSec -= result.computeTimeSec;

  result.completionStatus = "Complete";

  cudaFree(d_pixelValues);
  free(pixelValues); pixelValues = nullptr;

  return result;
}

bool
DumpResultsToFile(
  const std::string& filePath,
  const std::string& fileName,
  const std::string& delimeter,
  const std::vector<MandelbrotResult>& results) {

  std::ofstream fout;
  fout.clear();
  fout.open(filePath + fileName);

  if (!fout.good()) {
    puts("Invalid output destination!");
    return false;
  }

  fout << "Device Type" << delimeter
       << "Method" << delimeter
       << "Iterations" << delimeter
       << "Vector Size" << delimeter
       << "Compute Time (s)" << delimeter
       << "Data Transfer Time (s)" << delimeter
       << "Total Time (s)" << delimeter
       << "Compute Throughput (Flops)" << delimeter
       << "Total Throughput (Flops)" << delimeter
       << "Number of GPU Blocks" << delimeter
       << "Number of Threads per Block" << delimeter
       << "Completion Status" << std::endl;

  for (const MandelbrotResult& result : results) {
    double totalTime = result.computeTimeSec + result.dataXferTimeSec;
    fout << result.deviceType << delimeter
         << result.method << delimeter
         << result.iterations << delimeter
         << result.imageSize << delimeter
         << result.computeTimeSec << delimeter
         << result.dataXferTimeSec << delimeter
         << totalTime << delimeter
         << result.imageSize / result.computeTimeSec << delimeter
         << result.imageSize / totalTime << delimeter
         << result.numBlocks << delimeter
         << result.threadsPerBlock << delimeter
         << result.completionStatus << std::endl;
  }
  fout.close();

  fout.clear();
  fout.open(filePath + "gpu_properties.csv");
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