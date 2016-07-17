/*
  This program demonstrates the basics of working with cuda. We use
  the GPU to add two arrays. We also introduce cuda's approach to
  error handling and timing using cuda Events.

  This is the main program. You should also look at the header add.h
  for the important declarations, and then look at add.cu to see how
  to define functions that execute on the GPU.
 */

//
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//
#include <iostream>
#include <string>

#include "add.h"

std::string
GetGpuProperties(const int device_number);

int main() {
  
  // Arrays on the host (CPU)
  int a[N], b[N], c[N];
  
  /*
    These will point to memory on the GPU - notice the correspondence
    between these pointers and the arrays declared above.
   */
  int *dev_a, *dev_b, *dev_c;

  /*
    These calls allocate memory on the GPU (also called the
    device). This is similar to C's malloc, except that instead of
    directly returning a pointer to the allocated memory, cudaMalloc
    returns the pointer through its first argument, which must be a
    void**. The second argument is the number of bytes we want to
    allocate.

    NB: the return value of cudaMalloc (like most cuda functions) is
    an error code. Strictly speaking, we should check this value and
    perform error handling if anything went wrong. We do this for the
    first call to cudaMalloc so you can see what it looks like, but
    for all other function calls we just point out that you should do
    error checking.

    Actually, a good idea would be to wrap this error checking in a
    function or macro, which is what the Cuda By Example book does.
   */
  cudaError_t err = cudaMalloc( (void**) &dev_a, N * sizeof(int));
  if (err != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
  cudaMalloc((void**) &dev_b, N * sizeof(int));
  cudaMalloc((void**) &dev_c, N * sizeof(int));

  // These lines just fill the host arrays with some data so we can do
  // something interesting. Well, so we can add two arrays.
  for (int i = 0; i < N; ++i) {
    a[i] = i;
    b[i] = i;
  }

  /*
    The following code is responsible for handling timing for code
    that executes on the GPU. The cuda approach to this problem uses
    events. For timing purposes, an event is essentially a point in
    time. We create events for the beginning and end points of the
    process we want to time. When we want to start timing, we call
    cudaEventRecord.

    In this case, we want to record the time it takes to transfer data
    to the GPU, perform some computations, and transfer data back.
  */
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord( start, 0 );

  /*
    Once we have host arrays containing data and we have allocated
    memory on the GPU, we have to transfer data from the host to the
    device. Again, notice the similarity to C's memcpy function.

    The first argument is the destination of the copy - in this case a
    pointer to memory allocated on the device. The second argument is
    the source of the copy. The third argument is the number of bytes
    we want to copy. The last argument is a constant that tells
    cudaMemcpy the direction of the transfer.
   */
  cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_c, c, N * sizeof(int), cudaMemcpyHostToDevice);
  
  /*
    FINALLY we get to run some code on the GPU. At this point, if you
    haven't looked at add.cu (in this folder), you should. The
    comments in that file explain what the add function does, so here
    let's focus on how add is being called. The first thing to notice
    is the <<<...>>>, which you should recognize as _not_ being
    standard C. This syntactic extension tells nvidia's cuda compiler
    how to parallelize the execution of the function. We'll get into
    details as the course progresses, but for we'll say that <<<N,
    1>>> is creating N _blocks_ of 1 _thread_ each. Each of these
    threads is executing add with a different data element (details of
    the indexing are in add.cu). 

    In larger programs, you will typically have many more blocks, and
    each block will have many threads. Each thread will handle a
    different piece of data, and many threads can execute at the same
    time. This is how cuda can get such large speedups.
   */
  add<<<N, 1>>>(dev_a, dev_b, dev_c);

  /*
    Unfortunately, the GPU is to some extent a black box. In order to
    print the results of our call to add, we have to transfer the data
    back to the host. We do that with a call to cudaMemcpy, which is
    just like the cudaMemcpy calls above, except that the direction of
    the transfer (given by the last argument) is reversed. In a real
    program we would want to check the error code returned by this
    function.
  */
  cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

  /*
    This is the other end of the timing process. We record an event,
    synchronize on it, and then figure out the difference in time
    between the start and the stop.

    We have to call cudaEventSynchronize before we can safely _read_
    the value of the stop event. This is because the GPU may not have
    actually written to the event until all other work has finished.
   */
  cudaEventRecord( end, 0 );
  cudaEventSynchronize( end );

  float elapsedTime;
  cudaEventElapsedTime( &elapsedTime, start, end );

  /*
    Let's check that the results are what we expect.
   */
  for (int i = 0; i < N; ++i) {
    if (c[i] != a[i] + b[i]) {
      std::cerr << "Oh no! Something went wrong. You should check your cuda install and your GPU. :(" << std::endl;

      // clean up events - we should check for error codes here.
      cudaEventDestroy( start );
      cudaEventDestroy( end );

      // clean up device pointers - just like free in C. We don't have
      // to check error codes for this one.
      cudaFree(dev_a);
      cudaFree(dev_b);
      cudaFree(dev_c);
      exit(1);
    }
  }

  /*
    Let's let the user know that everything is ok and then display
    some information about the times we recorded above.
   */
  std::cout << "Yay! Your program's results are correct." << std::endl;
  std::cout << "Your program took: " << elapsedTime << " ms." << std::endl;

  std::cout << GetGpuProperties(0) << std::endl;
  
  // Cleanup in the event of success.
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  return 0;
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
    "Multiprocessors:     %i\n"
    "Clock Rate:          %i mHz\n"
    "Total Global Memory: %i MB\n"
    "Warp Size:           %i\n"
    "Max Threads/Block:   %i\n"
    "Max Threads-Dim:     %i x %i x %i\n"
    "Max Grid Size:       %i x %i x %i",
    cudaProperties.name,
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