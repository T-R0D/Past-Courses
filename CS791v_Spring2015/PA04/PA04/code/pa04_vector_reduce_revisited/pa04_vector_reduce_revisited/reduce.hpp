/**
 *
 */

#ifndef _REDUCE_HPP_
#define _REDUCE_HPP_ 1

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


/**
 * Reduces a vector to an smaller one of size equal to the number of blocks
 * specified for the kernel launch.
 */
__global__
void
PartialReduceKernel(
  int* resultVector,
  const int* inputVector,
  const unsigned vectorSize);

/**
 * Reduces a vector using successive kernel launches launched from the GPU
 * if necessary.
 */
__global__
void
KernelInKernelReduceKernel(
  int* sum,
  int* intermediateVector,
  const int* inputVector,
  const unsigned vectorSize);

/**
 * Reduces a vector by first reducing the vector to a number of elements equal
 * to the number of blocks used, then having one block complete the reduction.
 */
__global__
void
ThreadFenceReduceKernel(
  int* result,
  const int* inputVector,
  const unsigned vectorSize);


#endif //_REDUCE_HPP_