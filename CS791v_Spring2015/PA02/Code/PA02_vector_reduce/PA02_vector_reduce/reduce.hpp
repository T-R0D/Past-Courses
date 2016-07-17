/**
 *
 */

#ifndef _REDUCE_HPP_
#define _REDUCE_HPP_ 1

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


/**
 * A simple version of the reduction algorithm.
 */
__global__
void
EasyReduceKernel(
  int* result_vector, const int* input_vector, const unsigned size);

/**
 * A failed attempt to try things my own way. This wasted a lot of time.
 * DON'T WASTE YOURS READING IT.
 */
__global__
void
ReduceKernel(
  int* result,
  const int* input_vector,
  const unsigned size,
  const unsigned block_base_share,
  const unsigned remainder);

#endif //_REDUCE_HPP_