/**
 *
 */

#ifndef _REDUCE_CU_
#define _REDUCE_CU_ 1

#include "reduce.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>


__global__
void
EasyReduceKernel(int* result_vector, const int* input_vector, const unsigned size) {
  extern __shared__ int shared_data[];

  const unsigned& block_tid = threadIdx.x;
  unsigned global_tid = (blockIdx.x * blockDim.x) + block_tid;
  unsigned int total_threads = gridDim.x * blockDim.x;

  /* fetch the data from global into shared memory */
  shared_data[block_tid] = 0;
  for (unsigned i = global_tid; i < size; i += total_threads) {
    shared_data[block_tid] += input_vector[i];
  }

  unsigned power_2_padded_num_threads_in_block = 1;
  while (power_2_padded_num_threads_in_block < blockDim.x) {
    power_2_padded_num_threads_in_block *= 2;
  }
  __syncthreads();

  /* perform the reduction */
  unsigned stride = power_2_padded_num_threads_in_block / 2;

  // unroll the first iteration so the special check to see if threads are
  // accessing valid memory with their stride
  if (block_tid  + stride < blockDim.x && stride > 0) {
    shared_data[block_tid] += shared_data[block_tid + stride];
  }
  __syncthreads();

  // the main algorithm
  for (unsigned stride = power_2_padded_num_threads_in_block / 4;
       stride > 0;
       stride /= 2) {

    if (block_tid < stride) {
      shared_data[block_tid] += shared_data[block_tid + stride];
    }
    __syncthreads();
  }

  // TODO: Optimize things for the last warp. Unfortunately, if we want it to
  //       be portable, it will involve a lot of if-logic that may not be
  //       worth implementing for programming and performance reasons. 

  /* store the final result */
  if (block_tid == 0) {
    result_vector[blockIdx.x] = shared_data[0];
  }
}


__global__
void
ReduceKernel(
  int* result,
  const int* input_vector,
  const unsigned size,
  const unsigned block_base_share,
  const unsigned remainder) {

  extern __shared__ int shared_data[];

  // compute shares and indices
  unsigned block_share = block_base_share;
  unsigned block_start_index = block_base_share * blockIdx.x; 
  unsigned block_end_index;
  const unsigned& block_tid = threadIdx.x;
  
  if (blockIdx.x < remainder) {
    block_share++;
    block_start_index += blockIdx.x;
  } else {
    block_start_index += remainder;
  }
  block_end_index = block_start_index + block_share;

  unsigned power_2_padded_size = 1;
  while (power_2_padded_size < blockDim.x) {
    power_2_padded_size *= 2;
  }

  // fetch memory
  // a simple striding approach should do
  shared_data[block_tid] = 0;
  for (unsigned i = block_start_index + block_tid;
       i < block_end_index;
       i += blockDim.x) {
   
if(i >= size){
  result[0] = 666;
  return;
}

    shared_data[block_tid] += input_vector[i];
  }
  __syncthreads();

  // the O(lg n) threads algorithm
  // TODO: move the first iteration out of the loop for performance
  for (unsigned stride = power_2_padded_size / 2; stride > warpSize; stride /= 2) {
    if (block_tid + stride < blockDim.x) {
      shared_data[block_tid] += shared_data[block_tid + stride];
    }
    __syncthreads();
  }

  // using only one warp should be fast
  if (block_tid < warpSize) {
    for (unsigned stride = warpSize; stride > 0; stride /= 2) {
      if (block_tid + stride < blockDim.x) {
        shared_data[block_tid] += shared_data[block_tid + stride];
      }
    }
  }

  if (block_tid == 0) {
    result[blockIdx.x] = shared_data[0];
  }
}

#endif //_REDUCE_CU_