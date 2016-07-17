#ifndef _VECTOR_ADD_CPP_
#define _VECTOR_ADD_CPP_

#include "vector_add.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__
void
SimpleVectorAdd(
  int* result,
  const int* vector_a,
  const int* vector_b,
  const int vector_size) {

  int num_threads = blockDim.x * gridDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int base_share = vector_size / num_threads;
  int remainder = vector_size % num_threads;
  int start_index = tid * base_share;
  int end_index;

  if (tid < remainder) {
    start_index += tid;
    end_index = start_index + base_share + 1;
  } else {
    start_index += remainder;
    end_index = start_index + base_share;
  }

  for (int i = start_index; i < end_index; ++i) {
  	result[i] = vector_a[i] + vector_b[i];
  }
}

__global__
void
StridingVectorAdd(
  int* result,
  const int* vector_a,
  const int* vector_b,
  const int vector_size) {

  int num_threads = blockDim.x * gridDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = tid; i < vector_size; i += num_threads) {
  	result[i] = vector_a[i] + vector_b[i];
  }
}

__global__
void
SegmentedSimpleVectorAdd(
  int* result,
  const int* vector_a,
  const int* vector_b,
  const int start_index,
  const int end_index) {

  int segment_size = end_index - start_index;
  int num_threads = blockDim.x * gridDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int base_share = segment_size / num_threads;
  int remainder = segment_size % num_threads;
  int start = start_index + (tid * base_share);
  int end;

  if (tid < remainder) {
    start += tid;
    end = start + base_share + 1;
  } else {
    start += remainder;
    end = start + base_share;
  }

  for (int i = start; i < end; ++i) {
  	result[i] = vector_a[i] + vector_b[i];
  }
}

__global__
void
SegmentedStridingVectorAdd(
  int* result,
  const int* vector_a,
  const int* vector_b,
  const int start_index,
  const int end_index) {

  int num_threads = blockDim.x * gridDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = tid + start_index; i < end_index; i += num_threads) {
  	result[i] = vector_a[i] + vector_b[i];
  }
}

#endif //_VECTOR_ADD_CPP_