#ifndef _VECTOR_ADD_HPP_
#define _VECTOR_ADD_HPP_

#include "cuda_runtime.h"

/**
 * A simple vector addition method that stores the result in the
 * given array.
 *
 * Implicitly, 
 */
__global__
void
SimpleVectorAdd(
  int* result, const int* vector_a, const int* vector_b, const int vector_size);

/**
 *  
 */
__global__
void
StridingVectorAdd(
  int* result, const int* vector_a, const int* vector_b, const int vector_size);

#endif //_VECTOR_ADD_HPP