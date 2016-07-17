/*
  This header demonstrates how we build cuda programs spanning
  multiple files. 
 */

#ifndef ADD_H_
#define ADD_H_

// Necessary headers in VS - my guess: not compiling directly with nvcc
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//---

// This is the number of elements we want to process.
#define N 1024

// This is the declaration of the function that will execute on the GPU.
__global__ void add(int*, int*, int*);

#endif // ADD_H_