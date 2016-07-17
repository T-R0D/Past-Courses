/**
 *  All-pairs-shortest paths
 *	Device code (column-wise)
 *  Recursive in-place implementation 
 *  Copyright by Aydin Buluc
 *  June 2008
 */

#ifndef _I_APSP_KERNEL_H_
#define _I_APSP_KERNEL_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <common_functions.h>

#include <stdio.h>

#include <float.h>


/**
 * APSP using a single block (column version)
 * Iteration is within this kernel function, 
 * So, no looping is necessary when calling apsp_seq
 * start is the starting offset of nboth dimensions
 */
__global__
void
apsp_seq_I(float * A, const unsigned start, const unsigned multiplicand_width, const unsigned full_width) {
  // Thread index
  int t_x = threadIdx.x;
  int t_y = threadIdx.y;

  // Csub is used to store the element of the result
  // that is computed by the thread

  unsigned offset = start * full_width + start;	// memory offset
  unsigned result_idx = offset + t_y * full_width + t_x;

  for (int k = 0; k < multiplicand_width; ++k) {
    float weight_1 = A[offset + t_y * full_width + k  ];		// kth row
    float weight_2 = A[offset + k   * full_width + t_x];		// kth column

    A[result_idx] = fminf(weight_1 + weight_2, A[result_idx]);

    __syncthreads();
  }
}


__global__
void
apsp_seq_I_shared(float * A, const unsigned start, const unsigned multiplicand_width, const unsigned full_width) {
  // Thread index
  int t_x = threadIdx.x;
  int t_y = threadIdx.y;

  // Csub is used to store the element of the result
  // that is computed by the thread

  unsigned offset = start * full_width + start;	// memory offset
  unsigned result_idx = offset + t_y * full_width + t_x;
  unsigned shared_result_idx = t_y * multiplicand_width + t_x;

  extern __shared__ float shared_mem[];
  shared_mem[shared_result_idx] = A[result_idx];
  __syncthreads();


  for (int k = 0; k < multiplicand_width; ++k) {
    float weight_1 = shared_mem[t_y * multiplicand_width + k  ];		// kth row
    float weight_2 = shared_mem[k   * multiplicand_width + t_x];		// kth column

    shared_mem[shared_result_idx] = fminf(weight_1 + weight_2, shared_mem[shared_result_idx]);

    __syncthreads();
  }

  A[result_idx] = shared_mem[shared_result_idx];
}



__global__
void
apsp_seq_I_shared2(float * A, const unsigned start, const unsigned multiplicand_width, const unsigned full_width) {
  // Thread index
  int t_x = threadIdx.x;
  int t_y = threadIdx.y;

  unsigned offset = start * full_width + start;	// memory offset
  unsigned thread_width = blockDim.x;
  unsigned result_idx = offset + t_y * full_width + t_x;
  unsigned shared_result_idx = t_y * multiplicand_width + t_x;

  extern __shared__ float shared_mem[];
  shared_mem[shared_result_idx] = A[result_idx];
  shared_mem[shared_result_idx  + thread_width * multiplicand_width] = A[result_idx + thread_width * full_width];
  shared_mem[shared_result_idx + thread_width] = A[result_idx + thread_width];
  shared_mem[shared_result_idx + thread_width * multiplicand_width + thread_width] = A[result_idx + thread_width * full_width + thread_width];
  __syncthreads();


  float part_1;
  float part_2;
  for (int k = 0; k < multiplicand_width; ++k) {
    part_1 = shared_mem[t_y * multiplicand_width + k  ];		// kth row
    part_2 = shared_mem[k   * multiplicand_width + t_x];		// kth column
    shared_mem[shared_result_idx] = fminf(part_1 + part_2, shared_mem[shared_result_idx]);
    //__syncthreads();

    part_1 = shared_mem[(t_y + thread_width) * multiplicand_width + k  ];		// kth row
    part_2 = shared_mem[k   * multiplicand_width + t_x];		// kth column
    shared_mem[shared_result_idx + thread_width * multiplicand_width] =
      fminf(part_1 + part_2, shared_mem[shared_result_idx + thread_width * multiplicand_width]);
    //__syncthreads();

    part_1 = shared_mem[t_y * multiplicand_width + k  ];		// kth row
    part_2 = shared_mem[k   * multiplicand_width + t_x + thread_width];		// kth column
    shared_mem[shared_result_idx + thread_width] =
      fminf(part_1 + part_2, shared_mem[shared_result_idx + thread_width]);
    //__syncthreads();

    part_1 = shared_mem[(t_y + thread_width) * multiplicand_width + k  ];		// kth row
    part_2 = shared_mem[k   * multiplicand_width + t_x + thread_width];		// kth column
    shared_mem[shared_result_idx + thread_width * multiplicand_width + thread_width] =
      fminf(part_1 + part_2, shared_mem[shared_result_idx + thread_width * multiplicand_width + thread_width]);
    __syncthreads();
  }

  A[result_idx] = shared_mem[shared_result_idx];
  A[result_idx + thread_width * full_width] = shared_mem[shared_result_idx  + thread_width * multiplicand_width];
  A[result_idx + thread_width] = shared_mem[shared_result_idx + thread_width];
  A[result_idx + thread_width * full_width + thread_width] = shared_mem[shared_result_idx + thread_width * multiplicand_width + thread_width];
}



__global__
void
apsp_seq_I_shared_looped(float * A, const unsigned start, const unsigned multiplicand_width, const unsigned full_width) {
  #define BLOCKING_FACTOR 2 // makes 4 sub-blocks

  // Thread index
  int t_x = threadIdx.x;
  int t_y = threadIdx.y;

  unsigned offset = start * full_width + start;	// memory offset
  unsigned thread_width = blockDim.x;
  unsigned result_idx = offset + t_y * full_width + t_x;
  unsigned shared_result_idx = t_y * multiplicand_width + t_x;

  extern __shared__ float shared_mem[];

  #pragma unroll BLOCKING_FACTOR
  for (unsigned y = 0; y < BLOCKING_FACTOR; ++y) {
    #pragma unroll BLOCKING_FACTOR
    for (unsigned x = 0; x < BLOCKING_FACTOR; ++x) {
      unsigned shared_offset =
        ((y * thread_width) * multiplicand_width) + (x * thread_width);
      unsigned full_offset =
        ((y * thread_width) * full_width) + (x * thread_width);
      shared_mem[shared_result_idx + shared_offset] =
        A[result_idx + full_offset];
    }
  }
  __syncthreads();


  float part_1;
  float part_2;
  for (int k = 0; k < multiplicand_width; ++k) {
    #pragma unroll BLOCKING_FACTOR
    for (unsigned y = 0; y < BLOCKING_FACTOR; ++y) {
      #pragma unroll BLOCKING_FACTOR
      for (unsigned x = 0; x < BLOCKING_FACTOR; ++x) {
        part_1 = shared_mem[(t_y + (y * thread_width)) * multiplicand_width + k  ];		// kth row
        part_2 = shared_mem[k   * multiplicand_width + (t_x + (x * thread_width))];		// kth column
        shared_mem[shared_result_idx + ((y * thread_width) * multiplicand_width) + (x * thread_width)] =
          fminf(part_1 + part_2, shared_mem[shared_result_idx + ((y * thread_width) * multiplicand_width) + (x * thread_width)]);
      }
    }
    __syncthreads();
  }

  #pragma unroll BLOCKING_FACTOR
  for (unsigned y = 0; y < BLOCKING_FACTOR; ++y) {
    #pragma unroll BLOCKING_FACTOR
    for (unsigned x = 0; x < BLOCKING_FACTOR; ++x) {
      unsigned shared_offset =
        ((y * thread_width) * multiplicand_width) + (x * thread_width);
      unsigned full_offset =
        ((y * thread_width) * full_width) + (x * thread_width);
      A[result_idx + full_offset] =
        shared_mem[shared_result_idx + shared_offset];
    }
  }

  #undef BLOCKING_FACTOR
}



__global__
void
apsp_seq_I_shared_looped_4(float * A, const unsigned start, const unsigned multiplicand_width, const unsigned full_width) {
  #define BLOCKING_FACTOR 4 // makes 16 sub-blocks

  // Thread index
  int t_x = threadIdx.x;
  int t_y = threadIdx.y;

  unsigned offset = start * full_width + start;	// memory offset
  unsigned thread_width = blockDim.x;
  unsigned result_idx = offset + t_y * full_width + t_x;
  unsigned shared_result_idx = t_y * multiplicand_width + t_x;

  extern __shared__ float shared_mem[];

  #pragma unroll BLOCKING_FACTOR
  for (unsigned y = 0; y < BLOCKING_FACTOR; ++y) {
    #pragma unroll BLOCKING_FACTOR
    for (unsigned x = 0; x < BLOCKING_FACTOR; ++x) {
      unsigned shared_offset =
        ((y * thread_width) * multiplicand_width) + (x * thread_width);
      unsigned full_offset =
        ((y * thread_width) * full_width) + (x * thread_width);
      shared_mem[shared_result_idx + shared_offset] =
        A[result_idx + full_offset];
    }
  }
  __syncthreads();


  float part_1;
  float part_2;
  for (int k = 0; k < multiplicand_width; ++k) {
    #pragma unroll BLOCKING_FACTOR
    for (unsigned y = 0; y < BLOCKING_FACTOR; ++y) {
      #pragma unroll BLOCKING_FACTOR
      for (unsigned x = 0; x < BLOCKING_FACTOR; ++x) {
        part_1 = shared_mem[(t_y + (y * thread_width)) * multiplicand_width + k  ];		// kth row
        part_2 = shared_mem[k   * multiplicand_width + (t_x + (x * thread_width))];		// kth column
        shared_mem[shared_result_idx + ((y * thread_width) * multiplicand_width) + (x * thread_width)] =
          fminf(part_1 + part_2, shared_mem[shared_result_idx + ((y * thread_width) * multiplicand_width) + (x * thread_width)]);
      }
    }
    __syncthreads();
  }

  #pragma unroll BLOCKING_FACTOR
  for (unsigned y = 0; y < BLOCKING_FACTOR; ++y) {
    #pragma unroll BLOCKING_FACTOR
    for (unsigned x = 0; x < BLOCKING_FACTOR; ++x) {
      unsigned shared_offset =
        ((y * thread_width) * multiplicand_width) + (x * thread_width);
      unsigned full_offset =
        ((y * thread_width) * full_width) + (x * thread_width);
      A[result_idx + full_offset] =
        shared_mem[shared_result_idx + shared_offset];
    }
  }

  #undef BLOCKING_FACTOR
}


/**
 * Matrix multiplication on the device: C = A * B (column-major)
 * multiplicand_width is A's and B's width
 * Each block uses shared memory of (nIt * 2 * 16 * 16 * 4) = 2048 bytes (beta=16, sizeof(WORD)=4)
 * nIt is at most BLOCK_DIM/2 but does not affect the amount of shared memory used 
 * each multiprocessor can execute at most 8 blocks simultaneously (due to shared memory constraints) 
 */
__global__ void
matrixMul_I( float * C, float * A, float * B, int full_matrix_width, int multiplicand_width, int beta, int sCx, int sCy, int sAx, int sAy, int sBx, int sBy, int add)
{
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Remember... column-major
  int sa = sAx * full_matrix_width + sAy;
  int sb = sBx * full_matrix_width + sBy;
  int sc = sCx * full_matrix_width + sCy;
    
  int ba = beta * by;			// y-offset
  int bb = full_matrix_width * beta * bx;		// x-offset
    
  float min = FLOATINF;
    
  int nIt = multiplicand_width / beta;	// number of blocks in one dimension
    
  // Do block multiplication to update the C(i,j) block
  // Using A(i,1) * A(1,j) + A(i,2) * A(2,j) + ... + A(i,n) * A(n,j)

  extern __shared__ float AsAndBs[];
  unsigned shared_mem_offset = beta * beta;
  for(int m = 0; m < nIt; ++m) {
    AsAndBs[                    tx * beta + ty] = A[sa + ba + m * beta * full_matrix_width + tx *full_matrix_width + ty];
    AsAndBs[shared_mem_offset + tx * beta + ty] = B[sb + bb + m * beta  + tx *full_matrix_width + ty];
    __syncthreads();
    
    for(int k = 0; k < beta; ++k) {
      float a = AsAndBs[                    k  * beta + tx];	// (tx)th row
      float b = AsAndBs[shared_mem_offset + ty * beta +  k];	// (ty)th column

      min = fminf(a + b, min);
    }
    __syncthreads();    
  }
  // Write the block sub-matrix to device memory;
  // each thread writes one element

  if(add) {
    C[sc + ba + bb + ty * full_matrix_width + tx] = fminf(
      C[sc + ba + bb + ty * full_matrix_width + tx],
      min
    );
  }
  else {
    C[sc + ba + bb + ty * full_matrix_width + tx] = min;		// (tx,ty)th element
  }
}

#endif // #ifndef _APSP_KERNEL_H_