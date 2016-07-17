#ifndef _GPU_APSP_HPP
#define _GPU_APSP_HPP 1

#include <cmath>

#include "../Common/project_apsp.hpp"

__global__
void
ParallelSquareFloydWarshall(
  float* A, unsigned A_width, unsigned start, unsigned block_width);


__device__
unsigned
GetDataIndexFromMatrixIndices(unsigned x, unsigned y, unsigned matrix_width) {
  return (x * matrix_width) + y;
}



__global__
void
ParallelSquareFloydWarshall(
  float* A, unsigned A_width, unsigned start, unsigned block_width) {
/**
 * Executes the typical Floyd-Warshall algorithm but in for one GPU
 * thread in parallel.
 */

  // TODO: if applicable, turn into a shared mem using function (at least try)

  unsigned t_x = threadId.x;
  unsigned t_y = threadId.y;
  unsigned offset = (start * A_width) + start; // points to start of block
  unsigned result_index = offset + (t_x * A_width) + t_y;

  unsigned k;
  for (k = 0; k < block_width; ++k) {
    float path_part_1 = A[offset + (t_x * A_width) + k  ];
    float path_part_2 = A[offset + (k   * A_width) + t_y];

    A[result_index] = fminf(path_part_1 + path_part_2, A[result_index]);

    __syncthreads();
  }
}

__global__
void
SPMatrixMultiply(
  float* C, float* A, float* B,
  unsigned matrix_width,
  unsigned start_C_x, unsigned start_C_y, 
  unsigned start_A_x, unsigned start_A_y, 
  unsigned start_B_x, unsigned start_B_y,
  unsigned matrix_multiplicand_width,
  bool addition) {
/**
 * Shortest Paths Matrix Multiply: C = A %*% B or C = C + A %*% B
 * Can function both as a multiply or a multiply add function
 */

  // capture the block and thread indices
  unsigned block_width = blockDim.x;
  unsigned b_x = blockId.x;
  unsigned b_y = blockId.y;
  unsigned t_x = threadId.x;
  unsigned t_y = threadId.y;

  // compute the data array indices from the M[i, j] notations
  start_A = (start_A_x * matrix_width) + start_A_y;
  start_B = (start_B_x * matrix_width) + start_B_y;
  start_C = (start_C_x * matrix_width) + start_C_y;

  // compute the upper left index for this block (within the larger matrix)
  unsigned block_start_x = matrix_width * blockDim.x * b_x;
  unsigned block_start_y = blockDim.x * b_y;

  float min = FLOAT_INF;

  // rounds of computation required to tile the given matrices or sub-matrices
  unsigned blocks_required = matrix_multiplicand_width / block_width;
  unsigned i;
  for (i = 0; i < blocks_required; ++i) {
    // fetch into shared memory
    __shared__ float A_sub[block_width * block_width];
    __shared__ float B_sub[block_width * block_width];

    // each thread loads one element for a complete fetch in one go
    unsigned sub_matrix_index = t_x * block_width + t_y;
    A_sub[sub_matrix_index] =
      A[
        (start_A + block_start_x) +
        (i * block_width * matrix_width) +
        (t_x * matrix_width) + ty
      ];
    B_sub[sub_matrix_index] =
      B[
        (start_B + block_start_y) +
        (i * block_width) +
        (t_x * matrix_width) + ty
      ];
    __syncthreads();

    // perform a portion of the matrix multiplication
    unsigned k;
    for (k = 0; k < block_width; ++k) {
      float path_part_1 = A_sub[t_x * block_width + k  ];
      float path_part_2 = B_sub[k   * block_width + t_y];
      min = fminf(path_part_1 + path_part_2, min);
    }
    __syncthreads();
  }

  // write the final result
  unsigned result_index = start_C + block_start_x + block_start_y + t_x * matrix_width + t_y;
  if (addition) {
    C[result_index] = fminf(C[result_index], min);
  } else {
    C[result_index] = min;
  }

}


#endif //_GPU_APSP_HPP