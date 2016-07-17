#ifndef _PROJECT_APSP_HPP
#define _PROJECT_APSP_HPP 1

#include <float.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <sys/time.h>

#define FLOAT_INF FLT_MAX


float
ComputeElapsedSeconds(struct timeval* start, struct timeval* end);

int
LoadMatrix(float** graph, char* fileName);

void GenerateErdosRenyiGraph(
  float** graph, unsigned size, unsigned average_degree);

void PrintGraph(FILE* stream, float* graph, unsigned size);


float
ComputeElapsedSeconds(struct timeval* start, struct timeval* end) {
/**
 * A simple method for computing the difference of timevals.
 *
 * Returns:
 *   The difference between the times indicated by the objects in seconds.
 *
 * Args:
 *   start: the earlier of the two times; when the 'stopwatch' was started.
 *   end: the later of the two times; when the 'stopwatch' was stopped.
 */

  float seconds = (float) (end->tv_usec - start->tv_usec) / 1000000.0;
  seconds += (float) (end->tv_sec - start->tv_sec);
  return seconds;
}

int
LoadMatrix(float** graph, char* fileName) {
/**
 * Loads adjacency matrix data from a file into an array (passed by "reference")
 *
 * Returns:
 *   The size of the graph (i.e. n of n x n fame)
 * Args:
 *   graph: a pointer to an array of floating point values that will represent
 *          an adjacency matrix (returned value)
 *   fileName: the name of the file to read graph data from
 */

  FILE* file = fopen(fileName, "r");

  if (file == NULL) {
    fprintf(stderr, "Failed to read file.\n");
    exit(1);
  }

  int rows, cols, num_non_zero;
  fscanf(file, "%d\t%d\t%d\n", &rows, &cols, &num_non_zero);

  *graph = (float*) malloc((rows * cols) * sizeof(float));
  float* g = *graph;

  int v1, v2;
  float weight;

  for (v1 = 0; v1 < rows; ++v1) {
    for (v2 = 0; v2 < cols; ++v2) {
      g[v1 * cols + v2] = FLOAT_INF;
    }
    g[v1 * cols + v1] = 0.0;
  }

  unsigned i;  
  for (i = num_non_zero; i > 0; --i) {
    fscanf(file, "%d\t%d\t%f\n", &v1, &v2, &weight);
    g[v1 * rows + v2] = weight;
  }

  fclose(file);

  return rows;
}

void
GenerateErdosRenyiGraph(float** graph, unsigned size, unsigned average_degree) {
/**
 * Generates a (di)graph in adjacency matrix representation with randomly
 * existing edges with random weights. For simplicity of use of the graph,
 * edge weights are kept below an arbitrary size. The adjacency matrix is
 * mapped into a 1-dimensional array in a row-major format.
 */
  unsigned num_elements = size * size;
  *graph = (float*) malloc(num_elements * sizeof(float));
  float* g = *graph;

  unsigned i;
  unsigned j;
  for(i = 0; i < size; ++i) {
    for (j = 0; j < size; ++j) {
      if (i != j) {
        unsigned rv = rand() % average_degree; // TODO: account for self edge
        if (rv == 0) {
          g[i * size + j] = (float) (rand() % 9999);
        } else {
          g[i * size + j] = FLOAT_INF;
        }
      } else {
        g[i * size + j] = 0;
      }
    }
  }
}

void
PrintGraph(FILE* stream, float* graph, unsigned size) {
  unsigned i, j;
  for (i = 0; i < size; ++i) {
    for (j = 0; j < size; ++j) {
      if (graph[i * size + j] >= (FLOAT_INF - 5.0)) {
        fprintf(stream, "\t   !!!  ");
      } else {
        fprintf(stream, "\t%8.2f", graph[i * size + j]);
      }
    }
    fprintf(stream, "\n");
  }
}

// bool
// graphsAreEquivalent(float* one, float* other, unsigned size) {
//   return graphsAreEquivalent(one, other, size, 0.001);
// }

// bool
// graphsAreEquivalent(float* one, float* other, unsigned size, float tolerance) {
// /**
//  *
//  */
//   unsigned i;
//   for (i = 0; i < size; ++i) {
//     if (abs(one[i] - other[i]) > tolerance) {
//       return false;
//     }
//   }

//   return true;
// }

#endif //_PROJECT_APSP_HPP