#ifndef _PROJECT_APSP_HPP
#define _PROJECT_APSP_HPP 1

#include <float.h>
#include <cstdlib>
#include <cstdio>
#include <math.h>
//#include <sys/time.h>

#define FLOAT_INF FLT_MAX


float easy_graph_a[] = {      0.0,       1.0, FLOAT_INF, FLOAT_INF,
                        FLOAT_INF,       0.0,       1.0,       1.1,
                        FLOAT_INF, FLOAT_INF,       0.0,       0.5,
                        FLOAT_INF, FLOAT_INF, FLOAT_INF,       0.0};

float easy_graph_b[] = {      0.0,       1.0,       2.0, FLOAT_INF,
                        FLOAT_INF,       0.0,       0.5,       1.1,
                        FLOAT_INF, FLOAT_INF,       0.0,       0.5,
                        FLOAT_INF, FLOAT_INF, FLOAT_INF,       0.0};

float
ComputeElapsedSeconds(struct timeval* start, struct timeval* end);

int
LoadSparseMatrix(float** graph, char* fileName);

void GenerateErdosRenyiGraph(
  float** graph, unsigned size, unsigned average_degree);

void PrintGraph(FILE* stream, float* graph, unsigned size);


#if 0
float
ComputeElapsedSeconds(struct timeval* start, struct timeval* end) {
/**
 *
 */

  float seconds = (float) (end->tv_usec - start->tv_usec) / 1000000.0;
  seconds += (float) (end->tv_sec - start->tv_sec);
  return seconds;
}
#endif

int
LoadSparseMatrix(float** graph, char* fileName) {
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


int
WriteSparseMatrix(
  const float* matrix,
  const unsigned width,
  const char* file_name) {
/**
 *
 */
  FILE* file = fopen(file_name, "w");
  
  unsigned nnz = 0;
  for (unsigned i = 0; i < width * width; ++i) {
    if (matrix[i] != FLOAT_INF) {
      nnz++;
    }
  }  

  fprintf(file, "%d\t%d\t%d\n", width, width, nnz);
  for (unsigned i = 0; i < width; ++i) {
    for (unsigned j = 0; j < width; ++j) {
      if (matrix[i * width + j] != FLOAT_INF) {
        fprintf(file, "%d\t%d\t%f\n", i, j, matrix[i * width + j]);
      }
    }
  }

  fclose(file);

  return 0;
}

void
WriteGraph(FILE* file, const float* graph, const unsigned size) {
/**
 *
 */
  fprintf(file, "%d, %d, %d\n", size, size, -1);

  unsigned i, j, k;
  for (i = 0; i < size; ++i) {
    for (j = 0; j < size; ++j) {
      k = i * size + j;
      fprintf(file, "%4d %4d %f\n", i, j, graph[k]);
    }
  }

}


void
GenerateErdosRenyiGraph(float** graph, unsigned size, unsigned average_degree) {
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


 bool
 GraphsAreEquivalent(const float* one, const float* other, unsigned size, float tolerance) {
 /**
  *
  */
   unsigned i;
   for (i = 0; i < size * size; ++i) {
     if (fabs(one[i] - other[i]) > tolerance) {
       return false;
     }
   }

   return true;
 }


 bool
 GraphsAreEquivalentDefault(const float* one, const float* other, unsigned size) {
 /**
  *
  */
   return GraphsAreEquivalent(one, other, size, 0.01);
 }



#endif //_PROJECT_APSP_HPP