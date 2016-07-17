#ifndef _SEQUENTIAL_APSP_
#define _SEQUENTIAL_APSP_ 1


void
NaiveFloydWarshall(float* graph, unsigned size);

void
NaiveFloydWarshall(float* graph, unsigned size) {
/**
 * Executes the plain, old Floyd Warshall algorithm on the given adjacency
 * matrix; currently does not also create the predecessor matrix
 *
 * Args:
 *   graph: an (n * n) x 1, row major adjacency matrix
 *   size: the size of each row/column n
 */
  unsigned i, j, k;
  for (k = 0; k < size; ++k) {
    for (i = 0; i < size; ++i) {
      for (j = 0; j < size; ++j) {
        graph[i * size + j] = fminf(
          graph[i * size + j],
          graph[i * size + k] + graph[k * size + j]
        );
      }
    }
  }
}

#endif // _SEQUENTIAL_APSP_