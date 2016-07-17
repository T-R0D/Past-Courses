#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "project_apsp.hpp"
#include "my_process_result.hpp"
#include "sequential_apsp.hpp"
#include "gpu_apsp.hpp"
#include "old_asps.hpp"

#include <vector>
#include <iostream>

char* TEST_GRAPH = "C:/Users/Terence/Documents/GitHub/CS791v_Spring2015/Project/Code/Graphs/random_1024_1024_graph.txt";
char* TEST_SOLUTION = "C:/Users/Terence/Documents/GitHub/CS791v_Spring2015/Project/Code/Graphs/random_1024_1024_graph_solution.txt";


void GenerateGraphWithResult(const unsigned size);


MyCudaProcessResult< std::vector<float> >
DoApspOnGpu(
  const float* h_graph,
  const unsigned graph_size,
  const unsigned beta);

MyCudaProcessResult< std::vector<float> >
DoApspOnGpu(
  void(*ApspKernel)(float*, unsigned, unsigned, unsigned),
  const float* h_graph,
  const unsigned graph_size,
  const unsigned beta);


void GpuFloydWarshallBeta(float *data, int start, int width, const unsigned full_width, const unsigned beta);

void GpuFloydWarshallBeta2(
  float *data,
  int start,
  int width,
  const unsigned full_width,
  const unsigned beta);

int
main(int argc, char** argv) {
#if 0
  float* graph = easy_graph_b;
  
  for (unsigned k = 0;
   

#else
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  //GenerateGraphWithResult(4096);

  std::vector<float> f;

  puts("loading graphs...");
  float* graph;
  unsigned graph_size = LoadSparseMatrix(
    &graph,
    TEST_GRAPH
  );
  float* solution;
  LoadSparseMatrix(
    &solution,
    TEST_SOLUTION
  );

  unsigned val = 16;
  for (unsigned beta = 4; beta <= val; beta *= 2) {
    printf("================== Beta: %d =======================\n", beta);

    puts("computing...");
    MyCudaProcessResult< std::vector<float> > reference_result = DoApspOnGpu(
      graph,
      graph_size,
      beta//32
    );

    puts("verifying against solution...");
    if (GraphsAreEquivalentDefault(reference_result.GetResult().data(), solution, graph_size)) {
      puts("The solution is correct - Hooray!");
    } else {
      puts("Uh oh... SOMETHING WENT WRONG");
    }

    printf("Solution took %lf seconds to complete.\n\n", reference_result.GetTimeToComplete());

    ///////////////////////////////////////////////////////////////
    puts("computing 2nd version...");
    MyCudaProcessResult< std::vector<float> > result = DoApspOnGpu2(
      graph,
      graph_size,
      4 * beta//32
    );

    printf("\nSolution took %lf seconds to complete.\n\n", result.GetTimeToComplete());

    puts("verifying against solution...");
    if (GraphsAreEquivalentDefault(result.GetResult().data(), solution, graph_size)) {
      puts("The solution is correct - Hooray!");
    } else {
      puts("Uh oh... SOMETHING WENT WRONG");
    }

    printf("\n~~~\nSpeedup Factor: %lf\n~~~\n", reference_result.GetTimeToComplete() / result.GetTimeToComplete());
  }

  free(graph);
  free(solution);
#endif

  return 0;
}


void GenerateGraphWithResult(const unsigned size) {
  char start_file[200];
  char solution_file[200];
  sprintf(start_file, "C:/Users/Terence/Documents/GitHub/CS791v_Spring2015/Project/Code/Graphs/random_%d_%d_graph.txt", size, size);
  sprintf(solution_file, "C:/Users/Terence/Documents/GitHub/CS791v_Spring2015/Project/Code/Graphs/random_%d_%d_graph_solution.txt", size, size);

  float* graph = NULL;
  GenerateErdosRenyiGraph(&graph, size, 6);

  WriteSparseMatrix(graph, size, start_file), 

  NaiveFloydWarshall(graph, size);

  WriteSparseMatrix(graph, size, solution_file);
}


MyCudaProcessResult< std::vector<float> >
DoApspOnGpu(
  const float* h_graph,
  const unsigned graph_size,
  const unsigned beta) {
/**
 *
 */
  unsigned mem_size = graph_size * graph_size * sizeof(float);
  float* d_graph = NULL;
  MyCudaProcessResult< std::vector<float> > result;
  std::vector<float> h_result(graph_size * graph_size);

  result.CudaStatus() = cudaMalloc((void**) &d_graph, mem_size);
  if (result.CudaStatus() != cudaSuccess) {
    std::cout << cudaGetErrorName(result.CudaStatus()) << std::endl;
    return result;
  }

  result.CudaStatus() = cudaMemcpy(
    d_graph,
    h_graph,
    mem_size,
    cudaMemcpyHostToDevice
  );
  if (result.CudaStatus() != cudaSuccess) {
    std::cout << cudaGetErrorName(result.CudaStatus()) << std::endl;
    return result;
  }

  cudaEvent_t start, end;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);

  GpuFloydWarshallBeta(d_graph, 0, graph_size, graph_size, beta);
  
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  result.CudaStatus() = cudaMemcpy(
    h_result.data(),
    d_graph,
    mem_size,
    cudaMemcpyDeviceToHost
  );
  if (result.CudaStatus() != cudaSuccess) {
    std::cout << cudaGetErrorName(result.CudaStatus()) << std::endl;
    return result;
  }

  float t;
  cudaEventElapsedTime(&t, start, end);

  result.SetResult(h_result);
  result.SetTimeToComplete(t / 1000.0);
  result.SetSuccess(true);
  return result;
}

void GpuFloydWarshallBeta(
  float *data,
  int start,
  int width,
  const unsigned full_width,
  const unsigned beta) {
/**
 *
 */
  if (width <= beta) {        
    // the computation now can fit in one block
    dim3 threads(width, width);
    dim3 grid(1, 1);
    apsp_seq_I<<< grid, threads >>>(data, start, width, full_width);

  } else if(width <= FAST_GEMM)	{
    int nw = width / 2;		// new width
    unsigned shared_mem_size = 2 * beta * beta * sizeof(float); // we need 2 of them
      
    // setup execution parameters
    dim3 threadsmult(beta, beta);
    dim3 gridmult(nw / beta, nw / beta);
    
    // Do FW for A  
    GpuFloydWarshallBeta(data, start, nw, full_width, beta);

    // execute the kernel B = AB
    matrixMul_I<<< gridmult, threadsmult, shared_mem_size >>>
      (data, data, data, full_width, nw, beta, start+nw, start, start,start,start+nw, start,0);
        
    // execute the kernel C = CA
    matrixMul_I<<< gridmult, threadsmult, shared_mem_size >>>
      (data, data, data, full_width, nw, beta, start, start+nw,start,start+nw,start, start,0);

    // execute the kernel D += CB      
    matrixMul_I<<< gridmult, threadsmult, shared_mem_size >>>
      (data, data, data, full_width, nw, beta, start+nw,start+nw,start,start+nw, start+nw, start,1);

    // do FW for D
    GpuFloydWarshallBeta(data, start+nw, nw, full_width, beta);

    // execute the kernel B = BD
    matrixMul_I<<< gridmult, threadsmult, shared_mem_size >>>
      (data, data, data, full_width, nw, beta, start+nw, start, start+nw,start,start+nw, start+nw,0);

    // execute the kernel C = DC
    matrixMul_I<<< gridmult, threadsmult, shared_mem_size >>>
      (data, data, data, full_width, nw, beta, start, start+nw,start+nw,start+nw,start, start+nw,0);

    // execute the kernel A += BC
    matrixMul_I<<< gridmult, threadsmult, shared_mem_size >>>
      (data, data, data, full_width, nw, beta, start,start,start+nw,start, start, start+nw,1);

  } else {
    /*A=floyd-warshall(A);
    B=AB;
    C=CA;
    D=D+CB;
    D=floyd-warshall(D);
    B=BD;
    C=DC;
    A=A+BC;*/
        
    int nw = width / 2;		// new width

    // setup execution parameters
    dim3 gemmgrid( nw /64, nw/16 );
    dim3 gemmthreads( 16, 4 );

    // Remember: Column-major
    float * A = data + start * full_width + start;
    float * B = data + (start+nw) * full_width + start;
    float * C = data + start * full_width + (start+nw);
    float * D = data + (start+nw) * full_width + (start+nw);

    // sgemmNN_MinPlus( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
    // no need to send m & n since they are known through grid dimensions !

    // Do FW for A
    GpuFloydWarshallBeta(data, start, nw, full_width, beta);
    
    // execute the parallel multiplication kernel B = AB
    sgemmNN_MinPlus<<<gemmgrid, gemmthreads>>>(A, full_width, B, full_width, B, full_width, nw,  FLOATINF );

    // execute the parallel multiplication kernel C = CA
    sgemmNN_MinPlus<<<gemmgrid, gemmthreads>>>(C, full_width, A, full_width, C, full_width, nw,  FLOATINF );
        
     
    // execute the parallel multiplication kernel  D += CB 
    sgemmNN_MinPlus<<<gemmgrid, gemmthreads>>>(C, full_width, B, full_width, D, full_width, nw,  1 );

    // do FW for D
    GpuFloydWarshallBeta(data, start+nw, nw, full_width, beta);

    // execute the parallel multiplication kernel B = BD
    sgemmNN_MinPlus<<<gemmgrid, gemmthreads>>>(B, full_width, D, full_width, B, full_width, nw,  FLOATINF );

    // execute the parallel multiplication kernel C = DC
    sgemmNN_MinPlus<<<gemmgrid, gemmthreads>>>(D, full_width, C, full_width, C, full_width, nw,  FLOATINF );

    // execute the parallel multiplication kernel A += BC
    sgemmNN_MinPlus<<<gemmgrid, gemmthreads>>>(B, full_width, C, full_width, A, full_width, nw,  1 );
  }
}


////////////////////////////////////////////////////////////////////////////////
MyCudaProcessResult< std::vector<float> >
DoApspOnGpu2(
  const float* h_graph,
  const unsigned graph_size,
  const unsigned beta) {
/**
 *
 */
  unsigned mem_size = graph_size * graph_size * sizeof(float);
  float* d_graph = NULL;
  MyCudaProcessResult< std::vector<float> > result;
  std::vector<float> h_result;

  result.CudaStatus() = cudaMalloc((void**) &d_graph, mem_size);
  if (result.CudaStatus() != cudaSuccess) {
    std::cout << cudaGetErrorName(result.CudaStatus()) << std::endl;
    return result;
  }

  result.CudaStatus() = cudaMemcpy(
    d_graph,
    h_graph,
    mem_size,
    cudaMemcpyHostToDevice
  );
  if (result.CudaStatus() != cudaSuccess) {
    std::cout << cudaGetErrorName(result.CudaStatus()) << std::endl;
    return result;
  }

  cudaEvent_t start, end;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);

  puts("calling the new thing");
  GpuFloydWarshallBeta2(d_graph, 0, graph_size, graph_size, beta);
  
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float* h_result_ptr = (float*) malloc(graph_size * graph_size * sizeof(float));
  result.CudaStatus() = cudaMemcpy(
    h_result_ptr,
    d_graph,
    mem_size,
    cudaMemcpyDeviceToHost
  );
  if (result.CudaStatus() != cudaSuccess) {
    std::cout << cudaGetErrorName(result.CudaStatus()) << std::endl;
    return result;
  }

  for (unsigned i = 0; i < graph_size * graph_size; ++i) {
    h_result.push_back(h_result_ptr[i]);
  }

  float t;
  cudaEventElapsedTime(&t, start, end);

  result.SetResult(h_result);
  result.SetTimeToComplete(t / 1000.0);
  result.SetSuccess(true);
  return result;
}


void GpuFloydWarshallBeta2(
  float *data,
  int start,
  int width,
  const unsigned full_width,
  const unsigned beta) {
/**
 *
 */
  if (width <= beta) {        
    // the computation now can fit in one block
    dim3 threads(width / 4, width / 4);
    dim3 grid(1, 1);
    unsigned shared_mem_size = width * width * sizeof(float);
    apsp_seq_I_shared_looped_4<<< grid, threads, shared_mem_size >>>(data, start, width, full_width);

  } else if(width <= FAST_GEMM)	{
    int nw = width / 2;		// new width
    unsigned shared_mem_size = 2 * beta * beta * sizeof(float); // we need 2 of them
      
    // setup execution parameters
    dim3 threadsmult(beta, beta);
    dim3 gridmult(nw / beta, nw / beta);
    
    // Do FW for A  
    GpuFloydWarshallBeta2(data, start, nw, full_width, beta);

    // execute the kernel B = AB
    matrixMul_I<<< gridmult, threadsmult, shared_mem_size >>>
      (data, data, data, full_width, nw, beta, start+nw, start, start,start,start+nw, start,0);
        
    // execute the kernel C = CA
    matrixMul_I<<< gridmult, threadsmult, shared_mem_size >>>
      (data, data, data, full_width, nw, beta, start, start+nw,start,start+nw,start, start,0);

    // execute the kernel D += CB      
    matrixMul_I<<< gridmult, threadsmult, shared_mem_size >>>
      (data, data, data, full_width, nw, beta, start+nw,start+nw,start,start+nw, start+nw, start,1);

    // do FW for D
    GpuFloydWarshallBeta2(data, start+nw, nw, full_width, beta);

    // execute the kernel B = BD
    matrixMul_I<<< gridmult, threadsmult, shared_mem_size >>>
      (data, data, data, full_width, nw, beta, start+nw, start, start+nw,start,start+nw, start+nw,0);

    // execute the kernel C = DC
    matrixMul_I<<< gridmult, threadsmult, shared_mem_size >>>
      (data, data, data, full_width, nw, beta, start, start+nw,start+nw,start+nw,start, start+nw,0);

    // execute the kernel A += BC
    matrixMul_I<<< gridmult, threadsmult, shared_mem_size >>>
      (data, data, data, full_width, nw, beta, start,start,start+nw,start, start, start+nw,1);

  } else {
    /*A=floyd-warshall(A);
    B=AB;
    C=CA;
    D=D+CB;
    D=floyd-warshall(D);
    B=BD;
    C=DC;
    A=A+BC;*/
        
    int nw = width / 2;		// new width

    // setup execution parameters
    dim3 gemmgrid( nw /64, nw/16 );
    dim3 gemmthreads( 16, 4 );

    // Remember: Column-major
    float * A = data + start * full_width + start;
    float * B = data + (start+nw) * full_width + start;
    float * C = data + start * full_width + (start+nw);
    float * D = data + (start+nw) * full_width + (start+nw);

    // sgemmNN_MinPlus( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
    // no need to send m & n since they are known through grid dimensions !

    // Do FW for A
    GpuFloydWarshallBeta2(data, start, nw, full_width, beta);
    
    // execute the parallel multiplication kernel B = AB
    sgemmNN_MinPlus<<<gemmgrid, gemmthreads>>>(A, full_width, B, full_width, B, full_width, nw,  FLOATINF );

    // execute the parallel multiplication kernel C = CA
    sgemmNN_MinPlus<<<gemmgrid, gemmthreads>>>(C, full_width, A, full_width, C, full_width, nw,  FLOATINF );
        
     
    // execute the parallel multiplication kernel  D += CB 
    sgemmNN_MinPlus<<<gemmgrid, gemmthreads>>>(C, full_width, B, full_width, D, full_width, nw,  1 );

    // do FW for D
    GpuFloydWarshallBeta(data, start+nw, nw, full_width, beta);

    // execute the parallel multiplication kernel B = BD
    sgemmNN_MinPlus<<<gemmgrid, gemmthreads>>>(B, full_width, D, full_width, B, full_width, nw,  FLOATINF );

    // execute the parallel multiplication kernel C = DC
    sgemmNN_MinPlus<<<gemmgrid, gemmthreads>>>(D, full_width, C, full_width, C, full_width, nw,  FLOATINF );

    // execute the parallel multiplication kernel A += BC
    sgemmNN_MinPlus<<<gemmgrid, gemmthreads>>>(B, full_width, C, full_width, A, full_width, nw,  1 );
  }
}


typedef void (*ApspSeqKernel)(float*, unsigned, unsigned, unsigned);
typedef void (*ApspFunction)(ApspSeqKernel, float*, unsigned, unsigned, unsigned, unsigned);


MyCudaProcessResult< std::vector<float> >
DoApspOnGpu(
  ApspFunction apspFunction,
  ApspSeqKernel apspSeqKernel,
  const float* h_graph,
  const unsigned graph_size,
  const unsigned beta) {
/**
 *
 */
  unsigned mem_size = graph_size * graph_size * sizeof(float);
  float* d_graph = NULL;
  MyCudaProcessResult< std::vector<float> > result;
  std::vector<float> h_result;

  result.CudaStatus() = cudaMalloc((void**) &d_graph, mem_size);
  if (result.CudaStatus() != cudaSuccess) {
    std::cout << cudaGetErrorName(result.CudaStatus()) << std::endl;
    return result;
  }

  result.CudaStatus() = cudaMemcpy(
    d_graph,
    h_graph,
    mem_size,
    cudaMemcpyHostToDevice
  );
  if (result.CudaStatus() != cudaSuccess) {
    std::cout << cudaGetErrorName(result.CudaStatus()) << std::endl;
    return result;
  }

  cudaEvent_t start, end;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);

  *(ApspFunction)(apspSeqKernel, d_graph, 0, graph_size, graph_size, beta);
  
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float* h_result_ptr = (float*) malloc(graph_size * graph_size * sizeof(float));
  result.CudaStatus() = cudaMemcpy(
    h_result_ptr,
    d_graph,
    mem_size,
    cudaMemcpyDeviceToHost
  );
  if (result.CudaStatus() != cudaSuccess) {
    std::cout << cudaGetErrorName(result.CudaStatus()) << std::endl;
    return result;
  }

  for (unsigned i = 0; i < graph_size * graph_size; ++i) {
    h_result.push_back(h_result_ptr[i]);
  }

  float t;
  cudaEventElapsedTime(&t, start, end);

  result.SetResult(h_result);
  result.SetTimeToComplete(t / 1000.0);
  result.SetSuccess(true);
  return result;
}


void GpuFloydWarshallBeta(
  void (*ApspSeqKernel)(float*, unsigned, unsigned, unsigned),
  float *data,
  int start,
  int width,
  const unsigned full_width,
  const unsigned beta) {
/**
 *
 */
  if (width <= beta) {        
    // the computation now can fit in one block
    dim3 threads(width / 4, width / 4);
    dim3 grid(1, 1);
    unsigned shared_mem_size = width * width * sizeof(float);
    *ApspSeqKernel<<< grid, threads, shared_mem_size >>>(data, start, width, full_width);

  } else if(width <= FAST_GEMM)	{
    int nw = width / 2;		// new width
    unsigned shared_mem_size = 2 * beta * beta * sizeof(float); // we need 2 of them
      
    // setup execution parameters
    dim3 threadsmult(beta, beta);
    dim3 gridmult(nw / beta, nw / beta);
    
    // Do FW for A  
    GpuFloydWarshallBeta2(data, start, nw, full_width, beta);

    // execute the kernel B = AB
    matrixMul_I<<< gridmult, threadsmult, shared_mem_size >>>
      (data, data, data, full_width, nw, beta, start+nw, start, start,start,start+nw, start,0);
        
    // execute the kernel C = CA
    matrixMul_I<<< gridmult, threadsmult, shared_mem_size >>>
      (data, data, data, full_width, nw, beta, start, start+nw,start,start+nw,start, start,0);

    // execute the kernel D += CB      
    matrixMul_I<<< gridmult, threadsmult, shared_mem_size >>>
      (data, data, data, full_width, nw, beta, start+nw,start+nw,start,start+nw, start+nw, start,1);

    // do FW for D
    GpuFloydWarshallBeta2(data, start+nw, nw, full_width, beta);

    // execute the kernel B = BD
    matrixMul_I<<< gridmult, threadsmult, shared_mem_size >>>
      (data, data, data, full_width, nw, beta, start+nw, start, start+nw,start,start+nw, start+nw,0);

    // execute the kernel C = DC
    matrixMul_I<<< gridmult, threadsmult, shared_mem_size >>>
      (data, data, data, full_width, nw, beta, start, start+nw,start+nw,start+nw,start, start+nw,0);

    // execute the kernel A += BC
    matrixMul_I<<< gridmult, threadsmult, shared_mem_size >>>
      (data, data, data, full_width, nw, beta, start,start,start+nw,start, start, start+nw,1);

  } else {
    /*A=floyd-warshall(A);
    B=AB;
    C=CA;
    D=D+CB;
    D=floyd-warshall(D);
    B=BD;
    C=DC;
    A=A+BC;*/
        
    int nw = width / 2;		// new width

    // setup execution parameters
    dim3 gemmgrid( nw /64, nw/16 );
    dim3 gemmthreads( 16, 4 );

    // Remember: Column-major
    float * A = data + start * full_width + start;
    float * B = data + (start+nw) * full_width + start;
    float * C = data + start * full_width + (start+nw);
    float * D = data + (start+nw) * full_width + (start+nw);

    // sgemmNN_MinPlus( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
    // no need to send m & n since they are known through grid dimensions !

    // Do FW for A
    GpuFloydWarshallBeta2(data, start, nw, full_width, beta);
    
    // execute the parallel multiplication kernel B = AB
    sgemmNN_MinPlus<<<gemmgrid, gemmthreads>>>(A, full_width, B, full_width, B, full_width, nw,  FLOATINF );

    // execute the parallel multiplication kernel C = CA
    sgemmNN_MinPlus<<<gemmgrid, gemmthreads>>>(C, full_width, A, full_width, C, full_width, nw,  FLOATINF );
        
     
    // execute the parallel multiplication kernel  D += CB 
    sgemmNN_MinPlus<<<gemmgrid, gemmthreads>>>(C, full_width, B, full_width, D, full_width, nw,  1 );

    // do FW for D
    GpuFloydWarshallBeta(data, start+nw, nw, full_width, beta);

    // execute the parallel multiplication kernel B = BD
    sgemmNN_MinPlus<<<gemmgrid, gemmthreads>>>(B, full_width, D, full_width, B, full_width, nw,  FLOATINF );

    // execute the parallel multiplication kernel C = DC
    sgemmNN_MinPlus<<<gemmgrid, gemmthreads>>>(D, full_width, C, full_width, C, full_width, nw,  FLOATINF );

    // execute the parallel multiplication kernel A += BC
    sgemmNN_MinPlus<<<gemmgrid, gemmthreads>>>(B, full_width, C, full_width, A, full_width, nw,  1 );
  }
}