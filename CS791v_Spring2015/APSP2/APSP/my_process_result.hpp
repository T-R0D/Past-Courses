#ifndef _MY_CUDA_PROCESS_RESULT_
#define _MY_CUDA_PROCESS_RESULT_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template <typename Result_t>
class MyCudaProcessResult {
 public:
  MyCudaProcessResult() {
   success = false;
   time_to_complete = -1.0;
  };

  Result_t GetResult() const {return result;};
  bool Success() const {return success;};
  double GetTimeToComplete() const {return time_to_complete;};

  void SetResult(const Result_t& new_result) {result = new_result;};
  cudaError& CudaStatus() {return cudaStatus;}; // hmmmmmmm
  void SetSuccess(const bool new_success) {success = new_success;};
  void SetTimeToComplete(const double time) {time_to_complete = time;};

 private:
  Result_t result;
  cudaError cudaStatus;
  bool success;
  double time_to_complete;
}

#endif //define _MY_CUDA_PROCESS_RESULT_