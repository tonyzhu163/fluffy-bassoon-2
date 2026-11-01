#include <cuda_runtime.h>
#include <cuda/std/__cccl/cuda_toolkit.h>
static_assert(CUDART_VERSION/1000 == __CUDACC_VER_MAJOR__);
static_assert((CUDART_VERSION%1000)/10 == __CUDACC_VER_MINOR__);
int toolkit_probe(){return CUDART_VERSION;}
