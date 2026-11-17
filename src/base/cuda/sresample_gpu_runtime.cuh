#ifndef SRESAMPLE_GPU_RUNTIME_CUH
#define SRESAMPLE_GPU_RUNTIME_CUH

// The GPU runtime the device path is built against: CUDA, or HIP for AMD GPUs.
//
// sresample's device code is written against the CUDA runtime API and compiles unchanged under
// HIP, whose runtime mirrors it name for name. Under SRESAMPLE_ENABLE_HIP this header includes HIP
// and maps the cuda* names sresample calls onto their hip* counterparts; otherwise it includes the
// CUDA runtime. Every translation unit that calls the runtime includes it instead of
// <cuda_runtime.h>.
//
// smesh and SCCD have headers like this one, and a translation unit can include several. A macro
// may be redefined only with the same replacement, so a name mapped in more than one header must
// map to exactly the same tokens in each.

#include "sresample_config.hpp"

#if defined(SRESAMPLE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError

#define cudaMalloc hipMalloc
#define cudaFree hipFree

#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync

#define cudaStream_t hipStream_t
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize

#else

#include <cuda_runtime.h>

#endif

#endif  // SRESAMPLE_GPU_RUNTIME_CUH
