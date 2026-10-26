#ifndef SRESAMPLE_CUDA_BASE_CUH
#define SRESAMPLE_CUDA_BASE_CUH

#include <cuda_runtime.h>

#include "sresample_base.hpp"

namespace sresample {
    namespace cuda {

        // The launch shape of the grid-stride loops.
        static constexpr int block_size = 128;

        inline ptrdiff_t n_blocks(const ptrdiff_t n) {
            const ptrdiff_t b = (n + block_size - 1) / block_size;
            return b > 0 ? b : 1;
        }

        // A launch error (bad configuration, missing kernel image) as a return code. With
        // SRESAMPLE_DEBUG_SYNC it also waits for the device, so a kernel's fault is reported here.
        inline int check_launch() {
            cudaError_t err = cudaGetLastError();
#ifdef SRESAMPLE_DEBUG_SYNC
            if (err == cudaSuccess) err = cudaDeviceSynchronize();
#endif
            if (err != cudaSuccess) {
                SRESAMPLE_LOG(0, "CUDA: %s\n", cudaGetErrorString(err));
                return SRESAMPLE_FAILURE;
            }
            return SRESAMPLE_SUCCESS;
        }

    }  // namespace cuda

    namespace sdf {
        namespace cuda {
            using sresample::cuda::block_size;
            using sresample::cuda::check_launch;
            using sresample::cuda::n_blocks;
        }  // namespace cuda
    }  // namespace sdf

}  // namespace sresample

#endif  // SRESAMPLE_CUDA_BASE_CUH
