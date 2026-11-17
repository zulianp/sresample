#ifndef SRESAMPLE_ATOMIC_HPP
#define SRESAMPLE_ATOMIC_HPP

#include "sresample_base.hpp"

namespace sresample {

    // *p += v, atomically: an OpenMP atomic on the host, atomicAdd on the device.
    template <typename T>
    SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void atomic_add(T *const p, const T v) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        atomicAdd(p, v);
#else
#pragma omp atomic update
        *p += v;
#endif
    }

}  // namespace sresample

#endif  // SRESAMPLE_ATOMIC_HPP
