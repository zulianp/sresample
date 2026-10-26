#ifndef SRESAMPLE_BASE_HPP
#define SRESAMPLE_BASE_HPP

#include "sresample_config.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>

// Qualifiers. A function marked SRESAMPLE_HOST_DEVICE is compiled for the host and, by nvcc, for
// the device: the per-element and per-node kernels are written once and run on both.
#if defined(__CUDACC__)
#define SRESAMPLE_HOST_DEVICE __host__ __device__
#else
#define SRESAMPLE_HOST_DEVICE
#endif

#define SRESAMPLE_INLINE inline
#define SRESAMPLE_RESTRICT __restrict__

#define SRESAMPLE_UNUSED(x) ((void)(x))

namespace sresample {

    enum ReturnCode { SRESAMPLE_SUCCESS = 0, SRESAMPLE_FAILURE = 1 };

    // Where the arrays given to a call live, and the stream device work is ordered on (a
    // cudaStream_t value; null is the default stream). Host calls ignore the stream.
    enum class MemorySpace { HOST, DEVICE };

    struct Exec {
        MemorySpace space{MemorySpace::HOST};
        void       *stream{nullptr};

        static Exec host() { return Exec{MemorySpace::HOST, nullptr}; }
        static Exec device(void *const stream = nullptr) { return Exec{MemorySpace::DEVICE, stream}; }
    };

}  // namespace sresample

// Errors the caller cannot recover from (a broken invariant): report and abort.
#define SRESAMPLE_ERROR(...)                                                      \
    do {                                                                          \
        fprintf(stderr, "[sresample] %s:%d: ", __FILE__, __LINE__);               \
        fprintf(stderr, __VA_ARGS__);                                             \
        fflush(stderr);                                                           \
        abort();                                                                  \
    } while (0)

// Messages, compiled out above SRESAMPLE_LOG_LEVEL (0 errors, 1 warnings, 2 info, 3 debug).
#define SRESAMPLE_LOG(level, ...)                                                 \
    do {                                                                          \
        if ((level) <= SRESAMPLE_LOG_LEVEL) {                                     \
            fprintf(stderr, "[sresample] ");                                      \
            fprintf(stderr, __VA_ARGS__);                                         \
        }                                                                         \
    } while (0)

#endif  // SRESAMPLE_BASE_HPP
