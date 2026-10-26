#ifndef SRESAMPLE_SURFACE_SAMPLE_CUDA_HPP
#define SRESAMPLE_SURFACE_SAMPLE_CUDA_HPP

#include "sresample_surface_sample.hpp"

// The device side of sresample_surface_sample.hpp: every array in device memory, every launch on
// `stream` (a cudaStream_t value), nothing allocated.

namespace sresample {
    namespace sdf {
        namespace cuda {

            template <typename TReal, typename TGeom, typename TIdx, typename TGrid>
            int surface_sample_local(const MeshView<TGeom, TIdx>     &mesh,
                                     const BSplineGrid<TGeom, TGrid> &grid,
                                     const SurfaceOutput<TReal>      &out,
                                     void *const                      stream);

            template <typename TReal, typename TGeom, typename TIdx>
            int surface_weights_local(const MeshView<TGeom, TIdx> &mesh, TReal *const w, void *const stream);

            template <typename TReal>
            int finish_surface_sample(const ptrdiff_t n, const TReal *const w, const SurfaceOutput<TReal> &out, void *const stream);

            // x[0 .. n) = 0 when x is not null.
            template <typename TReal>
            int zero(const ptrdiff_t n, TReal *const x, void *const stream);

        }  // namespace cuda
    }  // namespace sdf
}  // namespace sresample

#endif  // SRESAMPLE_SURFACE_SAMPLE_CUDA_HPP
