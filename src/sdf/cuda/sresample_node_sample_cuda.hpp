#ifndef SRESAMPLE_NODE_SAMPLE_CUDA_HPP
#define SRESAMPLE_NODE_SAMPLE_CUDA_HPP

#include "sresample_node_sample.hpp"

// The device side of sresample_node_sample.hpp: idx, disp, the points, the grid's coefficients and
// the outputs in device memory, the launch on `stream` (a cudaStream_t value).

namespace sresample {
    namespace sdf {
        namespace cuda {

            template <typename TReal, typename TGeom, typename TIdx, typename TGrid>
            int sample_nodes(const int                        order,
                             const ptrdiff_t                  n,
                             const TIdx *const                idx,
                             const NodePoints<TGeom>         &points,
                             const TReal *const               disp,
                             const BSplineGrid<TGeom, TGrid> &grid,
                             const NodeOutput<TReal>         &out,
                             void *const                      stream);

        }  // namespace cuda
    }  // namespace sdf
}  // namespace sresample

#endif  // SRESAMPLE_NODE_SAMPLE_CUDA_HPP
