#include "sresample_node_sample_cuda.hpp"

#include "sresample_cuda_base.cuh"
#include "sresample_instantiations.hpp"
#include "sresample_node_sample_kernel.hpp"

namespace sresample {
    namespace sdf {
        namespace cuda {

            namespace {
                template <int ORDER, typename TReal, typename TGeom, typename TIdx, typename TGrid>
                __global__ void sample_nodes_kernel(const ptrdiff_t                       n,
                                                    const TIdx *const SRESAMPLE_RESTRICT  idx,
                                                    const NodePoints<TGeom>               points,
                                                    const TReal *const SRESAMPLE_RESTRICT disp,
                                                    const BSplineGrid<TGeom, TGrid>       grid,
                                                    const NodeOutput<TReal>               out) {
                    for (ptrdiff_t i = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; i < n; i += (ptrdiff_t)blockDim.x * gridDim.x) {
                        sample_node<ORDER>(i, idx, points, disp, grid, out);
                    }
                }
            }  // namespace

            template <typename TReal, typename TGeom, typename TIdx, typename TGrid>
            int sample_nodes(const int                        order,
                             const ptrdiff_t                  n,
                             const TIdx *const                idx,
                             const NodePoints<TGeom>         &points,
                             const TReal *const               disp,
                             const BSplineGrid<TGeom, TGrid> &grid,
                             const NodeOutput<TReal>         &out,
                             void *const                      stream) {
                if (!n) return SRESAMPLE_SUCCESS;
                const cudaStream_t s = (cudaStream_t)stream;
                switch (order) {
                    case 0:
                        sample_nodes_kernel<0><<<n_blocks(n), block_size, 0, s>>>(n, idx, points, disp, grid, out);
                        break;
                    case 1:
                        sample_nodes_kernel<1><<<n_blocks(n), block_size, 0, s>>>(n, idx, points, disp, grid, out);
                        break;
                    default:
                        sample_nodes_kernel<2><<<n_blocks(n), block_size, 0, s>>>(n, idx, points, disp, grid, out);
                        break;
                }
                return check_launch();
            }

#define SRESAMPLE_INSTANTIATE_NODE_SAMPLE_CUDA(TReal, TGeom, TIdx, TGrid)             \
    template int sample_nodes<TReal, TGeom, TIdx, TGrid>(const int,                   \
                                                         const ptrdiff_t,             \
                                                         const TIdx *const,           \
                                                         const NodePoints<TGeom> &,   \
                                                         const TReal *const,          \
                                                         const BSplineGrid<TGeom, TGrid> &, \
                                                         const NodeOutput<TReal> &,   \
                                                         void *const);

            SRESAMPLE_FOREACH_TYPES(SRESAMPLE_INSTANTIATE_NODE_SAMPLE_CUDA)

        }  // namespace cuda
    }  // namespace sdf
}  // namespace sresample
