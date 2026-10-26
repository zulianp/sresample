#include "sresample_node_sample.hpp"

#include "sresample_instantiations.hpp"
#include "sresample_node_sample_kernel.hpp"

#ifdef SRESAMPLE_ENABLE_CUDA
#include "sresample_node_sample_cuda.hpp"
#endif

// Nodal sampling of an SDF grid's interpolating cubic B-spline: `data` and `stride` are its
// coefficients (sfem_sdf_sampling.hpp). The gap is -sdf / |grad sdf| (-sdf for the value alone),
// the normal the field's unit gradient.
namespace sresample {
    namespace sdf {

        namespace {
            // sample_node (node_interpolate_kernel.hpp) at every node.
            template <int ORDER, typename TReal, typename TGeom, typename TIdx, typename TGrid>
            void sample_nodes_host(const ptrdiff_t                  n,
                                   const TIdx *const                idx,
                                   const NodePoints<TGeom>         &points,
                                   const TReal *const               disp,
                                   const BSplineGrid<TGeom, TGrid> &grid,
                                   const NodeOutput<TReal>         &out) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; ++i) {
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
                         const Exec                      &exec) {
            if (order < 0 || order > 2) {
                SRESAMPLE_LOG(0, "sample_nodes: order %d is not 0, 1 or 2\n", order);
                return SRESAMPLE_FAILURE;
            }
            if (!n) return SRESAMPLE_SUCCESS;

            if (exec.space == MemorySpace::DEVICE) {
#ifdef SRESAMPLE_ENABLE_CUDA
                return cuda::sample_nodes(order, n, idx, points, disp, grid, out, exec.stream);
#else
                SRESAMPLE_LOG(0, "device execution requested, but sresample was built without CUDA\n");
                return SRESAMPLE_FAILURE;
#endif
            }

            switch (order) {
                case 0:
                    sample_nodes_host<0>(n, idx, points, disp, grid, out);
                    break;
                case 1:
                    sample_nodes_host<1>(n, idx, points, disp, grid, out);
                    break;
                default:
                    sample_nodes_host<2>(n, idx, points, disp, grid, out);
                    break;
            }
            return SRESAMPLE_SUCCESS;
        }

#define SRESAMPLE_INSTANTIATE_NODE_SAMPLE(TReal, TGeom, TIdx, TGrid)                 \
    template int sample_nodes<TReal, TGeom, TIdx, TGrid>(const int,                  \
                                                         const ptrdiff_t,            \
                                                         const TIdx *const,          \
                                                         const NodePoints<TGeom> &,  \
                                                         const TReal *const,         \
                                                         const BSplineGrid<TGeom, TGrid> &, \
                                                         const NodeOutput<TReal> &,  \
                                                         const Exec &);

        SRESAMPLE_FOREACH_TYPES(SRESAMPLE_INSTANTIATE_NODE_SAMPLE)

    }  // namespace sdf
}  // namespace sresample
