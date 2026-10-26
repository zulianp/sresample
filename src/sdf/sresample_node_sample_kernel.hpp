#ifndef SRESAMPLE_NODE_SAMPLE_KERNEL_HPP
#define SRESAMPLE_NODE_SAMPLE_KERNEL_HPP

#include <math.h>

#include "sresample_bspline.hpp"
#include "sresample_node_sample.hpp"

// One node of the nodal sampling, for the host loop (node_interpolate.cpp) and the device kernel
// (cuda/node_interpolate_cuda.cu) alike: everything it reads or writes is passed by value.
// (Those are now sresample_node_sample.cpp and cuda/sresample_node_sample.cu.)

namespace sresample {
    namespace sdf {

        // The ORDER-th derivatives of the B-spline s at the points X[node] + disp[3 node + d], for
        // node = idx[i] (i when idx is null; disp may be null), summed in real_t. Writes entry i of
        // whichever outputs are given:
        //   gap        -s, with ORDER >= 1 -s / |grad s|
        //   normals    grad s / |grad s|                              (ORDER >= 1, 3 rows)
        //   curvature  (H - n m^T - m n^T) / |grad s|, m = H n        (ORDER >= 2, 6 rows)
        // Outside the grid the gap is SFEM_SDF_UNREACHABLE and the normal and curvature are zero; so
        // are they where the gradient vanishes.
        // (real_t is TReal, SFEM_SDF_UNREACHABLE sdf::unreachable.)
        template <int ORDER, typename TReal, typename TGeom, typename TIdx, typename TGrid>
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void sample_node(const ptrdiff_t                     i,
                                                                const TIdx *const SRESAMPLE_RESTRICT  idx,
                                                                const NodePoints<TGeom>             &points,
                                                                const TReal *const SRESAMPLE_RESTRICT disp,
                                                                const BSplineGrid<TGeom, TGrid>     &grid,
                                                                const NodeOutput<TReal>             &out) {
            const ptrdiff_t node = idx ? (ptrdiff_t)idx[i] : i;
            TReal           p[3];
            for (int d = 0; d < 3; ++d) {
                p[d] = (TReal)points.xyz[d][node] + (disp ? disp[node * 3 + d] : TReal(0));
            }

            TReal s, g[3] = {0, 0, 0}, H[6] = {0, 0, 0, 0, 0, 0};
            if (!sample<ORDER>(grid, p[0], p[1], p[2], &s, g, H)) {
                if (out.gap) out.gap[i] = unreachable;
                if (out.normals[0]) {
                    for (int d = 0; d < 3; ++d) out.normals[d][i] = 0;
                }
                if (out.curvature[0]) {
                    for (int c = 0; c < 6; ++c) out.curvature[c][i] = 0;
                }
                return;
            }

            const TReal q = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
            if (ORDER == 0 || q == 0) {
                if (out.gap) out.gap[i] = -s;
                if (out.normals[0]) {
                    for (int d = 0; d < 3; ++d) out.normals[d][i] = 0;
                }
                if (out.curvature[0]) {
                    for (int c = 0; c < 6; ++c) out.curvature[c][i] = 0;
                }
                return;
            }

            const TReal nn[3] = {g[0] / q, g[1] / q, g[2] / q};
            if (out.gap) out.gap[i] = -s / q;
            if (out.normals[0]) {
                for (int d = 0; d < 3; ++d) out.normals[d][i] = nn[d];
            }

            if (ORDER >= 2 && out.curvature[0]) {
                // m = H n = grad |grad s|; the Hessian of the gap -s / |grad s| at s = 0 is
                // -(H - n m^T - m n^T) / |grad s|.
                const TReal m[3] = {H[0] * nn[0] + H[1] * nn[1] + H[2] * nn[2],
                                    H[1] * nn[0] + H[3] * nn[1] + H[4] * nn[2],
                                    H[2] * nn[0] + H[4] * nn[1] + H[5] * nn[2]};
                int         c    = 0;
                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                        out.curvature[c][i] = (H[c] - nn[a] * m[b] - m[a] * nn[b]) / q;
                        c++;
                    }
                }
            }
        }

    }  // namespace sdf
}  // namespace sresample

#endif  // SRESAMPLE_NODE_SAMPLE_KERNEL_HPP
