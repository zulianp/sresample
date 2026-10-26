#ifndef SRESAMPLE_BSPLINE_HPP
#define SRESAMPLE_BSPLINE_HPP

#include <math.h>
#include <stddef.h>

#include "sresample_base.hpp"
#include "sresample_views.hpp"

// Sampling of a signed distance field on a regular grid by its interpolating cubic B-spline.
//
// The interpolant is twice continuously differentiable, so the field's gradient (a contact
// normal) and Hessian (a contact curvature) do not jump between cells, as a trilinear
// interpolant's gradient does. It equals the samples at the grid nodes, reproduces linear
// fields exactly and converges as h^4 on smooth ones.
//
// The grid is given as its B-spline coefficients (see sdf_bspline_coefficients), padded by one
// layer on every side: `n` is the number of samples per axis, `origin` and `delta` place
// sample 0 and the spacing, `stride` is the padded array's, and `coeffs` points at the
// coefficient of sample (0, 0, 0), so indices -1 and n are valid. A point is inside the grid
// while its cell is, i.e. 0 <= floor((p - origin) / delta) <= n - 2 on every axis, the rule
// the trilinear sampling used.

namespace sresample {
    namespace sdf {

        // The gap every sampler reports where a point lies outside the grid: large, so the contact
        // constraint there stays inactive.
        static constexpr int unreachable = 10000;

        // The cell of p and its local coordinate in [0, 1) per axis; false outside the grid.
        template <typename TReal, typename TGeom, typename TGrid>
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE bool cell(const BSplineGrid<TGeom, TGrid> &grid,
                                                         const TReal                      x,
                                                         const TReal                      y,
                                                         const TReal                      z,
                                                         ptrdiff_t *const SRESAMPLE_RESTRICT c,
                                                         TReal *const SRESAMPLE_RESTRICT     t) {
            const TReal p[3] = {x, y, z};
            for (int d = 0; d < 3; ++d) {
                const TReal     g = (p[d] - (TReal)grid.origin[d]) / (TReal)grid.delta[d];
                const ptrdiff_t i = (ptrdiff_t)floor(g);
                if (i < 0 || i + 1 >= grid.n[d]) {
                    return false;
                }
                c[d] = i;
                t[d] = g - (TReal)i;
            }
            return true;
        }

        // Cubic B-spline weights of the four nodes i - 1 .. i + 2 at local coordinate t, and their
        // first and second derivatives in t.
        template <typename TReal>
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void bspline_weights(const TReal                    t,
                                                                    TReal *const SRESAMPLE_RESTRICT w,
                                                                    TReal *const SRESAMPLE_RESTRICT dw,
                                                                    TReal *const SRESAMPLE_RESTRICT ddw) {
            const TReal t2 = t * t;
            const TReal t3 = t2 * t;
            const TReal s  = 1 - t;

            w[0] = s * s * s / 6;
            w[1] = (3 * t3 - 6 * t2 + 4) / 6;
            w[2] = (-3 * t3 + 3 * t2 + 3 * t + 1) / 6;
            w[3] = t3 / 6;

            dw[0] = -s * s / 2;
            dw[1] = TReal(1.5) * t2 - 2 * t;
            dw[2] = TReal(-1.5) * t2 + t + TReal(0.5);
            dw[3] = t2 / 2;

            ddw[0] = s;
            ddw[1] = 3 * t - 2;
            ddw[2] = 1 - 3 * t;
            ddw[3] = t;
        }

        // The field at p, and with ORDER >= 1 its gradient, with ORDER >= 2 its Hessian packed
        // [xx, xy, xz, yy, yz, zz], both in physical coordinates. False, and nothing written, outside
        // the grid.
        template <int ORDER, typename TReal, typename TGeom, typename TGrid>
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE bool sample(const BSplineGrid<TGeom, TGrid> &grid,
                                                           const TReal                      x,
                                                           const TReal                      y,
                                                           const TReal                      z,
                                                           TReal *const SRESAMPLE_RESTRICT  value,
                                                           TReal *const SRESAMPLE_RESTRICT  gradient,
                                                           TReal *const SRESAMPLE_RESTRICT  hessian) {
            ptrdiff_t c[3];
            TReal     t[3];
            if (!cell(grid, x, y, z, c, t)) {
                return false;
            }

            TReal w[3][4], dw[3][4], ddw[3][4];
            for (int d = 0; d < 3; ++d) {
                bspline_weights(t[d], w[d], dw[d], ddw[d]);
            }

            const ptrdiff_t *const stride = grid.stride;
            const TGrid *const     base   = &grid.coeffs[(c[0] - 1) * stride[0] + (c[1] - 1) * stride[1] + (c[2] - 1) * stride[2]];

            TReal f    = 0;
            TReal g[3] = {0, 0, 0};
            TReal H[6] = {0, 0, 0, 0, 0, 0};
            for (int k = 0; k < 4; ++k) {
                for (int b = 0; b < 4; ++b) {
                    for (int a = 0; a < 4; ++a) {
                        const TReal v = (TReal)base[a * stride[0] + b * stride[1] + k * stride[2]];
                        f += w[0][a] * w[1][b] * w[2][k] * v;
                        if (ORDER >= 1) {
                            g[0] += dw[0][a] * w[1][b] * w[2][k] * v;
                            g[1] += w[0][a] * dw[1][b] * w[2][k] * v;
                            g[2] += w[0][a] * w[1][b] * dw[2][k] * v;
                        }
                        if (ORDER >= 2) {
                            H[0] += ddw[0][a] * w[1][b] * w[2][k] * v;
                            H[1] += dw[0][a] * dw[1][b] * w[2][k] * v;
                            H[2] += dw[0][a] * w[1][b] * dw[2][k] * v;
                            H[3] += w[0][a] * ddw[1][b] * w[2][k] * v;
                            H[4] += w[0][a] * dw[1][b] * dw[2][k] * v;
                            H[5] += w[0][a] * w[1][b] * ddw[2][k] * v;
                        }
                    }
                }
            }

            *value = f;
            if (ORDER >= 1) {
                for (int d = 0; d < 3; ++d) {
                    gradient[d] = g[d] / (TReal)grid.delta[d];
                }
            }
            if (ORDER >= 2) {
                const TReal dx = grid.delta[0], dy = grid.delta[1], dz = grid.delta[2];
                hessian[0]     = H[0] / (dx * dx);
                hessian[1]     = H[1] / (dx * dy);
                hessian[2]     = H[2] / (dx * dz);
                hessian[3]     = H[3] / (dy * dy);
                hessian[4]     = H[4] / (dy * dz);
                hessian[5]     = H[5] / (dz * dz);
            }
            return true;
        }

        // The interpolating cubic B-spline coefficients of the samples of a grid with `n` points per
        // axis (`samples` with strides `sample_stride`), written to a coefficient array with strides
        // `coeff_stride` whose (0, 0, 0) entry is `coeffs`: the n nodes per axis and the pad nodes -1
        // and n. Beyond its ends each grid line is extended by point reflection through its end
        // sample, so linear fields are reproduced exactly up to the boundary. Separable recursive
        // filtering (Unser, Aldroubi and Eden 1993), in double, once per grid line.
        // On the host; fails with fewer than 2 nodes on an axis.
        template <typename TGrid>
        int bspline_coefficients(const ptrdiff_t *const SRESAMPLE_RESTRICT n,
                                 const ptrdiff_t *const SRESAMPLE_RESTRICT sample_stride,
                                 const TGrid *const SRESAMPLE_RESTRICT     samples,
                                 const ptrdiff_t *const SRESAMPLE_RESTRICT coeff_stride,
                                 TGrid *const SRESAMPLE_RESTRICT           coeffs);

        // The padded coefficient array of a grid with n nodes per axis: its strides (x fastest), its
        // size in entries, and the offset of node (0, 0, 0) in it.
        SRESAMPLE_INLINE void bspline_layout(const ptrdiff_t *const n,
                                             ptrdiff_t *const       stride,
                                             ptrdiff_t *const       size,
                                             ptrdiff_t *const       offset) {
            stride[0] = 1;
            stride[1] = n[0] + 2;
            stride[2] = (n[0] + 2) * (n[1] + 2);
            *size     = (n[0] + 2) * (n[1] + 2) * (n[2] + 2);
            *offset   = stride[0] + stride[1] + stride[2];
        }

    }  // namespace sdf
}  // namespace sresample

#endif  // SRESAMPLE_BSPLINE_HPP
