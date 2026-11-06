#ifndef SRESAMPLE_TET4_TRANSFER_KERNEL_HPP
#define SRESAMPLE_TET4_TRANSFER_KERNEL_HPP

#include "sresample_atomic.hpp"
#include "sresample_tet4_intersect_kernel.hpp"
#include "sresample_tet4_transfer.hpp"

// One pair of the transfer (sresample_tet4_transfer.hpp), for the host loops and the device
// kernels alike.

namespace sresample {
    namespace intersect {

        template <typename TGeom, typename TIdx>
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void tet4_load(const MeshView<TGeom, TIdx> &mesh, const ptrdiff_t e, double x[4][3]) {
            for (int v = 0; v < 4; ++v) {
                const ptrdiff_t node = (ptrdiff_t)mesh.elements[v][e];
                for (int d = 0; d < 3; ++d) x[v][d] = (double)mesh.points[d][node];
            }
        }

        // B of pair k into the structure of arrays; returns its Tet4PairClass.
        template <typename TReal, typename TGeom, typename TIdx>
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE int tet4_pair_mixed_mass(const MeshView<TGeom, TIdx> &solid,
                                                                        const MeshView<TGeom, TIdx> &fluid,
                                                                        const ptrdiff_t              npairs,
                                                                        const ptrdiff_t              k,
                                                                        const int *const             pair_solid,
                                                                        const int *const             pair_fluid,
                                                                        TReal *const                 B) {
            double xs[4][3], xf[4][3], b[4][4];
            tet4_load(solid, pair_solid[k], xs);
            tet4_load(fluid, pair_fluid[k], xf);
            const int cls = tet4_mixed_mass_pair(xs, xf, b);
            for (int i = 0; i < 4; ++i) {
                for (int a = 0; a < 4; ++a) B[(4 * i + a) * npairs + k] = (TReal)b[i][a];
            }
            return cls;
        }

        template <typename TReal, typename TGeom, typename TIdx>
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void tet4_pair_apply(const MeshView<TGeom, TIdx> &solid,
                                                                    const MeshView<TGeom, TIdx> &fluid,
                                                                    const ptrdiff_t              npairs,
                                                                    const ptrdiff_t              k,
                                                                    const int *const             pair_solid,
                                                                    const int *const             pair_fluid,
                                                                    const TReal *const           B,
                                                                    const TReal *const           v0,
                                                                    const TReal *const           v1,
                                                                    const TReal *const           v2,
                                                                    TReal *const                 W,
                                                                    TReal *const                 U0,
                                                                    TReal *const                 U1,
                                                                    TReal *const                 U2) {
            const ptrdiff_t s = pair_solid[k], f = pair_fluid[k];
            TReal           vs[4][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
            if (U0) {
                for (int a = 0; a < 4; ++a) {
                    const ptrdiff_t node = (ptrdiff_t)solid.elements[a][s];
                    vs[a][0]             = v0[node];
                    vs[a][1]             = v1[node];
                    vs[a][2]             = v2[node];
                }
            }
            for (int i = 0; i < 4; ++i) {
                TReal w = 0, u[3] = {0, 0, 0};
                for (int a = 0; a < 4; ++a) {
                    const TReal bia = B[(4 * i + a) * npairs + k];
                    w += bia;
                    if (U0) {
                        for (int d = 0; d < 3; ++d) u[d] += bia * vs[a][d];
                    }
                }
                if (w == 0) continue;  // a separated pair
                const ptrdiff_t node = (ptrdiff_t)fluid.elements[i][f];
                atomic_add(&W[node], w);
                if (U0) {
                    atomic_add(&U0[node], u[0]);
                    atomic_add(&U1[node], u[1]);
                    atomic_add(&U2[node], u[2]);
                }
            }
        }

        template <typename TReal, typename TGeom, typename TIdx>
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void tet4_pair_apply_transpose(const MeshView<TGeom, TIdx> &solid,
                                                                              const MeshView<TGeom, TIdx> &fluid,
                                                                              const ptrdiff_t              npairs,
                                                                              const ptrdiff_t              k,
                                                                              const int *const             pair_solid,
                                                                              const int *const             pair_fluid,
                                                                              const TReal *const           B,
                                                                              const TReal *const           u0,
                                                                              const TReal *const           u1,
                                                                              const TReal *const           u2,
                                                                              TReal *const                 c,
                                                                              TReal *const                 g0,
                                                                              TReal *const                 g1,
                                                                              TReal *const                 g2) {
            const ptrdiff_t s = pair_solid[k], f = pair_fluid[k];
            TReal           uf[4][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
            if (g0) {
                for (int i = 0; i < 4; ++i) {
                    const ptrdiff_t node = (ptrdiff_t)fluid.elements[i][f];
                    uf[i][0]             = u0[node];
                    uf[i][1]             = u1[node];
                    uf[i][2]             = u2[node];
                }
            }
            for (int a = 0; a < 4; ++a) {
                TReal w = 0, g[3] = {0, 0, 0};
                for (int i = 0; i < 4; ++i) {
                    const TReal bia = B[(4 * i + a) * npairs + k];
                    w += bia;
                    if (g0) {
                        for (int d = 0; d < 3; ++d) g[d] += bia * uf[i][d];
                    }
                }
                if (w == 0) continue;
                const ptrdiff_t node = (ptrdiff_t)solid.elements[a][s];
                if (c) atomic_add(&c[node], w);
                if (g0) {
                    atomic_add(&g0[node], g[0]);
                    atomic_add(&g1[node], g[1]);
                    atomic_add(&g2[node], g[2]);
                }
            }
        }

    }  // namespace intersect
}  // namespace sresample

#endif  // SRESAMPLE_TET4_TRANSFER_KERNEL_HPP
