#ifndef SRESAMPLE_TET4_TRANSFER_HPP
#define SRESAMPLE_TET4_TRANSFER_HPP

#include <stddef.h>

#include "sresample_base.hpp"
#include "sresample_views.hpp"

// The exact L2 coupling of two TET4 meshes that overlap (a "solid" moving through a "fluid"), from
// a list of candidate element pairs (search::cell_list_*): per pair the mixed mass matrix
//
//   B_ia = int_{T_s cap T_f} phi_i psi_a,     i a fluid tet's node, a a solid tet's node,
//
// (sresample_tet4_intersect_kernel.hpp), and its application both ways:
//
//   fluid side:  W_i += sum_a B_ia,  U_i += sum_a B_ia v_a       (v given at the solid nodes)
//   solid side:  c_a += sum_i B_ia,  g_a += sum_i B_ia u_i       (u given at the fluid nodes)
//
// sum_i W_i = sum_a c_a = |T_s cap T_f| summed over the pairs, exactly up to round-off.
//
// On the host or the device (exec). For device calls the MeshView's arrays of pointers (elements,
// points) must be device memory too, as everything else.

namespace sresample {
    namespace intersect {

        // How the pairs were integrated (sresample_tet4_intersect_kernel.hpp's Tet4PairClass).
        struct Tet4PairCounts {
            ptrdiff_t separated{0};
            ptrdiff_t fluid_inside{0};
            ptrdiff_t solid_inside{0};
            ptrdiff_t clipped{0};
            ptrdiff_t overflow{0};
        };

        // B for the pairs (pair_solid[k], pair_fluid[k]), k < npairs, structure of arrays:
        // B[(4 i + a) npairs + k]. counts (a host pointer) may be null.
        template <typename TReal, typename TGeom, typename TIdx>
        int tet4_mixed_mass(const MeshView<TGeom, TIdx> &solid,
                            const MeshView<TGeom, TIdx> &fluid,
                            const ptrdiff_t              npairs,
                            const int *const             pair_solid,
                            const int *const             pair_fluid,
                            TReal *const                 B,
                            Tet4PairCounts *const        counts,
                            const Exec                  &exec);

        // The fluid side: W[i] += sum_a B_ia and, if U[0] is given, U[d][i] += sum_a B_ia v[d][a].
        template <typename TReal, typename TGeom, typename TIdx>
        int tet4_mixed_mass_apply(const MeshView<TGeom, TIdx> &solid,
                                  const MeshView<TGeom, TIdx> &fluid,
                                  const ptrdiff_t              npairs,
                                  const int *const             pair_solid,
                                  const int *const             pair_fluid,
                                  const TReal *const           B,
                                  const TReal *const           v[3],
                                  TReal *const                 W,
                                  TReal *const                 U[3],
                                  const Exec                  &exec);

        // The solid side: c[a] += sum_i B_ia (if c is given) and, if g[0] is given,
        // g[d][a] += sum_i B_ia u[d][i].
        template <typename TReal, typename TGeom, typename TIdx>
        int tet4_mixed_mass_apply_transpose(const MeshView<TGeom, TIdx> &solid,
                                            const MeshView<TGeom, TIdx> &fluid,
                                            const ptrdiff_t              npairs,
                                            const int *const             pair_solid,
                                            const int *const             pair_fluid,
                                            const TReal *const           B,
                                            const TReal *const           u[3],
                                            TReal *const                 c,
                                            TReal *const                 g[3],
                                            const Exec                  &exec);

    }  // namespace intersect
}  // namespace sresample

#endif  // SRESAMPLE_TET4_TRANSFER_HPP
