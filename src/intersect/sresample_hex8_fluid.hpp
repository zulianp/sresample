#ifndef SRESAMPLE_HEX8_FLUID_HPP
#define SRESAMPLE_HEX8_FLUID_HPP

#include <stddef.h>

#include "sresample_base.hpp"
#include "sresample_hex8_transfer.hpp"
#include "sresample_tet4_transfer.hpp"
#include "sresample_views.hpp"

// The exact L2 coupling of a TET4 or HEX8 "solid" with a HEX8 "fluid" that overlap, as
// sresample_tet4_transfer.hpp and sresample_hex8_transfer.hpp for TET4 fluids: from candidate
// pairs (search::cell_list_*), per pair the mixed mass matrix
//
//   B_ia = int_{S cap H_f} phi_i psi_a,     i < 8 a fluid hex's node, a < ns a solid element's,
//
// (sresample_hex8_fluid_kernel.hpp: each hex the region of its conforming split into tets; phi_i
// and psi_a each hex's trilinear basis, or with Hex8Basis::SPLIT the P1 basis of its split tets;
// a tet's P1 basis) and its application both ways:
//
//   fluid side:  W_i += sum_a B_ia,  U_i += sum_a B_ia v_a       (v given at the solid nodes)
//   solid side:  c_a += sum_i B_ia,  g_a += sum_i B_ia u_i       (u given at the fluid nodes)
//
// sum_i W_i = sum_a c_a = |split(S) cap split(H_f)| summed over the pairs, up to round-off. The
// splits follow the meshes' MeshView::ids where given, else their element ids (local nodes): a
// fluid distributed over ranks should give global ids, so that its ranks split shared faces alike.
// options.basis is the solid's (if a HEX8), options.fluid_basis the fluid's.
//
// On the host or the device (exec), the MeshView's arrays of pointers in the call's memory space.

namespace sresample {
    namespace intersect {

        // B for the pairs (pair_solid[k], pair_fluid[k]), k < npairs, structure of arrays:
        // B[(ns i + a) npairs + k], ns = nxe(solid.type). counts (a host pointer, may be null): the
        // classes of the pairs' (solid tet, fluid tet) pairs. max_it (a host pointer, may be null):
        // Newton's most iterations (hex8_newton_max_it: some point did not converge).
        template <typename TReal, typename TGeom, typename TIdx>
        int hex8_fluid_mixed_mass(const MeshView<TGeom, TIdx> &solid,
                                  const MeshView<TGeom, TIdx> &fluid,
                                  const ptrdiff_t              npairs,
                                  const int *const             pair_solid,
                                  const int *const             pair_fluid,
                                  TReal *const                 B,
                                  const Hex8TransferOptions   &options,
                                  Tet4PairCounts *const        counts,
                                  int *const                   max_it,
                                  const Exec                  &exec);

        // The fluid side: W[i] += sum_a B_ia and, if U[0] is given, U[d][i] += sum_a B_ia v[d][a].
        template <typename TReal, typename TGeom, typename TIdx>
        int hex8_fluid_mixed_mass_apply(const MeshView<TGeom, TIdx> &solid,
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
        int hex8_fluid_mixed_mass_apply_transpose(const MeshView<TGeom, TIdx> &solid,
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

#endif  // SRESAMPLE_HEX8_FLUID_HPP
