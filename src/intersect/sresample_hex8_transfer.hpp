#ifndef SRESAMPLE_HEX8_TRANSFER_HPP
#define SRESAMPLE_HEX8_TRANSFER_HPP

#include <stddef.h>

#include "sresample_base.hpp"
#include "sresample_tet4_transfer.hpp"
#include "sresample_views.hpp"

// The exact L2 coupling of a HEX8 "solid" with a TET4 "fluid" that overlap, as
// sresample_tet4_transfer.hpp for TET4 solids: from candidate pairs (search::cell_list_*, the
// solid's boxes by search::element_boxes), per pair the mixed mass matrix
//
//   B_ia = int_{H_s cap T_f} phi_i psi_a,     i a fluid tet's node, a a solid hex's node,
//
// (sresample_hex8_intersect_kernel.hpp: the hex's region its conforming split into tets; psi_a the
// trilinear basis, or with Hex8Basis::SPLIT the P1 basis of the split tets) and its application
// both ways:
//
//   fluid side:  W_i += sum_a B_ia,  U_i += sum_a B_ia v_a       (v given at the solid nodes)
//   solid side:  c_a += sum_i B_ia,  g_a += sum_i B_ia u_i       (u given at the fluid nodes)
//
// sum_i W_i = sum_a c_a = |split(H_s) cap T_f| summed over the pairs, up to round-off, in both bases.
// The solid's element ids (its MeshView's elements) choose each hex's split.
//
// On the host or the device (exec), the MeshView's arrays of pointers in the call's memory space.

namespace sresample {
    namespace intersect {

        // Newton's iterations at most, inverting a hex's trilinear map at a quadrature point.
        static constexpr int hex8_newton_max_it = 12;

        struct Hex8TransferOptions {
            int basis{0};        // Hex8Basis: 0 TRILINEAR, 1 SPLIT (a HEX8 solid's)
            int fluid_basis{0};  // the same for a HEX8 fluid (sresample_hex8_fluid.hpp)
        };

        // B for the pairs (pair_solid[k], pair_fluid[k]), k < npairs, structure of arrays:
        // B[(8 i + a) npairs + k]. counts (a host pointer, may be null): the classes of the pairs'
        // split tets against their fluid tets. max_it (a host pointer, may be null): Newton's most
        // iterations (TRILINEAR; hex8_newton_max_it: some point did not converge).
        template <typename TReal, typename TGeom, typename TIdx>
        int hex8_mixed_mass(const MeshView<TGeom, TIdx> &solid,
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
        int hex8_mixed_mass_apply(const MeshView<TGeom, TIdx> &solid,
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
        int hex8_mixed_mass_apply_transpose(const MeshView<TGeom, TIdx> &solid,
                                            const MeshView<TGeom, TIdx> &fluid,
                                            const ptrdiff_t              npairs,
                                            const int *const             pair_solid,
                                            const int *const             pair_fluid,
                                            const TReal *const           B,
                                            const TReal *const           u[3],
                                            TReal *const                 c,
                                            TReal *const                 g[3],
                                            const Exec                  &exec);

        // vol[e] = the volume of hex e's split tets, e < solid.nelements (the region the transfer
        // integrates: sum_e vol[e] = sum W where the fluid covers the solid).
        template <typename TReal, typename TGeom, typename TIdx>
        int hex8_split_volumes(const MeshView<TGeom, TIdx> &solid, TReal *const vol, const Exec &exec);

    }  // namespace intersect
}  // namespace sresample

#endif  // SRESAMPLE_HEX8_TRANSFER_HPP
