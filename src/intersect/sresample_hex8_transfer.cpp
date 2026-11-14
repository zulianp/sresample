#include "sresample_hex8_transfer.hpp"

#include "sresample_hex8_transfer_kernel.hpp"
#include "sresample_instantiations.hpp"

#ifdef SRESAMPLE_ENABLE_CUDA
#include "sresample_hex8_transfer_cuda.hpp"
#endif

namespace sresample {
    namespace intersect {

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
                            const Exec                  &exec) {
            if (exec.space == MemorySpace::DEVICE) {
#ifdef SRESAMPLE_ENABLE_CUDA
                return cuda::hex8_mixed_mass<TReal, TGeom, TIdx>(solid, fluid, npairs, pair_solid, pair_fluid, B, options, counts,
                                                                 max_it, exec.stream);
#else
                SRESAMPLE_LOG(0, "device execution requested, but sresample was built without CUDA\n");
                return SRESAMPLE_FAILURE;
#endif
            }
            const Hex8Basis basis = options.basis == 1 ? Hex8Basis::SPLIT : Hex8Basis::TRILINEAR;
            ptrdiff_t       n[5]  = {0, 0, 0, 0, 0};
            int             most  = 0;
#pragma omp parallel
            {
                int local[5]   = {0, 0, 0, 0, 0};
                int local_most = 0;
                ptrdiff_t sum[5] = {0, 0, 0, 0, 0};
#pragma omp for
                for (ptrdiff_t k = 0; k < npairs; ++k) {
                    hex8_pair_mixed_mass(solid, fluid, npairs, k, pair_solid, pair_fluid, B, basis, local, &local_most);
                    // Flushed per pair: an int per thread cannot hold the counts of a large mesh.
                    for (int c = 0; c < 5; ++c) {
                        sum[c] += local[c];
                        local[c] = 0;
                    }
                }
#pragma omp critical
                {
                    for (int c = 0; c < 5; ++c) n[c] += sum[c];
                    most = local_most > most ? local_most : most;
                }
            }
            if (counts) {
                counts->separated    = n[TET4_PAIR_SEPARATED];
                counts->fluid_inside = n[TET4_PAIR_FLUID_INSIDE];
                counts->solid_inside = n[TET4_PAIR_SOLID_INSIDE];
                counts->clipped      = n[TET4_PAIR_CLIPPED];
                counts->overflow     = n[TET4_PAIR_OVERFLOW];
            }
            if (max_it) *max_it = most;
            return SRESAMPLE_SUCCESS;
        }

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
                                  const Exec                  &exec) {
            if (exec.space == MemorySpace::DEVICE) {
#ifdef SRESAMPLE_ENABLE_CUDA
                return cuda::hex8_mixed_mass_apply<TReal, TGeom, TIdx>(solid, fluid, npairs, pair_solid, pair_fluid, B, v, W, U, exec.stream);
#else
                return SRESAMPLE_FAILURE;
#endif
            }
            const bool with_u = U && U[0];
#pragma omp parallel for
            for (ptrdiff_t k = 0; k < npairs; ++k) {
                hex8_pair_apply(solid, fluid, npairs, k, pair_solid, pair_fluid, B, with_u ? v[0] : nullptr,
                                with_u ? v[1] : nullptr, with_u ? v[2] : nullptr, W, with_u ? U[0] : nullptr,
                                with_u ? U[1] : nullptr, with_u ? U[2] : nullptr);
            }
            return SRESAMPLE_SUCCESS;
        }

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
                                            const Exec                  &exec) {
            if (exec.space == MemorySpace::DEVICE) {
#ifdef SRESAMPLE_ENABLE_CUDA
                return cuda::hex8_mixed_mass_apply_transpose<TReal, TGeom, TIdx>(solid, fluid, npairs, pair_solid, pair_fluid, B, u, c, g,
                                                                                 exec.stream);
#else
                return SRESAMPLE_FAILURE;
#endif
            }
            const bool with_g = g && g[0];
#pragma omp parallel for
            for (ptrdiff_t k = 0; k < npairs; ++k) {
                hex8_pair_apply_transpose(solid, fluid, npairs, k, pair_solid, pair_fluid, B, with_g ? u[0] : nullptr,
                                          with_g ? u[1] : nullptr, with_g ? u[2] : nullptr, c, with_g ? g[0] : nullptr,
                                          with_g ? g[1] : nullptr, with_g ? g[2] : nullptr);
            }
            return SRESAMPLE_SUCCESS;
        }

        template <typename TReal, typename TGeom, typename TIdx>
        int hex8_split_volumes(const MeshView<TGeom, TIdx> &solid, TReal *const vol, const Exec &exec) {
            if (exec.space == MemorySpace::DEVICE) {
#ifdef SRESAMPLE_ENABLE_CUDA
                return cuda::hex8_split_volumes<TReal, TGeom, TIdx>(solid, vol, exec.stream);
#else
                return SRESAMPLE_FAILURE;
#endif
            }
#pragma omp parallel for
            for (ptrdiff_t e = 0; e < solid.nelements; ++e) {
                double    xs[8][3];
                long long ids[8];
                hex8_load(solid, e, xs, ids);
                vol[e] = (TReal)hex8_split_volume(xs, ids);
            }
            return SRESAMPLE_SUCCESS;
        }

#define SRESAMPLE_INSTANTIATE_HEX8_TRANSFER(TReal, TGeom, TIdx)                                                          \
    template int hex8_mixed_mass<TReal, TGeom, TIdx>(const MeshView<TGeom, TIdx> &, const MeshView<TGeom, TIdx> &,       \
                                                     const ptrdiff_t, const int *const, const int *const, TReal *const,  \
                                                     const Hex8TransferOptions &, Tet4PairCounts *const, int *const,     \
                                                     const Exec &);                                                      \
    template int hex8_mixed_mass_apply<TReal, TGeom, TIdx>(const MeshView<TGeom, TIdx> &, const MeshView<TGeom, TIdx> &, \
                                                           const ptrdiff_t, const int *const, const int *const,          \
                                                           const TReal *const, const TReal *const[3], TReal *const,      \
                                                           TReal *const[3], const Exec &);                               \
    template int hex8_mixed_mass_apply_transpose<TReal, TGeom, TIdx>(                                                    \
            const MeshView<TGeom, TIdx> &, const MeshView<TGeom, TIdx> &, const ptrdiff_t, const int *const,             \
            const int *const, const TReal *const, const TReal *const[3], TReal *const, TReal *const[3], const Exec &);   \
    template int hex8_split_volumes<TReal, TGeom, TIdx>(const MeshView<TGeom, TIdx> &, TReal *const, const Exec &);

        SRESAMPLE_FOREACH_MESH_TYPES(SRESAMPLE_INSTANTIATE_HEX8_TRANSFER)

    }  // namespace intersect
}  // namespace sresample
