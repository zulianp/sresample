#ifndef SRESAMPLE_HEX8_TRANSFER_CUDA_HPP
#define SRESAMPLE_HEX8_TRANSFER_CUDA_HPP

#include "sresample_hex8_transfer.hpp"

// The HEX8-TET4 transfer on the device (sresample_hex8_transfer.cu), on `stream`.

namespace sresample {
    namespace intersect {
        namespace cuda {

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
                                void *const                  stream);

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
                                      void *const                  stream);

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
                                                void *const                  stream);

            template <typename TReal, typename TGeom, typename TIdx>
            int hex8_split_volumes(const MeshView<TGeom, TIdx> &solid, TReal *const vol, void *const stream);

        }  // namespace cuda
    }  // namespace intersect
}  // namespace sresample

#endif  // SRESAMPLE_HEX8_TRANSFER_CUDA_HPP
