#include "sresample_tet4_transfer_cuda.hpp"

#include "sresample_cuda_base.cuh"
#include "sresample_instantiations.hpp"
#include "sresample_tet4_transfer_kernel.hpp"

// One thread per pair, grid-stride. The pair classes are counted per block in shared memory and
// added to five global counters once per block.

namespace sresample {
    namespace intersect {
        namespace cuda {

            using sresample::cuda::block_size;
            using sresample::cuda::check_launch;
            using sresample::cuda::n_blocks;

            namespace {

                template <typename TReal, typename TGeom, typename TIdx>
                __global__ void mixed_mass_kernel(const MeshView<TGeom, TIdx> solid,
                                                  const MeshView<TGeom, TIdx> fluid,
                                                  const ptrdiff_t             npairs,
                                                  const int *const            pair_solid,
                                                  const int *const            pair_fluid,
                                                  TReal *const                B,
                                                  unsigned long long *const   counts) {
                    __shared__ unsigned int block_counts[5];
                    if (threadIdx.x < 5) block_counts[threadIdx.x] = 0;
                    __syncthreads();
                    for (ptrdiff_t k = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; k < npairs; k += (ptrdiff_t)blockDim.x * gridDim.x) {
                        const int cls = tet4_pair_mixed_mass(solid, fluid, npairs, k, pair_solid, pair_fluid, B);
                        atomicAdd(&block_counts[cls], 1u);
                    }
                    __syncthreads();
                    if (threadIdx.x < 5 && block_counts[threadIdx.x]) atomicAdd(&counts[threadIdx.x], (unsigned long long)block_counts[threadIdx.x]);
                }

                template <typename TReal, typename TGeom, typename TIdx>
                __global__ void apply_kernel(const MeshView<TGeom, TIdx> solid,
                                             const MeshView<TGeom, TIdx> fluid,
                                             const ptrdiff_t             npairs,
                                             const int *const            pair_solid,
                                             const int *const            pair_fluid,
                                             const TReal *const          B,
                                             const TReal *const          v0,
                                             const TReal *const          v1,
                                             const TReal *const          v2,
                                             TReal *const                W,
                                             TReal *const                U0,
                                             TReal *const                U1,
                                             TReal *const                U2) {
                    for (ptrdiff_t k = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; k < npairs; k += (ptrdiff_t)blockDim.x * gridDim.x) {
                        tet4_pair_apply(solid, fluid, npairs, k, pair_solid, pair_fluid, B, v0, v1, v2, W, U0, U1, U2);
                    }
                }

                template <typename TReal, typename TGeom, typename TIdx>
                __global__ void apply_transpose_kernel(const MeshView<TGeom, TIdx> solid,
                                                       const MeshView<TGeom, TIdx> fluid,
                                                       const ptrdiff_t             npairs,
                                                       const int *const            pair_solid,
                                                       const int *const            pair_fluid,
                                                       const TReal *const          B,
                                                       const TReal *const          u0,
                                                       const TReal *const          u1,
                                                       const TReal *const          u2,
                                                       TReal *const                c,
                                                       TReal *const                g0,
                                                       TReal *const                g1,
                                                       TReal *const                g2) {
                    for (ptrdiff_t k = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; k < npairs; k += (ptrdiff_t)blockDim.x * gridDim.x) {
                        tet4_pair_apply_transpose(solid, fluid, npairs, k, pair_solid, pair_fluid, B, u0, u1, u2, c, g0, g1, g2);
                    }
                }

            }  // namespace

            template <typename TReal, typename TGeom, typename TIdx>
            int tet4_mixed_mass(const MeshView<TGeom, TIdx> &solid,
                                const MeshView<TGeom, TIdx> &fluid,
                                const ptrdiff_t              npairs,
                                const int *const             pair_solid,
                                const int *const             pair_fluid,
                                TReal *const                 B,
                                Tet4PairCounts *const        counts,
                                void *const                  stream) {
                const cudaStream_t  s        = (cudaStream_t)stream;
                unsigned long long *d_counts = nullptr;
                if (cudaMalloc(&d_counts, 5 * sizeof(unsigned long long)) != cudaSuccess) return SRESAMPLE_FAILURE;
                cudaMemsetAsync(d_counts, 0, 5 * sizeof(unsigned long long), s);
                if (npairs) {
                    mixed_mass_kernel<TReal, TGeom, TIdx><<<n_blocks(npairs), block_size, 0, s>>>(solid, fluid, npairs, pair_solid, pair_fluid, B, d_counts);
                }
                int err = check_launch();
                if (counts && !err) {
                    unsigned long long h[5];
                    cudaMemcpyAsync(h, d_counts, sizeof(h), cudaMemcpyDeviceToHost, s);
                    if (cudaStreamSynchronize(s) != cudaSuccess) err = SRESAMPLE_FAILURE;
                    counts->separated    = (ptrdiff_t)h[TET4_PAIR_SEPARATED];
                    counts->fluid_inside = (ptrdiff_t)h[TET4_PAIR_FLUID_INSIDE];
                    counts->solid_inside = (ptrdiff_t)h[TET4_PAIR_SOLID_INSIDE];
                    counts->clipped      = (ptrdiff_t)h[TET4_PAIR_CLIPPED];
                    counts->overflow     = (ptrdiff_t)h[TET4_PAIR_OVERFLOW];
                }
                cudaFree(d_counts);
                return err;
            }

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
                                      void *const                  stream) {
                if (!npairs) return SRESAMPLE_SUCCESS;
                const bool with_u = U && U[0];
                apply_kernel<TReal, TGeom, TIdx><<<n_blocks(npairs), block_size, 0, (cudaStream_t)stream>>>(
                        solid, fluid, npairs, pair_solid, pair_fluid, B, with_u ? v[0] : nullptr, with_u ? v[1] : nullptr,
                        with_u ? v[2] : nullptr, W, with_u ? U[0] : nullptr, with_u ? U[1] : nullptr, with_u ? U[2] : nullptr);
                return check_launch();
            }

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
                                                void *const                  stream) {
                if (!npairs) return SRESAMPLE_SUCCESS;
                const bool with_g = g && g[0];
                apply_transpose_kernel<TReal, TGeom, TIdx><<<n_blocks(npairs), block_size, 0, (cudaStream_t)stream>>>(
                        solid, fluid, npairs, pair_solid, pair_fluid, B, with_g ? u[0] : nullptr, with_g ? u[1] : nullptr,
                        with_g ? u[2] : nullptr, c, with_g ? g[0] : nullptr, with_g ? g[1] : nullptr, with_g ? g[2] : nullptr);
                return check_launch();
            }

#define SRESAMPLE_INSTANTIATE_TET4_TRANSFER_CUDA(TReal, TGeom, TIdx)                                                    \
    template int tet4_mixed_mass<TReal, TGeom, TIdx>(const MeshView<TGeom, TIdx> &, const MeshView<TGeom, TIdx> &,      \
                                                     const ptrdiff_t, const int *const, const int *const, TReal *const, \
                                                     Tet4PairCounts *const, void *const);                               \
    template int tet4_mixed_mass_apply<TReal, TGeom, TIdx>(const MeshView<TGeom, TIdx> &, const MeshView<TGeom, TIdx> &, \
                                                           const ptrdiff_t, const int *const, const int *const,          \
                                                           const TReal *const, const TReal *const[3], TReal *const,      \
                                                           TReal *const[3], void *const);                                \
    template int tet4_mixed_mass_apply_transpose<TReal, TGeom, TIdx>(                                                    \
            const MeshView<TGeom, TIdx> &, const MeshView<TGeom, TIdx> &, const ptrdiff_t, const int *const,             \
            const int *const, const TReal *const, const TReal *const[3], TReal *const, TReal *const[3], void *const);

            SRESAMPLE_FOREACH_MESH_TYPES(SRESAMPLE_INSTANTIATE_TET4_TRANSFER_CUDA)

        }  // namespace cuda
    }  // namespace intersect
}  // namespace sresample
