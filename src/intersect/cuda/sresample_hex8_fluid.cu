#include "sresample_hex8_fluid_cuda.hpp"

#include "sresample_cuda_base.cuh"
#include "sresample_hex8_fluid_kernel.hpp"
#include "sresample_instantiations.hpp"

// One thread per pair, grid-stride, as sresample_hex8_transfer.cu. The
// classes of the split tets are counted per block in shared memory, then once per block globally;
// Newton's iterations by an atomic maximum.

namespace sresample {
    namespace intersect {
        namespace cuda {

            using sresample::cuda::block_size;
            using sresample::cuda::check_launch;
            using sresample::cuda::n_blocks;

            namespace {

                template <typename TReal, typename TGeom, typename TIdx>
                __global__ void hex8_fluid_mixed_mass_kernel(const MeshView<TGeom, TIdx> solid,
                                                       const MeshView<TGeom, TIdx> fluid,
                                                       const ptrdiff_t             npairs,
                                                       const int *const            pair_solid,
                                                       const int *const            pair_fluid,
                                                       TReal *const                B,
                                                       const int                   basis,
                                                       const int                   fluid_basis,
                                                       unsigned long long *const   counts,
                                                       int *const                  max_it) {
                    __shared__ unsigned int block_counts[5];
                    __shared__ int          block_max;
                    if (threadIdx.x < 5) block_counts[threadIdx.x] = 0;
                    if (threadIdx.x == 0) block_max = 0;
                    __syncthreads();
                    const Hex8Basis b  = basis == 1 ? Hex8Basis::SPLIT : Hex8Basis::TRILINEAR;
                    const Hex8Basis fb = fluid_basis == 1 ? Hex8Basis::SPLIT : Hex8Basis::TRILINEAR;
                    int             most = 0;
                    for (ptrdiff_t k = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; k < npairs; k += (ptrdiff_t)blockDim.x * gridDim.x) {
                        int local[5] = {0, 0, 0, 0, 0};
                        hex8_fluid_pair_mixed_mass(solid, fluid, npairs, k, pair_solid, pair_fluid, B, b, fb, local, &most);
                        for (int c = 0; c < 5; ++c) {
                            if (local[c]) atomicAdd(&block_counts[c], (unsigned int)local[c]);
                        }
                    }
                    atomicMax(&block_max, most);
                    __syncthreads();
                    if (threadIdx.x < 5 && block_counts[threadIdx.x]) atomicAdd(&counts[threadIdx.x], (unsigned long long)block_counts[threadIdx.x]);
                    if (threadIdx.x == 0) atomicMax(max_it, block_max);
                }

                template <typename TReal, typename TGeom, typename TIdx>
                __global__ void hex8_fluid_apply_kernel(const MeshView<TGeom, TIdx> solid,
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
                        hex8_fluid_pair_apply(solid, fluid, npairs, k, pair_solid, pair_fluid, B, v0, v1, v2, W, U0, U1, U2);
                    }
                }

                template <typename TReal, typename TGeom, typename TIdx>
                __global__ void hex8_fluid_apply_transpose_kernel(const MeshView<TGeom, TIdx> solid,
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
                        hex8_fluid_pair_apply_transpose(solid, fluid, npairs, k, pair_solid, pair_fluid, B, u0, u1, u2, c, g0, g1, g2);
                    }
                }

            }  // namespace

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
                                void *const                  stream) {
                const cudaStream_t  s        = (cudaStream_t)stream;
                unsigned long long *d_counts = nullptr;
                if (cudaMalloc(&d_counts, 5 * sizeof(unsigned long long) + sizeof(int)) != cudaSuccess) return SRESAMPLE_FAILURE;
                int *d_max = (int *)(d_counts + 5);
                cudaMemsetAsync(d_counts, 0, 5 * sizeof(unsigned long long) + sizeof(int), s);
                if (npairs) {
                    hex8_fluid_mixed_mass_kernel<TReal, TGeom, TIdx><<<n_blocks(npairs), block_size, 0, s>>>(
                            solid, fluid, npairs, pair_solid, pair_fluid, B, options.basis, options.fluid_basis, d_counts, d_max);
                }
                int err = check_launch();
                if ((counts || max_it) && !err) {
                    unsigned long long h[5];
                    int                hmax = 0;
                    cudaMemcpyAsync(h, d_counts, sizeof(h), cudaMemcpyDeviceToHost, s);
                    cudaMemcpyAsync(&hmax, d_max, sizeof(int), cudaMemcpyDeviceToHost, s);
                    if (cudaStreamSynchronize(s) != cudaSuccess) err = SRESAMPLE_FAILURE;
                    if (counts) {
                        counts->separated    = (ptrdiff_t)h[TET4_PAIR_SEPARATED];
                        counts->fluid_inside = (ptrdiff_t)h[TET4_PAIR_FLUID_INSIDE];
                        counts->solid_inside = (ptrdiff_t)h[TET4_PAIR_SOLID_INSIDE];
                        counts->clipped      = (ptrdiff_t)h[TET4_PAIR_CLIPPED];
                        counts->overflow     = (ptrdiff_t)h[TET4_PAIR_OVERFLOW];
                    }
                    if (max_it) *max_it = hmax;
                }
                cudaFree(d_counts);
                return err;
            }

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
                                      void *const                  stream) {
                if (!npairs) return SRESAMPLE_SUCCESS;
                const bool with_u = U && U[0];
                hex8_fluid_apply_kernel<TReal, TGeom, TIdx><<<n_blocks(npairs), block_size, 0, (cudaStream_t)stream>>>(
                        solid, fluid, npairs, pair_solid, pair_fluid, B, with_u ? v[0] : nullptr, with_u ? v[1] : nullptr,
                        with_u ? v[2] : nullptr, W, with_u ? U[0] : nullptr, with_u ? U[1] : nullptr, with_u ? U[2] : nullptr);
                return check_launch();
            }

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
                                                void *const                  stream) {
                if (!npairs) return SRESAMPLE_SUCCESS;
                const bool with_g = g && g[0];
                hex8_fluid_apply_transpose_kernel<TReal, TGeom, TIdx><<<n_blocks(npairs), block_size, 0, (cudaStream_t)stream>>>(
                        solid, fluid, npairs, pair_solid, pair_fluid, B, with_g ? u[0] : nullptr, with_g ? u[1] : nullptr,
                        with_g ? u[2] : nullptr, c, with_g ? g[0] : nullptr, with_g ? g[1] : nullptr, with_g ? g[2] : nullptr);
                return check_launch();
            }

#define SRESAMPLE_INSTANTIATE_HEX8_FLUID_CUDA(TReal, TGeom, TIdx)                                                    \
    template int hex8_fluid_mixed_mass<TReal, TGeom, TIdx>(const MeshView<TGeom, TIdx> &, const MeshView<TGeom, TIdx> &,      \
                                                     const ptrdiff_t, const int *const, const int *const, TReal *const, \
                                                     const Hex8TransferOptions &, Tet4PairCounts *const, int *const,    \
                                                     void *const);                                                      \
    template int hex8_fluid_mixed_mass_apply<TReal, TGeom, TIdx>(const MeshView<TGeom, TIdx> &, const MeshView<TGeom, TIdx> &, \
                                                           const ptrdiff_t, const int *const, const int *const,          \
                                                           const TReal *const, const TReal *const[3], TReal *const,      \
                                                           TReal *const[3], void *const);                                \
    template int hex8_fluid_mixed_mass_apply_transpose<TReal, TGeom, TIdx>(                                                    \
            const MeshView<TGeom, TIdx> &, const MeshView<TGeom, TIdx> &, const ptrdiff_t, const int *const,             \
            const int *const, const TReal *const, const TReal *const[3], TReal *const, TReal *const[3], void *const);


            SRESAMPLE_FOREACH_MESH_TYPES(SRESAMPLE_INSTANTIATE_HEX8_FLUID_CUDA)

        }  // namespace cuda
    }  // namespace intersect
}  // namespace sresample
