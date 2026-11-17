#include "sresample_hex8_fluid_cuda.hpp"

#include "sresample_cuda_base.cuh"
#include "sresample_hex8_fluid_kernel.hpp"
#include "sresample_hex8_pieces_warp.cuh"
#include "sresample_instantiations.hpp"

// B by convex pieces one warp a pair (sresample_hex8_pieces_warp.cuh), the pairs that do not go by
// pieces split tet by split tet one thread a pair; or (Hex8TransferOptions::pieces 0, or neither
// side TRILINEAR) all of them split tet by split tet. The classes are counted per block in shared
// memory, then once per block globally; Newton's iterations by an atomic maximum.

namespace sresample {
    namespace intersect {
        namespace cuda {

            using sresample::cuda::block_size;
            using sresample::cuda::check_launch;
            using sresample::cuda::n_blocks;

            namespace {

                // Split tet by split tet, one thread a pair: all pairs, or (subset) the n_subset[0] pairs
                // subset[j] (the pairs the warp kernel left: n_subset read on the device).
                template <typename TReal, typename TGeom, typename TIdx>
                __global__ void hex8_fluid_mixed_mass_kernel(const MeshView<TGeom, TIdx> solid,
                                                       const MeshView<TGeom, TIdx> fluid,
                                                       const ptrdiff_t             npairs,
                                                       const int *const            pair_solid,
                                                       const int *const            pair_fluid,
                                                       TReal *const                B,
                                                       const int                   basis,
                                                       const int                   fluid_basis,
                                                       const int *const            subset,
                                                       const int *const            n_subset,
                                                       unsigned long long *const   counts,
                                                       int *const                  max_it) {
                    __shared__ unsigned int block_counts[5];
                    __shared__ int          block_max;
                    if (threadIdx.x < 5) block_counts[threadIdx.x] = 0;
                    if (threadIdx.x == 0) block_max = 0;
                    __syncthreads();
                    const Hex8Basis b    = basis == 1 ? Hex8Basis::SPLIT : Hex8Basis::TRILINEAR;
                    const Hex8Basis fb   = fluid_basis == 1 ? Hex8Basis::SPLIT : Hex8Basis::TRILINEAR;
                    const ptrdiff_t n    = subset ? (ptrdiff_t)n_subset[0] : npairs;
                    int             most = 0;
                    for (ptrdiff_t j = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; j < n; j += (ptrdiff_t)blockDim.x * gridDim.x) {
                        const ptrdiff_t k        = subset ? (ptrdiff_t)subset[j] : j;
                        int             local[5] = {0, 0, 0, 0, 0};
                        hex8_fluid_pair_mixed_mass(solid, fluid, npairs, k, pair_solid, pair_fluid, B, b, fb, false, local, &most);
                        for (int c = 0; c < 5; ++c) {
                            if (local[c]) atomicAdd(&block_counts[c], (unsigned int)local[c]);
                        }
                    }
                    atomicMax(&block_max, most);
                    __syncthreads();
                    if (threadIdx.x < 5 && block_counts[threadIdx.x]) atomicAdd(&counts[threadIdx.x], (unsigned long long)block_counts[threadIdx.x]);
                    if (threadIdx.x == 0) atomicMax(max_it, block_max);
                }

                // By convex pieces, one warp a pair (sresample_hex8_pieces_warp.cuh). A pair that does
                // not go by pieces (a non-convex K, an overflow) is appended to fallback, for the kernel
                // above.
                static constexpr int hex8_warps = block_size / hex8_warp_lanes;

                template <typename TReal, typename TGeom, typename TIdx>
                __global__ void __launch_bounds__(block_size) hex8_fluid_pieces_kernel(const MeshView<TGeom, TIdx> solid,
                                                                                       const MeshView<TGeom, TIdx> fluid,
                                                                                       const ptrdiff_t             npairs,
                                                                                       const int *const            pair_solid,
                                                                                       const int *const            pair_fluid,
                                                                                       TReal *const                B,
                                                                                       const int                   basis,
                                                                                       const int                   fluid_basis,
                                                                                       int *const                  fallback,
                                                                                       int *const                  n_fallback,
                                                                                       unsigned long long *const   counts,
                                                                                       int *const                  max_it) {
                    // Raw storage, doubles for the alignment (no initializers run on __shared__ data).
                    __shared__ double raw[(hex8_warps * sizeof(Hex8WarpPair) + sizeof(double) - 1) / sizeof(double)];
                    __shared__ unsigned int              block_counts[5];
                    __shared__ int                       block_max;
                    if (threadIdx.x < 5) block_counts[threadIdx.x] = 0;
                    if (threadIdx.x == 0) block_max = 0;
                    __syncthreads();
                    const int     lane  = threadIdx.x % hex8_warp_lanes;
                    const int     wid   = threadIdx.x / hex8_warp_lanes;
                    Hex8WarpPair &st    = reinterpret_cast<Hex8WarpPair *>(raw)[wid];
                    const int     ns    = nxe(solid.type);
                    const bool    s_tri = ns == 8 && basis != 1;
                    const bool    f_tri = fluid_basis != 1;
                    int           most = 0, local[5] = {0, 0, 0, 0, 0};
                    for (ptrdiff_t k = blockIdx.x * (ptrdiff_t)hex8_warps + wid; k < npairs; k += (ptrdiff_t)hex8_warps * gridDim.x) {
                        if (lane == 0) {
                            hex8_fluid_load(solid, pair_solid[k], ns, st.xs, st.ids_s);
                            hex8_fluid_load(fluid, pair_fluid[k], 8, st.xf, st.ids_f);
                            hex8_warp_setup(st, ns, 8, s_tri, f_tri);
                        }
                        hex8_syncwarp();
                        double acc[2] = {0, 0};
                        int    c5[5]  = {0, 0, 0, 0, 0};
                        const bool done = st.ok && hex8_warp_piece_sets(st, lane, s_tri && f_tri, 8, ns, acc, c5, &most);
                        if (!done) {
                            if (lane == 0) fallback[atomicAdd(n_fallback, 1)] = (int)k;
                            hex8_syncwarp();
                            continue;
                        }
                        for (int c = 0; c < 5; ++c) local[c] += c5[c];
                        for (int j = 0; j < 2; ++j) {
                            const int e = lane + hex8_warp_lanes * j, i = e / 8, a = e % 8;
                            if (a < ns) B[(ns * i + a) * npairs + k] = (TReal)acc[j];
                        }
                        hex8_syncwarp();
                    }
                    most = hex8_warp_max(most);
                    if (lane == 0) {
                        for (int c = 0; c < 5; ++c) {
                            if (local[c]) atomicAdd(&block_counts[c], (unsigned int)local[c]);
                        }
                        atomicMax(&block_max, most);
                    }
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
                // One stream-ordered allocation (no device-wide synchronisation): the classes, Newton's
                // maximum, and with pieces the fallback count and list.
                const cudaStream_t s      = (cudaStream_t)stream;
                const bool         pieces = options.pieces && (options.fluid_basis != 1 || (nxe(solid.type) == 8 && options.basis != 1));
                const size_t       head   = 5 * sizeof(unsigned long long) + 2 * sizeof(int);
                const size_t       bytes  = head + (pieces ? (size_t)npairs * sizeof(int) : 0);
                unsigned char     *buf    = nullptr;
                if (cudaMallocAsync((void **)&buf, bytes, s) != cudaSuccess) return SRESAMPLE_FAILURE;
                unsigned long long *d_counts   = (unsigned long long *)buf;
                int                *d_max      = (int *)(d_counts + 5);
                int                *d_nfall    = d_max + 1;
                int                *d_fallback = (int *)(buf + head);
                cudaMemsetAsync(buf, 0, head, s);
                if (npairs && pieces) {
                    const ptrdiff_t blocks = (npairs + hex8_warps - 1) / hex8_warps;
                    hex8_fluid_pieces_kernel<TReal, TGeom, TIdx><<<blocks < 65535 * 16 ? blocks : 65535 * 16, block_size, 0, s>>>(
                            solid, fluid, npairs, pair_solid, pair_fluid, B, options.basis, options.fluid_basis, d_fallback, d_nfall, d_counts, d_max);
                    // The pairs left, split tet by split tet (a grid for all; it reads their number).
                    hex8_fluid_mixed_mass_kernel<TReal, TGeom, TIdx><<<n_blocks(npairs) < 1024 ? n_blocks(npairs) : 1024, block_size, 0, s>>>(
                            solid, fluid, npairs, pair_solid, pair_fluid, B, options.basis, options.fluid_basis, d_fallback, d_nfall, d_counts, d_max);
                } else if (npairs) {
                    hex8_fluid_mixed_mass_kernel<TReal, TGeom, TIdx><<<n_blocks(npairs), block_size, 0, s>>>(
                            solid, fluid, npairs, pair_solid, pair_fluid, B, options.basis, options.fluid_basis, nullptr, nullptr, d_counts, d_max);
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
                cudaFreeAsync(buf, s);
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
