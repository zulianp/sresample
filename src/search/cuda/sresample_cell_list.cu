#include "sresample_cell_list_cuda.hpp"

#include "sresample_config.hpp"

// hipCUB mirrors CUB's interface in its own namespace, so under HIP `cub` names it.
#if defined(SRESAMPLE_ENABLE_HIP)
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif

#include "sresample_cell_list_kernel.hpp"
#include "sresample_cuda_base.cuh"
#include "sresample_instantiations.hpp"

// The cell list on the device. The build, per level: count each box's columns, scan, write one
// (column << 32 | lower z) key per entry, radix sort, column pointers by binary search in the sorted
// keys, and the running maximum of the upper z by a scan by key. Nothing serial; deterministic
// (the sort is stable and every box writes its entries at its scanned offset).

namespace sresample {
    namespace search {
        namespace cuda {

            using sresample::cuda::block_size;
            using sresample::cuda::check_launch;
            using sresample::cuda::n_blocks;

            namespace {

#define SRESAMPLE_CL_CHECK(call)                                                       \
    do {                                                                               \
        const cudaError_t e__ = (call);                                                \
        if (e__ != cudaSuccess) {                                                      \
            SRESAMPLE_LOG(0, "cell list (CUDA): %s\n", cudaGetErrorString(e__));       \
            return SRESAMPLE_FAILURE;                                                  \
        }                                                                              \
    } while (0)

                SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE float key_float(uint32_t u) {
                    u = (u & 0x80000000u) ? (u & 0x7fffffffu) : ~u;
                    float f;
                    memcpy(&f, &u, sizeof(f));
                    return f;
                }

                // Per-level statistics, as order-preserving keys: min x, min y, max x, max y, max extent.
                static constexpr int n_stats = 5;

                __global__ void class_kernel(const Boxes b, int *const cls, unsigned int *const hist) {
                    for (ptrdiff_t i = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; i < b.n; i += (ptrdiff_t)blockDim.x * gridDim.x) {
                        const int c = box_size_class(b.hi[0][i] - b.lo[0][i], b.hi[1][i] - b.lo[1][i]);
                        cls[i]      = c;
                        atomicAdd(&hist[c], 1u);
                    }
                }

                __global__ void level_kernel(const Boxes        b,
                                             const int          cmin,
                                             const int          spl,
                                             const int          max_levels,
                                             int *const         level,
                                             unsigned int *const stats) {
                    for (ptrdiff_t i = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; i < b.n; i += (ptrdiff_t)blockDim.x * gridDim.x) {
                        int lv   = (level[i] - cmin) / spl;
                        lv       = lv < max_levels - 1 ? lv : max_levels - 1;
                        level[i] = lv;
                        unsigned int *s  = stats + n_stats * lv;
                        const float   dx = b.hi[0][i] - b.lo[0][i], dy = b.hi[1][i] - b.lo[1][i];
                        atomicMin(&s[0], float_key(b.lo[0][i]));
                        atomicMin(&s[1], float_key(b.lo[1][i]));
                        atomicMax(&s[2], float_key(b.hi[0][i]));
                        atomicMax(&s[3], float_key(b.hi[1][i]));
                        atomicMax(&s[4], float_key(dx > dy ? dx : dy));
                    }
                }

                __global__ void entries_count_kernel(const Boxes         b,
                                                     const int *const    level,
                                                     const int           lv,
                                                     const CellListLevel l,
                                                     ptrdiff_t *const    count) {
                    for (ptrdiff_t i = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; i < b.n; i += (ptrdiff_t)blockDim.x * gridDim.x) {
                        count[i] = level[i] != lv ? 0
                                                  : (ptrdiff_t)(column_of(l, 0, b.hi[0][i]) - column_of(l, 0, b.lo[0][i]) + 1) *
                                                            (column_of(l, 1, b.hi[1][i]) - column_of(l, 1, b.lo[1][i]) + 1);
                    }
                }

                __global__ void entries_fill_kernel(const Boxes            b,
                                                    const int *const       level,
                                                    const int              lv,
                                                    const CellListLevel    l,
                                                    const ptrdiff_t *const ptr,
                                                    uint64_t *const        keys,
                                                    int *const             vals) {
                    for (ptrdiff_t i = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; i < b.n; i += (ptrdiff_t)blockDim.x * gridDim.x) {
                        if (level[i] != lv) continue;
                        ptrdiff_t       k  = ptr[i];
                        const uint32_t  zk = float_key(b.lo[2][i]);
                        for (int cy = column_of(l, 1, b.lo[1][i]); cy <= column_of(l, 1, b.hi[1][i]); ++cy) {
                            for (int cx = column_of(l, 0, b.lo[0][i]); cx <= column_of(l, 0, b.hi[0][i]); ++cx) {
                                const uint64_t col = (uint64_t)cy * l.n[0] + cx;
                                keys[k]            = (col << 32) | zk;
                                vals[k]            = (int)i;
                                ++k;
                            }
                        }
                    }
                }

                // col_ptr[c] = the first entry of column c (c <= n_cols: col_ptr[n_cols] = n_entries).
                __global__ void col_ptr_kernel(const ptrdiff_t n_cols,
                                               const uint64_t *const keys,
                                               const ptrdiff_t n_entries,
                                               ptrdiff_t *const col_ptr) {
                    for (ptrdiff_t c = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; c <= n_cols; c += (ptrdiff_t)blockDim.x * gridDim.x) {
                        const uint64_t key   = (uint64_t)c << 32;
                        ptrdiff_t      first = 0, last = n_entries;
                        while (first < last) {
                            const ptrdiff_t mid = first + (last - first) / 2;
                            if (keys[mid] < key) {
                                first = mid + 1;
                            } else {
                                last = mid;
                            }
                        }
                        col_ptr[c] = first;
                    }
                }

                __global__ void entry_z_kernel(const ptrdiff_t       n_entries,
                                               const uint64_t *const keys,
                                               const int *const      entry_box,
                                               const Boxes           b,
                                               float *const          zlo,
                                               float *const          zhi,
                                               unsigned int *const   column) {
                    for (ptrdiff_t e = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < n_entries; e += (ptrdiff_t)blockDim.x * gridDim.x) {
                        const int box = entry_box[e];
                        zlo[e]        = b.lo[2][box];
                        zhi[e]        = b.hi[2][box];
                        column[e]     = (unsigned int)(keys[e] >> 32);
                    }
                }

                struct MaxOp {
                    __device__ float operator()(const float a, const float b) const { return a > b ? a : b; }
                };
                struct EqualOp {
                    __device__ bool operator()(const unsigned int a, const unsigned int b) const { return a == b; }
                };

                struct CountVisit {
                    ptrdiff_t                *n;
                    SRESAMPLE_HOST_DEVICE void operator()(const int) const { ++*n; }
                };

                struct FillVisit {
                    int                      *pair_query;
                    int                      *pair_box;
                    ptrdiff_t                *k;
                    int                       q;
                    SRESAMPLE_HOST_DEVICE void operator()(const int b) const {
                        pair_query[*k] = q;
                        pair_box[*k]   = b;
                        ++*k;
                    }
                };

                __global__ void count_kernel(const CellList list, const Boxes q, ptrdiff_t *const count) {
                    for (ptrdiff_t i = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; i < q.n; i += (ptrdiff_t)blockDim.x * gridDim.x) {
                        const float lo[3] = {q.lo[0][i], q.lo[1][i], q.lo[2][i]};
                        const float hi[3] = {q.hi[0][i], q.hi[1][i], q.hi[2][i]};
                        ptrdiff_t   n     = 0;
                        cell_list_visit(list, lo, hi, CountVisit{&n});
                        count[i] = n;
                    }
                }

                __global__ void fill_kernel(const CellList         list,
                                            const Boxes            q,
                                            const ptrdiff_t *const ptr,
                                            int *const             pair_query,
                                            int *const             pair_box) {
                    for (ptrdiff_t i = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; i < q.n; i += (ptrdiff_t)blockDim.x * gridDim.x) {
                        const float lo[3] = {q.lo[0][i], q.lo[1][i], q.lo[2][i]};
                        const float hi[3] = {q.hi[0][i], q.hi[1][i], q.hi[2][i]};
                        ptrdiff_t   k     = ptr[i];
                        cell_list_visit(list, lo, hi, FillVisit{pair_query, pair_box, &k, (int)i});
                    }
                }

                template <typename TGeom, typename TIdx>
                __global__ void tet4_boxes_kernel(const MeshView<TGeom, TIdx> mesh,
                                                  const ptrdiff_t             begin,
                                                  const ptrdiff_t             end,
                                                  const double                ox,
                                                  const double                oy,
                                                  const double                oz,
                                                  float *const lo0, float *const lo1, float *const lo2,
                                                  float *const hi0, float *const hi1, float *const hi2) {
                    const double origin[3] = {ox, oy, oz};
                    for (ptrdiff_t e = begin + blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < end; e += (ptrdiff_t)blockDim.x * gridDim.x) {
                        float l[3], h[3];
                        tet4_box(mesh, e, origin, l, h);
                        const ptrdiff_t i = e - begin;
                        lo0[i]            = l[0];
                        lo1[i]            = l[1];
                        lo2[i]            = l[2];
                        hi0[i]            = h[0];
                        hi1[i]            = h[1];
                        hi2[i]            = h[2];
                    }
                }

                // An exclusive scan of n + 1 entries through a temporary (in-place-safe).
                int scan_ptrdiff(const ptrdiff_t *in, ptrdiff_t *out, const ptrdiff_t n1, cudaStream_t s) {
                    size_t bytes = 0;
                    SRESAMPLE_CL_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, bytes, in, out, (int)n1, s));
                    void *tmp = nullptr;
                    SRESAMPLE_CL_CHECK(cudaMalloc(&tmp, bytes));
                    SRESAMPLE_CL_CHECK(cub::DeviceScan::ExclusiveSum(tmp, bytes, in, out, (int)n1, s));
                    SRESAMPLE_CL_CHECK(cudaStreamSynchronize(s));
                    cudaFree(tmp);
                    return SRESAMPLE_SUCCESS;
                }

                int build_level(const Boxes &b, const int *level, const int lv, CellListLevel &l, cudaStream_t s) {
                    ptrdiff_t *count = nullptr, *ptr = nullptr;
                    SRESAMPLE_CL_CHECK(cudaMalloc(&count, sizeof(ptrdiff_t) * (b.n + 1)));
                    SRESAMPLE_CL_CHECK(cudaMalloc(&ptr, sizeof(ptrdiff_t) * (b.n + 1)));
                    SRESAMPLE_CL_CHECK(cudaMemsetAsync(count + b.n, 0, sizeof(ptrdiff_t), s));
                    entries_count_kernel<<<n_blocks(b.n), block_size, 0, s>>>(b, level, lv, l, count);
                    if (check_launch()) return SRESAMPLE_FAILURE;
                    if (scan_ptrdiff(count, ptr, b.n + 1, s)) return SRESAMPLE_FAILURE;
                    ptrdiff_t n_entries = 0;
                    SRESAMPLE_CL_CHECK(cudaMemcpy(&n_entries, ptr + b.n, sizeof(ptrdiff_t), cudaMemcpyDeviceToHost));
                    l.n_entries = n_entries;

                    const ptrdiff_t n_cols = (ptrdiff_t)l.n[0] * l.n[1];
                    SRESAMPLE_CL_CHECK(cudaMalloc(&l.col_ptr, sizeof(ptrdiff_t) * (n_cols + 1)));
                    if (!n_entries) {
                        SRESAMPLE_CL_CHECK(cudaMemsetAsync(l.col_ptr, 0, sizeof(ptrdiff_t) * (n_cols + 1), s));
                        cudaFree(count);
                        cudaFree(ptr);
                        return SRESAMPLE_SUCCESS;
                    }

                    uint64_t *keys = nullptr, *keys_sorted = nullptr;
                    int      *vals = nullptr;
                    SRESAMPLE_CL_CHECK(cudaMalloc(&keys, sizeof(uint64_t) * n_entries));
                    SRESAMPLE_CL_CHECK(cudaMalloc(&keys_sorted, sizeof(uint64_t) * n_entries));
                    SRESAMPLE_CL_CHECK(cudaMalloc(&vals, sizeof(int) * n_entries));
                    SRESAMPLE_CL_CHECK(cudaMalloc(&l.entry_box, sizeof(int) * n_entries));
                    entries_fill_kernel<<<n_blocks(b.n), block_size, 0, s>>>(b, level, lv, l, ptr, keys, vals);
                    if (check_launch()) return SRESAMPLE_FAILURE;

                    // Sort by (column, lower z); the column needs only as many bits as columns.
                    int col_bits = 1;
                    while (((ptrdiff_t)1 << col_bits) < n_cols) ++col_bits;
                    size_t bytes = 0;
                    SRESAMPLE_CL_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys, keys_sorted, vals, l.entry_box, (int)n_entries, 0, 32 + col_bits, s));
                    void *tmp = nullptr;
                    SRESAMPLE_CL_CHECK(cudaMalloc(&tmp, bytes));
                    SRESAMPLE_CL_CHECK(cub::DeviceRadixSort::SortPairs(tmp, bytes, keys, keys_sorted, vals, l.entry_box, (int)n_entries, 0, 32 + col_bits, s));

                    col_ptr_kernel<<<n_blocks(n_cols + 1), block_size, 0, s>>>(n_cols, keys_sorted, n_entries, l.col_ptr);
                    if (check_launch()) return SRESAMPLE_FAILURE;

                    float        *zhi    = nullptr;
                    unsigned int *column = nullptr;
                    SRESAMPLE_CL_CHECK(cudaMalloc(&l.entry_zlo, sizeof(float) * n_entries));
                    SRESAMPLE_CL_CHECK(cudaMalloc(&l.entry_zhi_max, sizeof(float) * n_entries));
                    SRESAMPLE_CL_CHECK(cudaMalloc(&zhi, sizeof(float) * n_entries));
                    SRESAMPLE_CL_CHECK(cudaMalloc(&column, sizeof(unsigned int) * n_entries));
                    entry_z_kernel<<<n_blocks(n_entries), block_size, 0, s>>>(n_entries, keys_sorted, l.entry_box, b, l.entry_zlo, zhi, column);
                    if (check_launch()) return SRESAMPLE_FAILURE;

                    size_t bytes2 = 0;
                    SRESAMPLE_CL_CHECK(cub::DeviceScan::InclusiveScanByKey(nullptr, bytes2, column, zhi, l.entry_zhi_max, MaxOp(), (int)n_entries, EqualOp(), s));
                    void *tmp2 = nullptr;
                    SRESAMPLE_CL_CHECK(cudaMalloc(&tmp2, bytes2));
                    SRESAMPLE_CL_CHECK(cub::DeviceScan::InclusiveScanByKey(tmp2, bytes2, column, zhi, l.entry_zhi_max, MaxOp(), (int)n_entries, EqualOp(), s));
                    SRESAMPLE_CL_CHECK(cudaStreamSynchronize(s));

                    cudaFree(tmp);
                    cudaFree(tmp2);
                    cudaFree(zhi);
                    cudaFree(column);
                    cudaFree(keys);
                    cudaFree(keys_sorted);
                    cudaFree(vals);
                    cudaFree(count);
                    cudaFree(ptr);
                    return SRESAMPLE_SUCCESS;
                }

            }  // namespace

            int cell_list_build(const Boxes &boxes, const CellListOptions &options, CellList *list, void *const stream) {
                const cudaStream_t s = (cudaStream_t)stream;
                const ptrdiff_t    n = boxes.n;
                list->space          = MemorySpace::DEVICE;
                list->n_boxes        = n;
                for (int d = 0; d < 3; ++d) {
                    SRESAMPLE_CL_CHECK(cudaMalloc(&list->box_lo[d], sizeof(float) * (n ? n : 1)));
                    SRESAMPLE_CL_CHECK(cudaMalloc(&list->box_hi[d], sizeof(float) * (n ? n : 1)));
                    if (n) {
                        SRESAMPLE_CL_CHECK(cudaMemcpyAsync(list->box_lo[d], boxes.lo[d], sizeof(float) * n, cudaMemcpyDeviceToDevice, s));
                        SRESAMPLE_CL_CHECK(cudaMemcpyAsync(list->box_hi[d], boxes.hi[d], sizeof(float) * n, cudaMemcpyDeviceToDevice, s));
                    }
                }
                if (!n) return SRESAMPLE_SUCCESS;
                Boxes b;
                b.n = n;
                for (int d = 0; d < 3; ++d) {
                    b.lo[d] = list->box_lo[d];
                    b.hi[d] = list->box_hi[d];
                }

                // Size classes, levels and each level's region.
                int          *level = nullptr;
                unsigned int *hist  = nullptr, *stats = nullptr;
                SRESAMPLE_CL_CHECK(cudaMalloc(&level, sizeof(int) * n));
                SRESAMPLE_CL_CHECK(cudaMalloc(&hist, sizeof(unsigned int) * 256));
                SRESAMPLE_CL_CHECK(cudaMalloc(&stats, sizeof(unsigned int) * n_stats * cell_list_max_levels));
                SRESAMPLE_CL_CHECK(cudaMemsetAsync(hist, 0, sizeof(unsigned int) * 256, s));
                class_kernel<<<n_blocks(n), block_size, 0, s>>>(b, level, hist);
                if (check_launch()) return SRESAMPLE_FAILURE;
                unsigned int h_hist[256];
                SRESAMPLE_CL_CHECK(cudaMemcpyAsync(h_hist, hist, sizeof(h_hist), cudaMemcpyDeviceToHost, s));
                SRESAMPLE_CL_CHECK(cudaStreamSynchronize(s));
                int cmin = 0;
                while (cmin < 255 && !h_hist[cmin]) ++cmin;
                int cmax = 255;
                while (cmax > 0 && !h_hist[cmax]) --cmax;
                const int max_levels = options.max_levels < 1 ? 1 : (options.max_levels > cell_list_max_levels ? cell_list_max_levels : options.max_levels);
                const int spl        = options.sizes_per_level < 1 ? 1 : options.sizes_per_level;
                int       n_levels   = (cmax - cmin) / spl + 1;
                n_levels             = n_levels < max_levels ? n_levels : max_levels;
                list->n_levels       = n_levels;

                unsigned int h_stats[n_stats * cell_list_max_levels];
                for (int lv = 0; lv < cell_list_max_levels; ++lv) {
                    h_stats[n_stats * lv + 0] = h_stats[n_stats * lv + 1] = 0xffffffffu;
                    h_stats[n_stats * lv + 2] = h_stats[n_stats * lv + 3] = h_stats[n_stats * lv + 4] = 0u;
                }
                SRESAMPLE_CL_CHECK(cudaMemcpyAsync(stats, h_stats, sizeof(h_stats), cudaMemcpyHostToDevice, s));
                level_kernel<<<n_blocks(n), block_size, 0, s>>>(b, cmin, spl, max_levels, level, stats);
                if (check_launch()) return SRESAMPLE_FAILURE;
                SRESAMPLE_CL_CHECK(cudaMemcpyAsync(h_stats, stats, sizeof(h_stats), cudaMemcpyDeviceToHost, s));
                SRESAMPLE_CL_CHECK(cudaStreamSynchronize(s));

                for (int lv = 0; lv < n_levels; ++lv) {
                    CellListLevel     &l  = list->level[lv];
                    const unsigned int *st = h_stats + n_stats * lv;
                    if (st[0] == 0xffffffffu) {  // an empty level
                        l.n[0] = l.n[1] = 1;
                        l.inv_h[0] = l.inv_h[1] = 1;
                        SRESAMPLE_CL_CHECK(cudaMalloc(&l.col_ptr, 2 * sizeof(ptrdiff_t)));
                        SRESAMPLE_CL_CHECK(cudaMemsetAsync(l.col_ptr, 0, 2 * sizeof(ptrdiff_t), s));
                        continue;
                    }
                    const float lo[2] = {key_float(st[0]), key_float(st[1])};
                    const float hi[2] = {key_float(st[2]), key_float(st[3])};
                    float       h     = key_float(st[4]);
                    if (!(h > 0)) h = fmaxf(fmaxf(hi[0] - lo[0], hi[1] - lo[1]), 1e-30f);
                    for (;;) {
                        for (int a = 0; a < 2; ++a) {
                            const double ext = (double)hi[a] - (double)lo[a];
                            l.n[a]           = (int)fmax(1.0, ceil(ext / h));
                        }
                        if ((ptrdiff_t)l.n[0] * l.n[1] <= options.max_columns) break;
                        h *= 2;
                    }
                    for (int a = 0; a < 2; ++a) {
                        l.origin[a] = lo[a];
                        l.inv_h[a]  = 1 / h;
                    }
                    if (build_level(b, level, lv, l, s)) return SRESAMPLE_FAILURE;
                }
                cudaFree(level);
                cudaFree(hist);
                cudaFree(stats);
                return SRESAMPLE_SUCCESS;
            }

            int cell_list_destroy(CellList *list) {
                for (int d = 0; d < 3; ++d) {
                    cudaFree(list->box_lo[d]);
                    cudaFree(list->box_hi[d]);
                }
                for (int lv = 0; lv < cell_list_max_levels; ++lv) {
                    CellListLevel &l = list->level[lv];
                    cudaFree(l.col_ptr);
                    cudaFree(l.entry_box);
                    cudaFree(l.entry_zlo);
                    cudaFree(l.entry_zhi_max);
                }
                *list = CellList();
                return SRESAMPLE_SUCCESS;
            }

            int cell_list_count(const CellList &list, const Boxes &queries, ptrdiff_t *count, void *const stream) {
                if (!queries.n) return SRESAMPLE_SUCCESS;
                count_kernel<<<n_blocks(queries.n), block_size, 0, (cudaStream_t)stream>>>(list, queries, count);
                return check_launch();
            }

            int cell_list_fill(const CellList  &list,
                               const Boxes     &queries,
                               const ptrdiff_t *ptr,
                               int             *pair_query,
                               int             *pair_box,
                               void *const      stream) {
                if (!queries.n) return SRESAMPLE_SUCCESS;
                fill_kernel<<<n_blocks(queries.n), block_size, 0, (cudaStream_t)stream>>>(list, queries, ptr, pair_query, pair_box);
                return check_launch();
            }

            int exclusive_scan(ptrdiff_t *a, const ptrdiff_t n, ptrdiff_t *total, void *const stream) {
                const cudaStream_t s   = (cudaStream_t)stream;
                ptrdiff_t         *tmp = nullptr;
                SRESAMPLE_CL_CHECK(cudaMalloc(&tmp, sizeof(ptrdiff_t) * (n + 1)));
                if (scan_ptrdiff(a, tmp, n + 1, s)) return SRESAMPLE_FAILURE;
                SRESAMPLE_CL_CHECK(cudaMemcpyAsync(a, tmp, sizeof(ptrdiff_t) * (n + 1), cudaMemcpyDeviceToDevice, s));
                SRESAMPLE_CL_CHECK(cudaMemcpyAsync(total, tmp + n, sizeof(ptrdiff_t), cudaMemcpyDeviceToHost, s));
                SRESAMPLE_CL_CHECK(cudaStreamSynchronize(s));
                cudaFree(tmp);
                return SRESAMPLE_SUCCESS;
            }

            template <typename TReal, typename TGeom, typename TIdx>
            int tet4_boxes(const MeshView<TGeom, TIdx> &mesh,
                           const ptrdiff_t              begin,
                           const ptrdiff_t              end,
                           const double                 origin[3],
                           float *const                 lo[3],
                           float *const                 hi[3],
                           void *const                  stream) {
                if (end <= begin) return SRESAMPLE_SUCCESS;
                tet4_boxes_kernel<TGeom, TIdx><<<n_blocks(end - begin), block_size, 0, (cudaStream_t)stream>>>(
                        mesh, begin, end, origin[0], origin[1], origin[2], lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]);
                return check_launch();
            }

#define SRESAMPLE_INSTANTIATE_TET4_BOXES_CUDA(TReal, TGeom, TIdx)                                                \
    template int tet4_boxes<TReal, TGeom, TIdx>(const MeshView<TGeom, TIdx> &, const ptrdiff_t, const ptrdiff_t, \
                                                const double[3], float *const[3], float *const[3], void *const);

            SRESAMPLE_FOREACH_MESH_TYPES(SRESAMPLE_INSTANTIATE_TET4_BOXES_CUDA)

        }  // namespace cuda
    }  // namespace search
}  // namespace sresample
