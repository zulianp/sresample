#include "sresample_cell_list.hpp"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sresample_cell_list_kernel.hpp"
#include "sresample_instantiations.hpp"

#ifdef SRESAMPLE_ENABLE_CUDA
#include "sresample_cell_list_cuda.hpp"
#endif

namespace sresample {
    namespace search {

        namespace {

            struct Entry {
                uint64_t key;
                int      box;
            };

            int compare_entries(const void *a, const void *b) {
                const Entry *x = (const Entry *)a, *y = (const Entry *)b;
                if (x->key != y->key) return x->key < y->key ? -1 : 1;
                return x->box < y->box ? -1 : (x->box > y->box ? 1 : 0);
            }

            void free_host(CellList *list) {
                for (int d = 0; d < 3; ++d) {
                    free(list->box_lo[d]);
                    free(list->box_hi[d]);
                    list->box_lo[d] = list->box_hi[d] = nullptr;
                }
                for (int lv = 0; lv < cell_list_max_levels; ++lv) {
                    CellListLevel &l = list->level[lv];
                    free(l.col_ptr);
                    free(l.entry_box);
                    free(l.entry_zlo);
                    free(l.entry_zhi_max);
                    l = CellListLevel();
                }
                list->n_levels = 0;
                list->n_boxes  = 0;
            }

            int build_host(const Boxes &boxes, const CellListOptions &options, CellList *list) {
                const ptrdiff_t n = boxes.n;
                list->space       = MemorySpace::HOST;
                list->n_boxes     = n;
                for (int d = 0; d < 3; ++d) {
                    list->box_lo[d] = (float *)malloc(sizeof(float) * (n ? n : 1));
                    list->box_hi[d] = (float *)malloc(sizeof(float) * (n ? n : 1));
                    if (n) {
                        memcpy(list->box_lo[d], boxes.lo[d], sizeof(float) * n);
                        memcpy(list->box_hi[d], boxes.hi[d], sizeof(float) * n);
                    }
                }
                if (!n) return SRESAMPLE_SUCCESS;

                // Size classes and levels.
                int *level = (int *)malloc(sizeof(int) * n);
                int  cmin  = 255;
                for (ptrdiff_t b = 0; b < n; ++b) {
                    level[b] = box_size_class(boxes.hi[0][b] - boxes.lo[0][b], boxes.hi[1][b] - boxes.lo[1][b]);
                    cmin     = level[b] < cmin ? level[b] : cmin;
                }
                const int max_levels = options.max_levels < 1 ? 1 : (options.max_levels > cell_list_max_levels ? cell_list_max_levels : options.max_levels);
                const int spl        = options.sizes_per_level < 1 ? 1 : options.sizes_per_level;
                int       n_levels   = 0;
                for (ptrdiff_t b = 0; b < n; ++b) {
                    int lv   = (level[b] - cmin) / spl;
                    lv       = lv < max_levels - 1 ? lv : max_levels - 1;
                    level[b] = lv;
                    n_levels = lv + 1 > n_levels ? lv + 1 : n_levels;
                }
                list->n_levels = n_levels;

                for (int lv = 0; lv < n_levels; ++lv) {
                    CellListLevel &l = list->level[lv];
                    // The level's region and its largest box.
                    float     lo[2] = {INFINITY, INFINITY}, hi[2] = {-INFINITY, -INFINITY}, h = 0;
                    ptrdiff_t count = 0;
                    for (ptrdiff_t b = 0; b < n; ++b) {
                        if (level[b] != lv) continue;
                        ++count;
                        for (int a = 0; a < 2; ++a) {
                            lo[a]         = boxes.lo[a][b] < lo[a] ? boxes.lo[a][b] : lo[a];
                            hi[a]         = boxes.hi[a][b] > hi[a] ? boxes.hi[a][b] : hi[a];
                            const float e = boxes.hi[a][b] - boxes.lo[a][b];
                            h             = e > h ? e : h;
                        }
                    }
                    if (!count) {
                        l.n[0] = l.n[1] = 1;
                        l.inv_h[0] = l.inv_h[1] = 1;
                        l.col_ptr  = (ptrdiff_t *)calloc(2, sizeof(ptrdiff_t));
                        continue;
                    }
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

                    // One entry per (box, column), sorted by (column, lower z, box).
                    ptrdiff_t n_entries = 0;
                    for (ptrdiff_t b = 0; b < n; ++b) {
                        if (level[b] != lv) continue;
                        n_entries += (ptrdiff_t)(column_of(l, 0, boxes.hi[0][b]) - column_of(l, 0, boxes.lo[0][b]) + 1) *
                                     (column_of(l, 1, boxes.hi[1][b]) - column_of(l, 1, boxes.lo[1][b]) + 1);
                    }
                    Entry    *entries = (Entry *)malloc(sizeof(Entry) * n_entries);
                    ptrdiff_t k       = 0;
                    for (ptrdiff_t b = 0; b < n; ++b) {
                        if (level[b] != lv) continue;
                        for (int cy = column_of(l, 1, boxes.lo[1][b]); cy <= column_of(l, 1, boxes.hi[1][b]); ++cy) {
                            for (int cx = column_of(l, 0, boxes.lo[0][b]); cx <= column_of(l, 0, boxes.hi[0][b]); ++cx) {
                                const uint64_t col = (uint64_t)cy * l.n[0] + cx;
                                entries[k].key     = (col << 32) | float_key(boxes.lo[2][b]);
                                entries[k].box     = (int)b;
                                ++k;
                            }
                        }
                    }
                    qsort(entries, n_entries, sizeof(Entry), compare_entries);

                    const ptrdiff_t n_cols = (ptrdiff_t)l.n[0] * l.n[1];
                    l.n_entries            = n_entries;
                    l.col_ptr              = (ptrdiff_t *)calloc(n_cols + 1, sizeof(ptrdiff_t));
                    l.entry_box            = (int *)malloc(sizeof(int) * n_entries);
                    l.entry_zlo            = (float *)malloc(sizeof(float) * n_entries);
                    l.entry_zhi_max        = (float *)malloc(sizeof(float) * n_entries);
                    for (ptrdiff_t e = 0; e < n_entries; ++e) {
                        const ptrdiff_t col = (ptrdiff_t)(entries[e].key >> 32);
                        l.col_ptr[col + 1]++;
                        l.entry_box[e] = entries[e].box;
                        l.entry_zlo[e] = boxes.lo[2][entries[e].box];
                    }
                    for (ptrdiff_t c = 0; c < n_cols; ++c) l.col_ptr[c + 1] += l.col_ptr[c];
                    for (ptrdiff_t c = 0; c < n_cols; ++c) {
                        float m = -INFINITY;
                        for (ptrdiff_t e = l.col_ptr[c]; e < l.col_ptr[c + 1]; ++e) {
                            const float z     = boxes.hi[2][l.entry_box[e]];
                            m                 = z > m ? z : m;
                            l.entry_zhi_max[e] = m;
                        }
                    }
                    free(entries);
                }
                free(level);
                return SRESAMPLE_SUCCESS;
            }

            struct CountVisit {
                ptrdiff_t *n;
                void       operator()(const int) const { ++*n; }
            };

            struct FillVisit {
                int       *pair_query;
                int       *pair_box;
                ptrdiff_t *k;
                int        q;
                void       operator()(const int b) const {
                    pair_query[*k] = q;
                    pair_box[*k]   = b;
                    ++*k;
                }
            };

        }  // namespace

        int cell_list_build(const Boxes &boxes, const CellListOptions &options, CellList *list, const Exec &exec) {
            *list = CellList();
            if (exec.space == MemorySpace::DEVICE) {
#ifdef SRESAMPLE_ENABLE_CUDA
                return cuda::cell_list_build(boxes, options, list, exec.stream);
#else
                SRESAMPLE_LOG(0, "device execution requested, but sresample was built without CUDA\n");
                return SRESAMPLE_FAILURE;
#endif
            }
            return build_host(boxes, options, list);
        }

        int cell_list_destroy(CellList *list) {
            if (list->space == MemorySpace::DEVICE) {
#ifdef SRESAMPLE_ENABLE_CUDA
                return cuda::cell_list_destroy(list);
#else
                return SRESAMPLE_FAILURE;
#endif
            }
            free_host(list);
            return SRESAMPLE_SUCCESS;
        }

        int cell_list_count(const CellList &list, const Boxes &queries, ptrdiff_t *count, const Exec &exec) {
            if (exec.space == MemorySpace::DEVICE) {
#ifdef SRESAMPLE_ENABLE_CUDA
                return cuda::cell_list_count(list, queries, count, exec.stream);
#else
                return SRESAMPLE_FAILURE;
#endif
            }
#pragma omp parallel for
            for (ptrdiff_t q = 0; q < queries.n; ++q) {
                const float lo[3] = {queries.lo[0][q], queries.lo[1][q], queries.lo[2][q]};
                const float hi[3] = {queries.hi[0][q], queries.hi[1][q], queries.hi[2][q]};
                ptrdiff_t   n     = 0;
                cell_list_visit(list, lo, hi, CountVisit{&n});
                count[q] = n;
            }
            return SRESAMPLE_SUCCESS;
        }

        int cell_list_fill(const CellList  &list,
                           const Boxes     &queries,
                           const ptrdiff_t *ptr,
                           int             *pair_query,
                           int             *pair_box,
                           const Exec      &exec) {
            if (exec.space == MemorySpace::DEVICE) {
#ifdef SRESAMPLE_ENABLE_CUDA
                return cuda::cell_list_fill(list, queries, ptr, pair_query, pair_box, exec.stream);
#else
                return SRESAMPLE_FAILURE;
#endif
            }
#pragma omp parallel for
            for (ptrdiff_t q = 0; q < queries.n; ++q) {
                const float lo[3] = {queries.lo[0][q], queries.lo[1][q], queries.lo[2][q]};
                const float hi[3] = {queries.hi[0][q], queries.hi[1][q], queries.hi[2][q]};
                ptrdiff_t   k     = ptr[q];
                cell_list_visit(list, lo, hi, FillVisit{pair_query, pair_box, &k, (int)q});
            }
            return SRESAMPLE_SUCCESS;
        }

        int exclusive_scan(ptrdiff_t *a, const ptrdiff_t n, ptrdiff_t *total, const Exec &exec) {
            if (exec.space == MemorySpace::DEVICE) {
#ifdef SRESAMPLE_ENABLE_CUDA
                return cuda::exclusive_scan(a, n, total, exec.stream);
#else
                return SRESAMPLE_FAILURE;
#endif
            }
            ptrdiff_t s = 0;
            for (ptrdiff_t i = 0; i < n; ++i) {
                const ptrdiff_t v = a[i];
                a[i]              = s;
                s += v;
            }
            a[n]   = s;
            *total = s;
            return SRESAMPLE_SUCCESS;
        }

        template <typename TReal, typename TGeom, typename TIdx>
        int element_boxes(const MeshView<TGeom, TIdx> &mesh,
                          const ptrdiff_t              begin,
                          const ptrdiff_t              end,
                          const double                 origin[3],
                          float *const                 lo[3],
                          float *const                 hi[3],
                          const Exec                  &exec) {
            if (exec.space == MemorySpace::DEVICE) {
#ifdef SRESAMPLE_ENABLE_CUDA
                return cuda::element_boxes<TReal, TGeom, TIdx>(mesh, begin, end, origin, lo, hi, exec.stream);
#else
                return SRESAMPLE_FAILURE;
#endif
            }
#pragma omp parallel for
            for (ptrdiff_t e = begin; e < end; ++e) {
                float l[3], h[3];
                element_box(mesh, e, origin, l, h);
                for (int d = 0; d < 3; ++d) {
                    lo[d][e - begin] = l[d];
                    hi[d][e - begin] = h[d];
                }
            }
            return SRESAMPLE_SUCCESS;
        }

        template <typename TReal, typename TGeom, typename TIdx>
        int tet4_boxes(const MeshView<TGeom, TIdx> &mesh,
                       const ptrdiff_t              begin,
                       const ptrdiff_t              end,
                       const double                 origin[3],
                       float *const                 lo[3],
                       float *const                 hi[3],
                       const Exec                  &exec) {
            return element_boxes<TReal, TGeom, TIdx>(mesh, begin, end, origin, lo, hi, exec);
        }

#define SRESAMPLE_INSTANTIATE_TET4_BOXES(TReal, TGeom, TIdx)                                                        \
    template int element_boxes<TReal, TGeom, TIdx>(const MeshView<TGeom, TIdx> &, const ptrdiff_t, const ptrdiff_t, \
                                                   const double[3], float *const[3], float *const[3], const Exec &); \
    template int tet4_boxes<TReal, TGeom, TIdx>(const MeshView<TGeom, TIdx> &, const ptrdiff_t, const ptrdiff_t,    \
                                                const double[3], float *const[3], float *const[3], const Exec &);

        SRESAMPLE_FOREACH_MESH_TYPES(SRESAMPLE_INSTANTIATE_TET4_BOXES)

    }  // namespace search
}  // namespace sresample
