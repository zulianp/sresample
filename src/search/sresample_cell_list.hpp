#ifndef SRESAMPLE_CELL_LIST_HPP
#define SRESAMPLE_CELL_LIST_HPP

#include <stddef.h>

#include "sresample_base.hpp"
#include "sresample_views.hpp"

// A cell list of axis-aligned boxes for box-against-box queries (the broadphase of a mesh-mesh
// intersection), on the host or the device. It is legacy/grid_to_mesh/cell_list's 3D-2D map:
//
//   - a grid of columns in x and y; every box is entered in each column its xy extent touches;
//   - the entries of a column sorted by their lower z, with the running maximum of their upper z,
//     so that the entries a z range can touch are one interval, found by two binary searches;
//   - several levels by box size (the legacy "split" map, generalised): the boxes are classed by
//     log2 of their xy extent, and every `sizes_per_level` classes form a level, with columns as
//     wide as its largest box, so that a few large boxes do not coarsen the columns of the many
//     small ones. A box is entered in at most 2 x 2 columns of its level.
//
// A query reports the pairs (query box, box) whose closed boxes overlap, each exactly once: a pair
// found in several columns counts only in the column of the lower corner of the two boxes'
// intersection, (max of the lower x, max of the lower y).
//
// Boxes are float (the broadphase needs no more than conservative bounds: tet4_boxes rounds them
// outward, relative to an origin near the data).

namespace sresample {
    namespace search {

        static constexpr int cell_list_max_levels = 4;

        // Boxes, structure of arrays: box b is [lo[d][b], hi[d][b]], d = 0, 1, 2. A view.
        struct Boxes {
            ptrdiff_t    n{0};
            const float *lo[3]{nullptr, nullptr, nullptr};
            const float *hi[3]{nullptr, nullptr, nullptr};
        };

        // One level: n[0] x n[1] columns of width 1 / inv_h from origin. The entries of column
        // c = cy n[0] + cx are [col_ptr[c], col_ptr[c + 1]).
        struct CellListLevel {
            float      origin[2]{0, 0};
            float      inv_h[2]{0, 0};
            int        n[2]{0, 0};
            ptrdiff_t  n_entries{0};
            ptrdiff_t *col_ptr{nullptr};
            int       *entry_box{nullptr};
            float     *entry_zlo{nullptr};      // ascending within a column
            float     *entry_zhi_max{nullptr};  // running maximum of the upper z within a column
        };

        // The cell list owns its arrays (in `space`), including a copy of the boxes.
        struct CellList {
            MemorySpace   space{MemorySpace::HOST};
            int           n_levels{0};
            CellListLevel level[cell_list_max_levels];
            ptrdiff_t     n_boxes{0};
            float        *box_lo[3]{nullptr, nullptr, nullptr};
            float        *box_hi[3]{nullptr, nullptr, nullptr};
        };

        struct CellListOptions {
            int       max_levels{cell_list_max_levels};
            int       sizes_per_level{3};       // log2 classes per level (a size range of 8)
            ptrdiff_t max_columns{1 << 24};     // per level; wider columns beyond
        };

        // Builds the list over `boxes` (in the memory space of exec). Destroy it with
        // cell_list_destroy.
        int cell_list_build(const Boxes &boxes, const CellListOptions &options, CellList *list, const Exec &exec);
        int cell_list_destroy(CellList *list);

        // Two passes over the query boxes: count[q] = the number of boxes that overlap query q,
        // q < queries.n; then, with ptr the exclusive scan of the counts (queries.n + 1 entries),
        // the pairs of query q in [ptr[q], ptr[q + 1]): pair_query[k] = q, pair_box[k] = the box.
        // Within a query the order follows the levels and columns. Arrays in the list's space.
        int cell_list_count(const CellList &list, const Boxes &queries, ptrdiff_t *count, const Exec &exec);
        int cell_list_fill(const CellList &list,
                           const Boxes    &queries,
                           const ptrdiff_t *ptr,
                           int            *pair_query,
                           int            *pair_box,
                           const Exec     &exec);

        // a[0..n] (n + 1 entries) replaced by its exclusive scan; *total = the sum of a[0..n-1].
        int exclusive_scan(ptrdiff_t *a, const ptrdiff_t n, ptrdiff_t *total, const Exec &exec);

        // The boxes of a TET4 or HEX8 mesh's elements [begin, end), relative to `origin` and rounded
        // outward: lo[d][e - begin], hi[d][e - begin]. Arrays in the memory space of exec.
        // (TReal only distinguishes the instantiations.)
        template <typename TReal, typename TGeom, typename TIdx>
        int element_boxes(const MeshView<TGeom, TIdx> &mesh,
                          const ptrdiff_t              begin,
                          const ptrdiff_t              end,
                          const double                 origin[3],
                          float *const                 lo[3],
                          float *const                 hi[3],
                          const Exec                  &exec);

        // The same for a TET4 mesh (element_boxes).
        template <typename TReal, typename TGeom, typename TIdx>
        int tet4_boxes(const MeshView<TGeom, TIdx> &mesh,
                       const ptrdiff_t              begin,
                       const ptrdiff_t              end,
                       const double                 origin[3],
                       float *const                 lo[3],
                       float *const                 hi[3],
                       const Exec                  &exec);

    }  // namespace search
}  // namespace sresample

#endif  // SRESAMPLE_CELL_LIST_HPP
