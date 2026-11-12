#ifndef SRESAMPLE_CELL_LIST_KERNEL_HPP
#define SRESAMPLE_CELL_LIST_KERNEL_HPP

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "sresample_cell_list.hpp"

// The cell list's per-box and per-query work, for the host loops (sresample_cell_list.cpp) and
// the device kernels (cuda/sresample_cell_list.cu) alike.

namespace sresample {
    namespace search {

        // The column index of x along axis a of a level, clamped to the grid: the same function
        // for entering boxes and for the queries' deduplication.
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE int column_of(const CellListLevel &l, const int a, const float x) {
            int c = (int)floorf((x - l.origin[a]) * l.inv_h[a]);
            return c < 0 ? 0 : (c >= l.n[a] ? l.n[a] - 1 : c);
        }

        // An order-preserving map of floats to unsigned integers (the radix sort's key).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE uint32_t float_key(const float f) {
            uint32_t u;
            memcpy(&u, &f, sizeof(u));
            return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        }

        // The size class of a box: floor(log2) of its larger xy extent, offset to [0, 255]; 0 for
        // degenerate boxes.
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE int box_size_class(const float dx, const float dy) {
            const float e = dx > dy ? dx : dy;
            if (!(e > 0)) return 0;
            int c = (int)floorf(log2f(e)) + 128;
            return c < 0 ? 0 : (c > 255 ? 255 : c);
        }

        // Do the closed boxes overlap?
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE bool boxes_overlap(const float qlo[3], const float qhi[3], const float blo[3], const float bhi[3]) {
            return qlo[0] <= bhi[0] && blo[0] <= qhi[0] && qlo[1] <= bhi[1] && blo[1] <= qhi[1] && qlo[2] <= bhi[2] &&
                   blo[2] <= qhi[2];
        }

        // The first index in [first, last) with v[k] > x (v ascending).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE ptrdiff_t upper_bound_f(const float *const v, ptrdiff_t first, ptrdiff_t last, const float x) {
            while (first < last) {
                const ptrdiff_t mid = first + (last - first) / 2;
                if (v[mid] > x) {
                    last = mid;
                } else {
                    first = mid + 1;
                }
            }
            return first;
        }

        // The first index in [first, last) with v[k] >= x (v ascending).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE ptrdiff_t lower_bound_f(const float *const v, ptrdiff_t first, ptrdiff_t last, const float x) {
            while (first < last) {
                const ptrdiff_t mid = first + (last - first) / 2;
                if (v[mid] < x) {
                    first = mid + 1;
                } else {
                    last = mid;
                }
            }
            return first;
        }

        // Every box of the list that overlaps query box q, once: visit(box). The list's arrays must
        // be readable where this runs.
        template <class Visit>
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void cell_list_visit(const CellList &list,
                                                                   const float     qlo[3],
                                                                   const float     qhi[3],
                                                                   Visit         &&visit) {
            for (int lv = 0; lv < list.n_levels; ++lv) {
                const CellListLevel &l = list.level[lv];
                if (!l.n_entries) continue;
                const int cx0 = column_of(l, 0, qlo[0]), cx1 = column_of(l, 0, qhi[0]);
                const int cy0 = column_of(l, 1, qlo[1]), cy1 = column_of(l, 1, qhi[1]);
                for (int cy = cy0; cy <= cy1; ++cy) {
                    for (int cx = cx0; cx <= cx1; ++cx) {
                        const ptrdiff_t c     = (ptrdiff_t)cy * l.n[0] + cx;
                        const ptrdiff_t begin = l.col_ptr[c], end = l.col_ptr[c + 1];
                        if (begin == end) continue;
                        // Entries with zlo <= qhi_z, then of those zhi >= qlo_z: one interval.
                        const ptrdiff_t last  = upper_bound_f(l.entry_zlo, begin, end, qhi[2]);
                        const ptrdiff_t first = lower_bound_f(l.entry_zhi_max, begin, last, qlo[2]);
                        for (ptrdiff_t k = first; k < last; ++k) {
                            const int   b      = l.entry_box[k];
                            const float blo[3] = {list.box_lo[0][b], list.box_lo[1][b], list.box_lo[2][b]};
                            const float bhi[3] = {list.box_hi[0][b], list.box_hi[1][b], list.box_hi[2][b]};
                            if (!boxes_overlap(qlo, qhi, blo, bhi)) continue;
                            // Once: in the column of the intersection's lower corner.
                            const float px = qlo[0] > blo[0] ? qlo[0] : blo[0];
                            const float py = qlo[1] > blo[1] ? qlo[1] : blo[1];
                            if (column_of(l, 0, px) != cx || column_of(l, 1, py) != cy) continue;
                            visit(b);
                        }
                    }
                }
            }
        }

        // The box of element e (its nxe(mesh.type) nodes: TET4 or HEX8, whose trilinear map stays in
        // its vertices' convex hull) relative to origin, rounded outward to float.
        template <typename TGeom, typename TIdx>
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void element_box(const MeshView<TGeom, TIdx> &mesh,
                                                                const ptrdiff_t              e,
                                                                const double                 origin[3],
                                                                float                        lo[3],
                                                                float                        hi[3]) {
            const int n = nxe(mesh.type);
            for (int d = 0; d < 3; ++d) {
                double mn = 1e300, mx = -1e300;
                for (int v = 0; v < n; ++v) {
                    const double x = (double)mesh.points[d][mesh.elements[v][e]] - origin[d];
                    mn             = x < mn ? x : mn;
                    mx             = x > mx ? x : mx;
                }
                float fl = (float)mn, fh = (float)mx;
                if ((double)fl > mn) fl = nextafterf(fl, -INFINITY);
                if ((double)fh < mx) fh = nextafterf(fh, INFINITY);
                lo[d] = fl;
                hi[d] = fh;
            }
        }

        // The box of TET4 element e (element_box).
        template <typename TGeom, typename TIdx>
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void tet4_box(const MeshView<TGeom, TIdx> &mesh,
                                                             const ptrdiff_t              e,
                                                             const double                 origin[3],
                                                             float                        lo[3],
                                                             float                        hi[3]) {
            element_box(mesh, e, origin, lo, hi);
        }

    }  // namespace search
}  // namespace sresample

#endif  // SRESAMPLE_CELL_LIST_KERNEL_HPP
