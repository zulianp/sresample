#ifndef SRESAMPLE_CELL_LIST_CUDA_HPP
#define SRESAMPLE_CELL_LIST_CUDA_HPP

#include "sresample_cell_list.hpp"

// The cell list on the device (sresample_cell_list.cu): the same results as the host's, built with
// CUB (scan, radix sort, scan by key) on `stream`. For a MeshView the arrays of pointers
// (elements, points) must be device memory too.

namespace sresample {
    namespace search {
        namespace cuda {

            int cell_list_build(const Boxes &boxes, const CellListOptions &options, CellList *list, void *const stream);
            int cell_list_destroy(CellList *list);
            int cell_list_count(const CellList &list, const Boxes &queries, ptrdiff_t *count, void *const stream);
            int cell_list_fill(const CellList  &list,
                               const Boxes     &queries,
                               const ptrdiff_t *ptr,
                               int             *pair_query,
                               int             *pair_box,
                               void *const      stream);
            int exclusive_scan(ptrdiff_t *a, const ptrdiff_t n, ptrdiff_t *total, void *const stream);

            template <typename TReal, typename TGeom, typename TIdx>
            int element_boxes(const MeshView<TGeom, TIdx> &mesh,
                              const ptrdiff_t              begin,
                              const ptrdiff_t              end,
                              const double                 origin[3],
                              float *const                 lo[3],
                              float *const                 hi[3],
                              void *const                  stream);

        }  // namespace cuda
    }  // namespace search
}  // namespace sresample

#endif  // SRESAMPLE_CELL_LIST_CUDA_HPP
