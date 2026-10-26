#ifndef SRESAMPLE_ZSLAB_HPP
#define SRESAMPLE_ZSLAB_HPP

#include <mpi.h>
#include <stddef.h>

#include "sresample_base.hpp"

// A regular grid split into z-slabs across the ranks of a communicator: rank r holds nlocal[2]
// consecutive z-layers (in rank order), each layer stride[2] values with the whole grid's x and y
// extent.

namespace sresample {
    namespace mpi {

        // The z-layers [z_begin, z_end) of the whole grid, gathered from the ranks that hold them into
        // `out` (z_end - z_begin layers of stride[2] values, in host memory). Collective.
        template <typename TGrid>
        int zslab_gather(MPI_Comm               comm,
                         const ptrdiff_t        nlocal_z,
                         const ptrdiff_t        layer_size,
                         const TGrid *const     in,
                         const ptrdiff_t        z_begin,
                         const ptrdiff_t        z_end,
                         TGrid *const           out);

        // The z-slab of the grid that covers the z-coordinates of this rank's `nnodes` points, widened
        // by one cell and `z_margin` layers on either side (clamped to the grid, at least two layers):
        // allocated with malloc into *view (the caller frees it), its number of layers and the
        // z-coordinate of its first layer. Collective; on a single rank it fails.
        template <typename TGeom, typename TGrid>
        int zslab_view(MPI_Comm                                  comm,
                       const ptrdiff_t                           nnodes,
                       const TGeom *const SRESAMPLE_RESTRICT     z_coordinate,
                       const ptrdiff_t *const                    nlocal,
                       const ptrdiff_t *const SRESAMPLE_RESTRICT nglobal,
                       const ptrdiff_t *const SRESAMPLE_RESTRICT stride,
                       const TGeom *const                        origin,
                       const TGeom *const SRESAMPLE_RESTRICT     delta,
                       const TGrid *const                        grid,
                       const ptrdiff_t                           z_margin,
                       TGrid                                   **view,
                       ptrdiff_t                                *z_nlocal_out,
                       TGeom *const SRESAMPLE_RESTRICT           z_origin_out);

    }  // namespace mpi
}  // namespace sresample

#endif  // SRESAMPLE_ZSLAB_HPP
