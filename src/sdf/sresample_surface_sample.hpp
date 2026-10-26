#ifndef SRESAMPLE_SURFACE_SAMPLE_HPP
#define SRESAMPLE_SURFACE_SAMPLE_HPP

#include <stddef.h>

#include "sresample_base.hpp"
#include "sresample_views.hpp"

// Sampling of an SDF's cubic B-spline (sresample_bspline.hpp) on a surface mesh of shells or
// beams (TRISHELL3, QUADSHELL4, BEAM2), weighted with the elements' dual basis psi_i:
//
//   gap_i     = -int psi_i sdf / w_i,  w_i = int phi_i (the lumped mass)
//   normals_i =  int psi_i grad sdf / |grad sdf|, then normalised to unit length
//
// Outside the grid the sdf is taken as sdf::unreachable (so the gap is large and negative
// before the sign flip: the constraint stays inactive) and the normal as the element's own; where
// the gradient vanishes, the normal is the element's own too.

namespace sresample {
    namespace sdf {

        // The outputs of a surface sampling, any of which may be null (not computed). In the
        // memory space of the call.
        template <typename TReal>
        struct SurfaceOutput {
            TReal *gap{nullptr};
            TReal *normals[3]{nullptr, nullptr, nullptr};
        };

        // Adds the element integrals -int psi_i sdf (gap) and int psi_i n (normals) into the given
        // outputs: no division, no normalisation. The building block of distributed sampling,
        // whose callers sum over ranks and finish with finish_surface_sample.
        template <typename TReal, typename TGeom, typename TIdx, typename TGrid>
        int surface_sample_local(const MeshView<TGeom, TIdx>     &mesh,
                                 const BSplineGrid<TGeom, TGrid> &grid,
                                 const SurfaceOutput<TReal>      &out,
                                 const Exec                      &exec);

        // Adds the lumped mass w_i = int phi_i into w.
        template <typename TReal, typename TGeom, typename TIdx>
        int surface_weights_local(const MeshView<TGeom, TIdx> &mesh, TReal *const w, const Exec &exec);

        // gap_i /= w_i where w_i != 0, and the normals scaled to unit length where they are not zero,
        // over n entries.
        template <typename TReal>
        int finish_surface_sample(const ptrdiff_t n, const TReal *const w, const SurfaceOutput<TReal> &out, const Exec &exec);

        // The whole sampling: the outputs given are zeroed over mesh.nnodes, assembled, and
        // finished. With a gap, `workspace` holds mesh.nnodes entries for the lumped mass. An empty
        // mesh leaves the outputs as they are.
        template <typename TReal, typename TGeom, typename TIdx, typename TGrid>
        int surface_sample(const MeshView<TGeom, TIdx>     &mesh,
                           const BSplineGrid<TGeom, TGrid> &grid,
                           const SurfaceOutput<TReal>      &out,
                           TReal *const                     workspace,
                           const Exec                      &exec);

    }  // namespace sdf
}  // namespace sresample

#endif  // SRESAMPLE_SURFACE_SAMPLE_HPP
