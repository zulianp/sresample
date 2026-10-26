#ifndef SRESAMPLE_NODE_SAMPLE_HPP
#define SRESAMPLE_NODE_SAMPLE_HPP

#include <stddef.h>

#include "sresample_base.hpp"
#include "sresample_views.hpp"

// The nodal sampling (sresample_node_sample_kernel.hpp) at every node, on the host or the device.

namespace sresample {
    namespace sdf {

        // The node coordinates, by axis, and where the samples go; a null output is not written.
        template <typename TGeom>
        struct NodePoints {
            const TGeom *xyz[3]{nullptr, nullptr, nullptr};
        };

        template <typename TReal>
        struct NodeOutput {
            TReal *gap{nullptr};
            TReal *normals[3]{nullptr, nullptr, nullptr};
            TReal *curvature[6]{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        };

        // At the points X[idx[i]] + disp[3 idx[i] + d] (all nnodes nodes when idx is null; disp, AoS with
        // three components, may be null), summed in real_t: the gap -sdf / |grad sdf| and the unit normal
        // grad sdf / |grad sdf| (3 rows), entry i for point i. Outside the grid the gap is
        // SFEM_SDF_UNREACHABLE and the normal zero.
        // The obstacle's curvature at the same points: (H - n m^T - m n^T) / |grad sdf|, H the Hessian of
        // the B-spline, n its unit gradient and m = H n = grad |grad sdf|, packed xx, xy, xz, yy, yz, zz
        // (6 rows). Where the point is on the surface (sdf = 0) it is minus the Hessian of the gap, so a
        // reaction mu >= 0 there adds mu times it to the Newton tangent; for an exact SDF it is the SDF's
        // Hessian, -(I - n n^T) / R on a sphere of radius R. Zero outside the grid.
        //
        // From the prefiltered coefficients, as everything else: the B-spline interpolates the samples.
        // It is O(h^2) accurate, and exact for polynomials up to degree 3 away from the grid's ends; within
        // about 8 cells of them the filter's end extension pulls it towards zero.
        // One call for all three: `order` 0 is the value alone (-sdf), 1 adds the normal, 2 gives the
        // curvature; n points, outputs in the memory space of `exec`.
        template <typename TReal, typename TGeom, typename TIdx, typename TGrid>
        int sample_nodes(const int                        order,
                         const ptrdiff_t                  n,
                         const TIdx *const                idx,
                         const NodePoints<TGeom>         &points,
                         const TReal *const               disp,
                         const BSplineGrid<TGeom, TGrid> &grid,
                         const NodeOutput<TReal>         &out,
                         const Exec                      &exec);

    }  // namespace sdf
}  // namespace sresample

#endif  // SRESAMPLE_NODE_SAMPLE_HPP
