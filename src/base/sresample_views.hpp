#ifndef SRESAMPLE_VIEWS_HPP
#define SRESAMPLE_VIEWS_HPP

#include "sresample_base.hpp"

namespace sresample {

    // The element types sresample samples on. Shells and beams are surfaces and curves embedded
    // in 3D; TET4 is the linear tetrahedron and HEX8 the trilinear hexahedron (Exodus node order:
    // 0-3 the bottom face counterclockwise seen from the top, 4-7 above them), the intersection
    // module's volume elements.
    enum class ElemType { TRISHELL3, QUADSHELL4, BEAM2, TET4, HEX8 };

    SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE int nxe(const ElemType type) {
        switch (type) {
            case ElemType::TRISHELL3:
                return 3;
            case ElemType::QUADSHELL4:
                return 4;
            case ElemType::BEAM2:
                return 2;
            case ElemType::TET4:
                return 4;
            case ElemType::HEX8:
                return 8;
        }
        return 0;
    }

    // A mesh as the kernels read it, structure of arrays: node v of element e is
    // elements[v][e], and its coordinates are points[d][node], d = 0, 1, 2. Nothing is owned;
    // the arrays are in the memory space of the call they are given to. ids (optional): a global
    // id per node, the same on every rank that holds the node (a distributed mesh's local node
    // indices are not); where given, the hex split (sresample_hex8_split.hpp) uses them, so that
    // neighbours on different ranks cut their common face alike.
    template <typename TGeom, typename TIdx>
    struct MeshView {
        ElemType                  type{ElemType::TRISHELL3};
        ptrdiff_t                 nelements{0};
        ptrdiff_t                 nnodes{0};
        const TIdx *const        *elements{nullptr};
        const TGeom *const       *points{nullptr};
        const long long          *ids{nullptr};
    };

    // A cubic B-spline over a regular grid, as the samplers read it: n nodes per axis from
    // `origin`, spaced `delta`, and the coefficients `coeffs` with strides `stride`, `coeffs`
    // pointing at the coefficient of node (0, 0, 0). The coefficient array is padded by one node
    // on every side (indices -1 and n are valid; see bspline_coefficients). Passed by value to
    // device kernels, so only `coeffs` must be in device memory.
    template <typename TGeom, typename TGrid>
    struct BSplineGrid {
        ptrdiff_t    n[3]{0, 0, 0};
        ptrdiff_t    stride[3]{0, 0, 0};
        TGeom        origin[3]{0, 0, 0};
        TGeom        delta[3]{0, 0, 0};
        const TGrid *coeffs{nullptr};
    };

}  // namespace sresample

#endif  // SRESAMPLE_VIEWS_HPP
