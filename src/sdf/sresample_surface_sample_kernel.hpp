#ifndef SRESAMPLE_SURFACE_SAMPLE_KERNEL_HPP
#define SRESAMPLE_SURFACE_SAMPLE_KERNEL_HPP

#include <math.h>

#include "sresample_atomic.hpp"
#include "sresample_beam2.hpp"
#include "sresample_bspline.hpp"
#include "sresample_quadshell4.hpp"
#include "sresample_surface_sample.hpp"
#include "sresample_trishell3.hpp"

// One element of the surface sampling (sresample_surface_sample.hpp), for the host loop and the
// device kernel alike: Elem is Trishell3, Quadshell4 or Beam2.

namespace sresample {
    namespace sdf {

        template <class Elem, typename TGeom, typename TIdx>
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void load_element(const ptrdiff_t                e,
                                                                 const MeshView<TGeom, TIdx>   &mesh,
                                                                 Elem                          &el,
                                                                 TIdx *const SRESAMPLE_RESTRICT ev) {
            for (int v = 0; v < Elem::nxe; ++v) {
                ev[v] = mesh.elements[v][e];
            }

            for (int v = 0; v < Elem::nxe; ++v) {
                el.x[v] = mesh.points[0][ev[v]];
                el.y[v] = mesh.points[1][ev[v]];
                el.z[v] = mesh.points[2][ev[v]];
            }

            el.init();
        }

        template <class Elem, typename TReal, typename TGeom, typename TIdx, typename TGrid>
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void surface_sample_element(const ptrdiff_t                  e,
                                                                           const MeshView<TGeom, TIdx>     &mesh,
                                                                           const BSplineGrid<TGeom, TGrid> &grid,
                                                                           const SurfaceOutput<TReal>      &out) {
            constexpr int nxe   = Elem::nxe;
            const TReal   infty = unreachable;
            const TReal   eps   = 1e-12;

            Elem el;
            TIdx ev[nxe];
            load_element(e, mesh, el, ev);

            const bool with_gap     = out.gap != nullptr;
            const bool with_normals = out.normals[0] != nullptr;

            TReal element_gap[nxe];
            TReal element_normal[3][nxe];
            for (int i = 0; i < nxe; ++i) {
                element_gap[i]       = 0;
                element_normal[0][i] = 0;
                element_normal[1][i] = 0;
                element_normal[2][i] = 0;
            }

            const int nqp = el.nqp();
            for (int q = 0; q < nqp; ++q) {
                TReal p[3], f[nxe], dV, face_n[3];
                el.qp(q, p, f, &dV, face_n);

                TReal value, grad[3];
                // If outside
                const bool inside = with_normals ? sample<1>(grid, p[0], p[1], p[2], &value, grad, (TReal *)nullptr)
                                                 : sample<0>(grid, p[0], p[1], p[2], &value, grad, (TReal *)nullptr);

                if (!inside) {
                    // If outside grid: inject -infty before wg -= so stored gap becomes +infty.
                    // sample_value() feeds ShiftedPenalty upper bounds (n·u ≤ ub); +infty keeps the
                    // constraint inactive. (gap_local uses +infty here instead → stored -infty for
                    // signed-distance / viz semantics.)
                    for (int i = 0; i < nxe; i++) {
                        if (with_gap) element_gap[i] += -infty * f[i] * dV;
                        if (with_normals) {
                            element_normal[0][i] += face_n[0] * f[i] * dV;
                            element_normal[1][i] += face_n[1] * f[i] * dV;
                            element_normal[2][i] += face_n[2] * f[i] * dV;
                        }
                    }
                    continue;
                }

                // Integrate gap function
                if (with_gap) {
                    for (int i = 0; i < nxe; i++) {
                        element_gap[i] += value * f[i] * dV;
                    }
                }

                if (with_normals) {
                    TReal n[3] = {grad[0], grad[1], grad[2]};

                    const TReal denom = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);

                    if (denom >= eps) {
                        n[0] /= denom;
                        n[1] /= denom;
                        n[2] /= denom;
                    } else {
                        // Flat SDF cell: fall back to geometric face normal
                        n[0] = face_n[0];
                        n[1] = face_n[1];
                        n[2] = face_n[2];
                    }

                    for (int i = 0; i < nxe; i++) {
                        element_normal[0][i] += n[0] * f[i] * dV;
                        element_normal[1][i] += n[1] * f[i] * dV;
                        element_normal[2][i] += n[2] * f[i] * dV;
                    }
                }
            }

            for (int v = 0; v < nxe; ++v) {
                // Invert sign since distance field is negative inside and positive outside
                if (with_gap) atomic_add(&out.gap[ev[v]], -element_gap[v]);
                if (with_normals) {
                    atomic_add(&out.normals[0][ev[v]], element_normal[0][v]);
                    atomic_add(&out.normals[1][ev[v]], element_normal[1][v]);
                    atomic_add(&out.normals[2][ev[v]], element_normal[2][v]);
                }
            }
        }

        template <class Elem, typename TReal, typename TGeom, typename TIdx>
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void surface_weights_element(const ptrdiff_t                e,
                                                                            const MeshView<TGeom, TIdx>   &mesh,
                                                                            TReal *const SRESAMPLE_RESTRICT w) {
            Elem el;
            TIdx ev[Elem::nxe];
            load_element(e, mesh, el, ev);

            TReal element_w[Elem::nxe];
            el.lumped_weights(element_w);

            for (int v = 0; v < Elem::nxe; ++v) {
                atomic_add(&w[ev[v]], element_w[v]);
            }
        }

        // The element types' kernels, by the mesh's type: f.template operator()<Elem>().
        template <typename TReal, typename TGeom, class F>
        int dispatch_shell(const ElemType type, F &&f) {
            switch (type) {
                case ElemType::TRISHELL3:
                    f.template operator()<Trishell3<TReal, TGeom>>();
                    return SRESAMPLE_SUCCESS;
                case ElemType::QUADSHELL4:
                    f.template operator()<Quadshell4<TReal, TGeom>>();
                    return SRESAMPLE_SUCCESS;
                case ElemType::BEAM2:
                    f.template operator()<Beam2<TReal, TGeom>>();
                    return SRESAMPLE_SUCCESS;
            }
            SRESAMPLE_LOG(0, "surface sampling: not a shell or beam element type (%d)\n", (int)type);
            return SRESAMPLE_FAILURE;
        }

    }  // namespace sdf
}  // namespace sresample

#endif  // SRESAMPLE_SURFACE_SAMPLE_KERNEL_HPP
