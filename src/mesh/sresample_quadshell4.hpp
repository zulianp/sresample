#ifndef SRESAMPLE_QUADSHELL4_HPP
#define SRESAMPLE_QUADSHELL4_HPP

#include <math.h>

#include "sresample_base.hpp"
#include "sresample_quadrature.hpp"

namespace sresample {

    // A bilinear 4-node quadrilateral embedded in 3D, at the 6 x 6 Gauss-Legendre tensor points:
    // its geometry, and the dual (biorthogonal) basis psi_i with int psi_i phi_j = delta_ij int phi_j.
    //
    // The gap is weighted with the dual basis, as on TRISHELL3 and BEAM2: then
    // int psi_i phi_j = delta_ij m_j, the weighted gap's linearisation is the nodal n . u the
    // constraint operator applies, and re-linearising at the solution converges at once instead of
    // like I - M_l^-1 M_c.
    template <typename TReal, typename TGeom>
    struct Quadshell4 {
        static constexpr int nxe = 4;

        TGeom x[4], y[4], z[4];

        SRESAMPLE_HOST_DEVICE int nqp() const { return quadrature::line6_n * quadrature::line6_n; }

        SRESAMPLE_HOST_DEVICE void init() {}

        // The surface measure at (qx, qy) and the unit face normal n.
        SRESAMPLE_HOST_DEVICE TReal measure_and_normal(const TReal qx,
                                                       const TReal qy,
                                                       // Output unit face normal (mesh winding)
                                                       TReal *const n) const {
            const TReal px0 = x[0], px1 = x[1], px2 = x[2], px3 = x[3];
            const TReal py0 = y[0], py1 = y[1], py2 = y[2], py3 = y[3];
            const TReal pz0 = z[0], pz1 = z[1], pz2 = z[2], pz3 = z[3];

            const TReal x0 = qx - 1;
            const TReal x1 = -x0;
            // r_xi = (x2, x6, x8), r_eta = (x5, x7, x9)
            const TReal x2 = px0 * x0 - px1 * qx + px2 * qx + px3 * x1;
            const TReal x3 = qy - 1;
            const TReal x4 = -x3;
            const TReal x5 = px0 * x3 + px1 * x4 + px2 * qy - px3 * qy;
            const TReal x6 = py0 * x0 - py1 * qx + py2 * qx + py3 * x1;
            const TReal x7 = py0 * x3 + py1 * x4 + py2 * qy - py3 * qy;
            const TReal x8 = pz0 * x0 - pz1 * qx + pz2 * qx + pz3 * x1;
            const TReal x9 = pz0 * x3 + pz1 * x4 + pz2 * qy - pz3 * qy;

            const TReal cx      = x6 * x9 - x8 * x7;
            const TReal cy      = x8 * x5 - x2 * x9;
            const TReal cz      = x2 * x7 - x6 * x5;
            const TReal measure = sqrt(cx * cx + cy * cy + cz * cz);

            if (measure > TReal(1e-12)) {
                n[0] = cx / measure;
                n[1] = cy / measure;
                n[2] = cz / measure;
            } else {
                // Degenerate / collapsed quad at this QP
                n[0] = 1;
                n[1] = 0;
                n[2] = 0;
            }
            return measure;
        }

        // Quadrature point q = q_ix * 6 + q_iy: its physical position, the dual basis there, the
        // weight times the measure, and the face normal.
        SRESAMPLE_HOST_DEVICE void qp(const int q, TReal *const p, TReal *const f, TReal *const dV, TReal *const n) const {
            const int   q_ix = q / quadrature::line6_n;
            const int   q_iy = q % quadrature::line6_n;
            const TReal qx   = quadrature::line6_x(q_ix);
            const TReal qy   = quadrature::line6_x(q_iy);

            const TReal measure = measure_and_normal(qx, qy, n);

            {
                const TReal x0 = qx * qy;
                const TReal x1 = 1 - qy;
                const TReal x2 = qx * x1;
                const TReal x3 = 1 - qx;
                const TReal x4 = qy * x3;
                const TReal x5 = x1 * x3;

                p[0] = (TReal)x[0] * x5 + (TReal)x[1] * x2 + (TReal)x[2] * x0 + (TReal)x[3] * x4;
                p[1] = (TReal)y[0] * x5 + (TReal)y[1] * x2 + (TReal)y[2] * x0 + (TReal)y[3] * x4;
                p[2] = (TReal)z[0] * x5 + (TReal)z[1] * x2 + (TReal)z[2] * x0 + (TReal)z[3] * x4;
            }

            // DUAL basis function
            {
                const TReal x0 = 1 - qx;
                const TReal x1 = 1 - qy;
                const TReal f0 = x0 * x1;
                const TReal f1 = qx * x1;
                const TReal f2 = qx * qy;
                const TReal f3 = qy * x0;

                f[0] = 4 * f0 - 2 * f1 + f2 - 2 * f3;
                f[1] = -2 * f0 + 4 * f1 - 2 * f2 + f3;
                f[2] = f0 - 2 * f1 + 4 * f2 - 2 * f3;
                f[3] = -2 * f0 + f1 - 2 * f2 + 4 * f3;
            }

            *dV = measure * (TReal)quadrature::line6_w(q_ix) * (TReal)quadrature::line6_w(q_iy);
        }

        // The element's lumped mass: the integral of each dual basis function, which is the
        // integral of the nodal one.
        SRESAMPLE_HOST_DEVICE void lumped_weights(TReal *const w) const {
            for (int v = 0; v < 4; ++v) w[v] = 0;
            for (int q = 0; q < nqp(); ++q) {
                TReal p[3], f[4], dV, n[3];
                qp(q, p, f, &dV, n);
                for (int v = 0; v < 4; ++v) w[v] += f[v] * dV;
            }
        }
    };

}  // namespace sresample

#endif  // SRESAMPLE_QUADSHELL4_HPP
