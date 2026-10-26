#ifndef SRESAMPLE_BEAM2_HPP
#define SRESAMPLE_BEAM2_HPP

#include <math.h>

#include "sresample_base.hpp"
#include "sresample_quadrature.hpp"

namespace sresample {

    // A 2-node segment embedded in 3D, at the 12 Gauss-Legendre points: its geometry, and the
    // dual (biorthogonal) basis psi_i with int psi_i phi_j = delta_ij int phi_j.
    template <typename TReal, typename TGeom>
    struct Beam2 {
        static constexpr int nxe = 2;

        TGeom x[2], y[2], z[2];
        TReal measure;
        // The edge's own normal in the xy-plane, for points outside the grid and flat SDF cells.
        TReal face_n[3];

        SRESAMPLE_HOST_DEVICE int nqp() const { return quadrature::line12_n; }

        SRESAMPLE_HOST_DEVICE void init() {
            const TReal px0 = x[0], px1 = x[1], py0 = y[0], py1 = y[1], pz0 = z[0], pz1 = z[1];
            measure = sqrt(pow(-px0 + px1, 2) + pow(-py0 + py1, 2) + pow(-pz0 + pz1, 2));

            const TReal ux = x[1] - x[0], uy = y[1] - y[0];
            const TReal len = sqrt(ux * ux + uy * uy);
            face_n[0]       = uy / len;
            face_n[1]       = -ux / len;
            face_n[2]       = 0;
        }

        // Quadrature point q: its physical position, the dual basis there, the weight times the
        // measure, and the face normal.
        SRESAMPLE_HOST_DEVICE void qp(const int q, TReal *const p, TReal *const f, TReal *const dV, TReal *const n) const {
            const TReal t   = quadrature::line12_x(q);
            const TReal px0 = x[0], px1 = x[1], py0 = y[0], py1 = y[1], pz0 = z[0], pz1 = z[1];

            p[0] = px0 + t * (-px0 + px1);
            p[1] = py0 + t * (-py0 + py1);
            p[2] = pz0 + t * (-pz0 + pz1);

            // DUAL basis function
            const TReal f0 = 1 - t;
            const TReal f1 = t;
            f[0]           = 2 * f0 - f1;
            f[1]           = -f0 + 2 * f1;

            *dV = measure * (TReal)quadrature::line12_w(q);

            n[0] = face_n[0];
            n[1] = face_n[1];
            n[2] = face_n[2];
        }

        // The element's lumped mass, length / 2 per node.
        SRESAMPLE_HOST_DEVICE void lumped_weights(TReal *const w) const {
            const TReal px0 = x[0], px1 = x[1], py0 = y[0], py1 = y[1], pz0 = z[0], pz1 = z[1];
            const TReal x0  = (1.0 / 2.0) * sqrt(pow(px0 - px1, 2) + pow(py0 - py1, 2) + pow(pz0 - pz1, 2));
            w[0]            = x0;
            w[1]            = x0;
        }
    };

}  // namespace sresample

#endif  // SRESAMPLE_BEAM2_HPP
