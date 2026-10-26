#ifndef SRESAMPLE_TRISHELL3_HPP
#define SRESAMPLE_TRISHELL3_HPP

#include <math.h>

#include "sresample_base.hpp"
#include "sresample_quadrature.hpp"

namespace sresample {

    // A 3-node triangle embedded in 3D, at the points of the 12-point degree-6 triangle rule: its
    // geometry, and the dual (biorthogonal) basis psi_i with int psi_i phi_j = delta_ij int phi_j.
    template <typename TReal, typename TGeom>
    struct Trishell3 {
        static constexpr int nxe = 3;

        TGeom x[3], y[3], z[3];
        TReal measure;
        // The triangle's own normal, for points outside the grid and flat SDF cells.
        TReal face_n[3];

        SRESAMPLE_HOST_DEVICE int nqp() const { return quadrature::tri12_n; }

        SRESAMPLE_HOST_DEVICE void init() {
            const TReal x0 = -(TReal)x[0] + (TReal)x[1];
            const TReal x1 = -(TReal)x[0] + (TReal)x[2];
            const TReal x2 = -(TReal)y[0] + (TReal)y[1];
            const TReal x3 = -(TReal)y[0] + (TReal)y[2];
            const TReal x4 = -(TReal)z[0] + (TReal)z[1];
            const TReal x5 = -(TReal)z[0] + (TReal)z[2];
            measure        = (1.0 / 2.0) * sqrt((pow(x0, 2) + pow(x2, 2) + pow(x4, 2)) * (pow(x1, 2) + pow(x3, 2) + pow(x5, 2)) -
                                         pow(x0 * x1 + x2 * x3 + x4 * x5, 2));

            const TReal ux = x[1] - x[0], uy = y[1] - y[0], uz = z[1] - z[0];
            const TReal vx = x[2] - x[0], vy = y[2] - y[0], vz = z[2] - z[0];
            face_n[0]      = uy * vz - uz * vy;
            face_n[1]      = uz * vx - ux * vz;
            face_n[2]      = ux * vy - uy * vx;
            const TReal len = sqrt(face_n[0] * face_n[0] + face_n[1] * face_n[1] + face_n[2] * face_n[2]);
            face_n[0] /= len;
            face_n[1] /= len;
            face_n[2] /= len;
        }

        // Quadrature point q: its physical position, the dual basis there, the weight times the
        // measure, and the face normal.
        SRESAMPLE_HOST_DEVICE void qp(const int q, TReal *const p, TReal *const f, TReal *const dV, TReal *const n) const {
            const TReal qx = quadrature::tri12_x(q);
            const TReal qy = quadrature::tri12_y(q);

            const TReal phi0 = 1 - qx - qy;
            const TReal phi1 = qx;
            const TReal phi2 = qy;

            p[0] = phi0 * (TReal)x[0] + phi1 * (TReal)x[1] + phi2 * (TReal)x[2];
            p[1] = phi0 * (TReal)y[0] + phi1 * (TReal)y[1] + phi2 * (TReal)y[2];
            p[2] = phi0 * (TReal)z[0] + phi1 * (TReal)z[1] + phi2 * (TReal)z[2];

            // DUAL basis function
            const TReal f0 = 1 - qx - qy;
            const TReal f1 = qx;
            const TReal f2 = qy;

            f[0] = 3 * f0 - f1 - f2;
            f[1] = -f0 + 3 * f1 - f2;
            f[2] = -f0 - f1 + 3 * f2;

            *dV = measure * (TReal)quadrature::tri12_w(q);

            n[0] = face_n[0];
            n[1] = face_n[1];
            n[2] = face_n[2];
        }

        // The element's lumped mass, area / 3 per node.
        SRESAMPLE_HOST_DEVICE void lumped_weights(TReal *const w) const {
            const TReal x0 = -(TReal)x[0] + (TReal)x[1];
            const TReal x1 = -(TReal)x[0] + (TReal)x[2];
            const TReal x2 = -(TReal)y[0] + (TReal)y[1];
            const TReal x3 = -(TReal)y[0] + (TReal)y[2];
            const TReal x4 = -(TReal)z[0] + (TReal)z[1];
            const TReal x5 = -(TReal)z[0] + (TReal)z[2];
            const TReal x6 = (1.0 / 6.0) * sqrt((x0 * x0 + x2 * x2 + x4 * x4) * (x1 * x1 + x3 * x3 + x5 * x5) -
                                                (x0 * x1 + x2 * x3 + x4 * x5) * (x0 * x1 + x2 * x3 + x4 * x5));
            w[0]           = x6;
            w[1]           = x6;
            w[2]           = x6;
        }
    };

}  // namespace sresample

#endif  // SRESAMPLE_TRISHELL3_HPP
