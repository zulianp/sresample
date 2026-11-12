#ifndef SRESAMPLE_HEX8_INTERSECT_KERNEL_HPP
#define SRESAMPLE_HEX8_INTERSECT_KERNEL_HPP

#include <math.h>

#include "sresample_base.hpp"
#include "sresample_hex8_split.hpp"
#include "sresample_hex8_transfer.hpp"
#include "sresample_quadrature.hpp"
#include "sresample_tet4_intersect_kernel.hpp"

// The intersection of a hexahedron (HEX8, trilinear; the "solid") with a tetrahedron (TET4; the
// "fluid"): the mixed mass matrix
//
//   B_ia = int_{H_s cap T_f} phi_i psi_a dx,
//
// phi_i the fluid tet's P1 basis, psi_a the hex's basis, for the host loop and the device kernel
// alike (everything by value or on the stack).
//
// The hex's region is the union of its split tets (sresample_hex8_split.hpp: conforming across the
// mesh, so the solid's tets tile it without gaps or overlaps, the faces of a warped hex as two
// planar triangles). Each split tet T_k is intersected with the fluid tet as in
// sresample_tet4_intersect_kernel.hpp: separated, the fluid tet inside T_k, T_k inside the fluid
// tet, or the fluid tet clipped by T_k's faces (r3d). Then, by the basis:
//
//   Hex8Basis::TRILINEAR  psi_a the hex's trilinear basis, at the reference point s(x) of the
//                         trilinear map (Newton, from T_k's barycentric coordinates mapped to the
//                         reference cube's corners); the region (the fluid tet, T_k, or the clipped
//                         polyhedron fanned into tets) integrated by the 14-point rule of degree 5
//                         per tet. Exact for a parallelepiped hex (psi_a then a cubic in x).
//   Hex8Basis::SPLIT      psi_a the P1 basis of T_k on its four hex nodes (tet4_mixed_mass_pair,
//                         in closed form): the hex as its split tets.
//
// Either way sum_a psi_a = 1, so sum_a B_ia = int phi_i over the same region, to round-off: the
// coupled volume is the split tets', in both bases.

namespace sresample {
    namespace intersect {

        enum class Hex8Basis { TRILINEAR = 0, SPLIT = 1 };

        // The reference cube [0, 1]^3's corner a (Exodus order).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE double hex8_corner(const int a, const int d) {
            constexpr unsigned char c[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
            return (double)c[a][d];
        }

        // The trilinear basis at s in [0, 1]^3: N[a], and dN[a][d] = dN_a / ds_d if dN is given.
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void hex8_basis(const double s[3], double N[8], double dN[8][3]) {
            for (int a = 0; a < 8; ++a) {
                double f[3], g[3];
                for (int d = 0; d < 3; ++d) {
                    const bool one = hex8_corner(a, d) > 0.5;
                    f[d]           = one ? s[d] : 1 - s[d];
                    g[d]           = one ? 1 : -1;
                }
                N[a] = f[0] * f[1] * f[2];
                if (dN) {
                    dN[a][0] = g[0] * f[1] * f[2];
                    dN[a][1] = f[0] * g[1] * f[2];
                    dN[a][2] = f[0] * f[1] * g[2];
                }
            }
        }

        // s with x(s) = sum_a N_a(s) xs_a = x, by Newton from the s given; the trilinear basis there
        // into N. In coordinates relative to the hex's node 0, so that the residual's round-off is the
        // hex's size times epsilon, not the coordinates' (a hex of 0.1 mm at 1 cm from the origin
        // would otherwise stall Newton's step at ~1e-14). Converged when the step is below 1e-13.
        // Returns the iterations taken (hex8_newton_max_it, sresample_hex8_transfer.hpp: not
        // converged).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE int hex8_inverse(const double xs[8][3], const double x[3], double s[3], double N[8]) {
            double dN[8][3], xr[8][3], target[3];
            for (int a = 0; a < 8; ++a) {
                for (int d = 0; d < 3; ++d) xr[a][d] = xs[a][d] - xs[0][d];
            }
            for (int d = 0; d < 3; ++d) target[d] = x[d] - xs[0][d];
            int it = 0;
            for (; it < hex8_newton_max_it; ++it) {
                hex8_basis(s, N, dN);
                double r[3] = {-target[0], -target[1], -target[2]}, c[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
                for (int a = 0; a < 8; ++a) {
                    for (int d = 0; d < 3; ++d) {
                        r[d] += N[a] * xr[a][d];
                        // Column e of the Jacobian: dx / ds_e.
                        for (int e = 0; e < 3; ++e) c[e][d] += dN[a][e] * xr[a][d];
                    }
                }
                double inv[3][3];
                tet4_inverse3(c[0], c[1], c[2], inv);
                double step = 0;
                for (int e = 0; e < 3; ++e) {
                    const double ds = inv[e][0] * r[0] + inv[e][1] * r[1] + inv[e][2] * r[2];
                    s[e] -= ds;
                    step = fabs(ds) > step ? fabs(ds) : step;
                }
                if (step < 1e-13) break;
            }
            hex8_basis(s, N, nullptr);
            return it;
        }

        // The integration of one region of a split tet T (its hex corners corner[4]): B[i][a] +=
        // w phi_i psi_a at the points the callers give, each with T's barycentric coordinates lt
        // (Newton's start) and the fluid's phi.
        struct Hex8Accumulator {
            const double (*xs)[3];
            const int *corner;
            double (*B)[8];
            int max_it{0};

            SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void point(const double x[3], const double lt[4], const double phi[4], const double w) {
                double s[3] = {0, 0, 0}, N[8];
                for (int v = 0; v < 4; ++v) {
                    for (int d = 0; d < 3; ++d) s[d] += lt[v] * hex8_corner(corner[v], d);
                }
                const int it = hex8_inverse(xs, x, s, N);
                max_it       = it > max_it ? it : max_it;
                for (int i = 0; i < 4; ++i) {
                    const double wp = w * phi[i];
                    for (int a = 0; a < 8; ++a) B[i][a] += wp * N[a];
                }
            }
        };

        // The fan of a clipped polyhedron (in T's reference coordinates relative to c, y = xi - c)
        // through the 14-point rule: x = xt0 + J (y + c), dx = |det J| dy. The fan's apex is the
        // polyhedron's vertex average p0, inside it (the polyhedron is convex): every tet
        // (p0, a, b, c) and every point is in the region, where the trilinear inverse is defined.
        // (tet4_polyhedron_fan fans to the origin, here the fluid tet's centroid, often outside the
        // region: exact for polynomials with signed volumes, but its points could leave the hex.)
        struct Hex8FanVisit {
            Hex8Accumulator *acc;
            const double (*xt)[3];
            const double (*Jt)[3];      // rows: the edges xt_k - xt_0
            const double (*Jfinv)[3];   // the fluid's inverse map
            const double *xf0;
            const double *c;
            const double *p0;
            double        det;

            SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void operator()(const double *a, const double *b, const double *cc) {
                double ea[3], eb[3], ec[3];
                for (int d = 0; d < 3; ++d) {
                    ea[d] = a[d] - p0[d];
                    eb[d] = b[d] - p0[d];
                    ec[d] = cc[d] - p0[d];
                }
                const double V = tet4_det3(ea, eb, ec) / 6;
                if (V == 0) return;
                for (int q = 0; q < quadrature::tet14_n; ++q) {
                    double l[4];
                    quadrature::tet14_l(q, l);
                    double xi[3], x[3], lt[4], phi[4];
                    for (int d = 0; d < 3; ++d) xi[d] = p0[d] + l[1] * ea[d] + l[2] * eb[d] + l[3] * ec[d] + c[d];
                    for (int d = 0; d < 3; ++d) x[d] = xt[0][d] + xi[0] * Jt[0][d] + xi[1] * Jt[1][d] + xi[2] * Jt[2][d];
                    lt[0] = 1 - xi[0] - xi[1] - xi[2];
                    lt[1] = xi[0];
                    lt[2] = xi[1];
                    lt[3] = xi[2];
                    double r[3];
                    for (int d = 0; d < 3; ++d) r[d] = x[d] - xf0[d];
                    for (int k = 0; k < 3; ++k) phi[k + 1] = Jfinv[k][0] * r[0] + Jfinv[k][1] * r[1] + Jfinv[k][2] * r[2];
                    phi[0] = 1 - phi[1] - phi[2] - phi[3];
                    acc->point(x, lt, phi, quadrature::tet14_w(q) * V * det);
                }
            }
        };

        // The trilinear basis on one split tet T (vertices xt, hex corners corner) against the fluid
        // tet xf: B += int_{T cap T_f} phi_i psi_a. Returns the Tet4PairClass of (T, T_f).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE int hex8_trilinear_part(const double xs[8][3],
                                                                       const double xt[4][3],
                                                                       const int    corner[4],
                                                                       const double xf[4][3],
                                                                       double       B[4][8],
                                                                       int         *max_it) {
            double Jt[3][3], Jtinv[3][3];
            for (int d = 0; d < 3; ++d) {
                Jt[0][d] = xt[1][d] - xt[0][d];
                Jt[1][d] = xt[2][d] - xt[0][d];
                Jt[2][d] = xt[3][d] - xt[0][d];
            }
            const double det_t = fabs(tet4_inverse3(Jt[0], Jt[1], Jt[2], Jtinv));

            // The fluid vertices in T's reference coordinates and T's barycentric coordinates there.
            double xi[4][3], ls[4][4];
            for (int j = 0; j < 4; ++j) {
                double r[3];
                for (int d = 0; d < 3; ++d) r[d] = xf[j][d] - xt[0][d];
                for (int k = 0; k < 3; ++k) xi[j][k] = Jtinv[k][0] * r[0] + Jtinv[k][1] * r[1] + Jtinv[k][2] * r[2];
                ls[1][j] = xi[j][0];
                ls[2][j] = xi[j][1];
                ls[3][j] = xi[j][2];
                ls[0][j] = 1 - xi[j][0] - xi[j][1] - xi[j][2];
            }
            bool inside = true;
            for (int k = 0; k < 4; ++k) {
                bool out = true;
                for (int j = 0; j < 4; ++j) {
                    out    = out && ls[k][j] <= 0;
                    inside = inside && ls[k][j] >= 0;
                }
                if (out) return TET4_PAIR_SEPARATED;
            }

            double Jf[3][3], Jfinv[3][3];
            for (int d = 0; d < 3; ++d) {
                Jf[0][d] = xf[1][d] - xf[0][d];
                Jf[1][d] = xf[2][d] - xf[0][d];
                Jf[2][d] = xf[3][d] - xf[0][d];
            }
            const double vol_f = fabs(tet4_inverse3(Jf[0], Jf[1], Jf[2], Jfinv)) / 6;

            Hex8Accumulator acc{xs, corner, B, 0};
            if (inside) {
                // T_f in T: the rule on T_f, phi_i its barycentric coordinates.
                for (int q = 0; q < quadrature::tet14_n; ++q) {
                    double l[4], x[3] = {0, 0, 0}, lt[4] = {0, 0, 0, 0};
                    quadrature::tet14_l(q, l);
                    for (int j = 0; j < 4; ++j) {
                        for (int d = 0; d < 3; ++d) x[d] += l[j] * xf[j][d];
                        for (int k = 0; k < 4; ++k) lt[k] += l[j] * ls[k][j];
                    }
                    acc.point(x, lt, l, quadrature::tet14_w(q) * vol_f);
                }
                *max_it = acc.max_it;
                return TET4_PAIR_FLUID_INSIDE;
            }

            // The fluid's barycentric coordinates at T's vertices: lf[i][b] = phi_i(xt_b).
            double lf[4][4];
            bool   t_inside = true;
            for (int b = 0; b < 4; ++b) {
                double r[3];
                for (int d = 0; d < 3; ++d) r[d] = xt[b][d] - xf[0][d];
                double l[3];
                for (int k = 0; k < 3; ++k) l[k] = Jfinv[k][0] * r[0] + Jfinv[k][1] * r[1] + Jfinv[k][2] * r[2];
                lf[1][b] = l[0];
                lf[2][b] = l[1];
                lf[3][b] = l[2];
                lf[0][b] = 1 - l[0] - l[1] - l[2];
            }
            for (int i = 0; i < 4; ++i) {
                bool out = true;
                for (int b = 0; b < 4; ++b) {
                    out      = out && lf[i][b] <= 0;
                    t_inside = t_inside && lf[i][b] >= 0;
                }
                if (out) return TET4_PAIR_SEPARATED;
            }
            if (t_inside) {
                // T in T_f: the rule on T, phi_i affine.
                for (int q = 0; q < quadrature::tet14_n; ++q) {
                    double l[4], x[3] = {0, 0, 0}, phi[4] = {0, 0, 0, 0};
                    quadrature::tet14_l(q, l);
                    for (int b = 0; b < 4; ++b) {
                        for (int d = 0; d < 3; ++d) x[d] += l[b] * xt[b][d];
                        for (int i = 0; i < 4; ++i) phi[i] += l[b] * lf[i][b];
                    }
                    acc.point(x, l, phi, quadrature::tet14_w(q) * det_t / 6);
                }
                *max_it = acc.max_it;
                return TET4_PAIR_SOLID_INSIDE;
            }

            // The fluid tet in T's reference coordinates, relative to its centroid, clipped by T's
            // faces (as tet4_mixed_mass_pair).
            double c[3] = {0, 0, 0};
            for (int j = 0; j < 4; ++j) {
                for (int d = 0; d < 3; ++d) c[d] += xi[j][d] / 4;
            }
            double y[4][3];
            for (int j = 0; j < 4; ++j) {
                for (int d = 0; d < 3; ++d) y[j][d] = xi[j][d] - c[d];
            }
            ClipPolyhedron poly;
            tet4_polyhedron(y, &poly);
            {
                const double n0[3]   = {-1, -1, -1};
                const double e[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
                bool         ok      = tet4_clip_plane(&poly, 1 - c[0] - c[1] - c[2], n0);
                for (int k = 0; ok && k < 3; ++k) ok = tet4_clip_plane(&poly, c[k], e[k]);
                if (!ok) return TET4_PAIR_OVERFLOW;
            }
            if (poly.n == 0) return TET4_PAIR_SEPARATED;
            double p0[3] = {0, 0, 0};
            for (int v = 0; v < poly.n; ++v) {
                for (int d = 0; d < 3; ++d) p0[d] += poly.x[v][d] / poly.n;
            }
            Hex8FanVisit visit{&acc, xt, Jt, Jfinv, xf[0], c, p0, det_t};
            tet4_polyhedron_fan(&poly, visit);
            *max_it = acc.max_it;
            return TET4_PAIR_CLIPPED;
        }

        // B[i][a] for the hex xs (its nodes' global ids give its split) and the fluid tet xf, in the
        // basis asked for. counts[c] += the split tets of Tet4PairClass c; *max_it the Newton
        // iterations' maximum (TRILINEAR). Returns 0, or 1 if the ids have no split (not reached).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE int hex8_mixed_mass_pair(const double    xs[8][3],
                                                                        const long long ids[8],
                                                                        const double    xf[4][3],
                                                                        double          B[4][8],
                                                                        const Hex8Basis basis,
                                                                        int             counts[5],
                                                                        int            *max_it) {
            for (int i = 0; i < 4; ++i) {
                for (int a = 0; a < 8; ++a) B[i][a] = 0;
            }
            *max_it = 0;
            int       tets[6][4];
            const int n = hex8_split(hex8_split_pattern(ids), tets);
            if (!n) return 1;
            for (int k = 0; k < n; ++k) {
                double xt[4][3];
                for (int v = 0; v < 4; ++v) {
                    for (int d = 0; d < 3; ++d) xt[v][d] = xs[tets[k][v]][d];
                }
                int cls = 0;
                if (basis == Hex8Basis::SPLIT) {
                    double b[4][4];
                    cls = tet4_mixed_mass_pair(xt, xf, b);
                    for (int i = 0; i < 4; ++i) {
                        for (int v = 0; v < 4; ++v) B[i][tets[k][v]] += b[i][v];
                    }
                } else {
                    int it = 0;
                    cls    = hex8_trilinear_part(xs, xt, tets[k], xf, B, &it);
                    *max_it = it > *max_it ? it : *max_it;
                }
                counts[cls]++;
            }
            return 0;
        }

        // The volume of the hex's split tets (the region the transfer integrates over).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE double hex8_split_volume(const double xs[8][3], const long long ids[8]) {
            int       tets[6][4];
            const int n   = hex8_split(hex8_split_pattern(ids), tets);
            double    vol = 0;
            for (int k = 0; k < n; ++k) {
                double e[3][3];
                for (int j = 0; j < 3; ++j) {
                    for (int d = 0; d < 3; ++d) e[j][d] = xs[tets[k][j + 1]][d] - xs[tets[k][0]][d];
                }
                vol += fabs(tet4_det3(e[0], e[1], e[2])) / 6;
            }
            return vol;
        }

    }  // namespace intersect
}  // namespace sresample

#endif  // SRESAMPLE_HEX8_INTERSECT_KERNEL_HPP
