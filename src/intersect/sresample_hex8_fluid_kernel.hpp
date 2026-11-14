#ifndef SRESAMPLE_HEX8_FLUID_KERNEL_HPP
#define SRESAMPLE_HEX8_FLUID_KERNEL_HPP

#include <math.h>

#include "sresample_atomic.hpp"
#include "sresample_hex8_fluid.hpp"
#include "sresample_hex8_intersect_kernel.hpp"
#include "sresample_hex8_transfer_kernel.hpp"
#include "sresample_quadrature.hpp"
#include "sresample_tet4_intersect_kernel.hpp"
#include "sresample_tet4_transfer_kernel.hpp"

// One pair of the HEX8-fluid transfer (sresample_hex8_fluid.hpp): a HEX8 fluid element against a
// TET4 or HEX8 solid element, for the host loops and the device kernels alike (everything by value
// or on the stack).
//
// Each hex is its split tets (sresample_hex8_split.hpp), a tet itself. Every (solid tet T, fluid
// tet F) pair is intersected as in sresample_hex8_intersect_kernel.hpp (separated, F in T, T in F,
// or F clipped by T's faces and fanned from the polyhedron's vertex average) and integrated with
// each side's basis at the points:
//   a tet, or a hex in the SPLIT basis: the P1 basis of its (split) tet, from the point's
//                                       barycentric coordinates there;
//   a hex in the TRILINEAR basis:       the trilinear basis at the inverse of the hex's map (Newton,
//                                       hex8_inverse, from the barycentric coordinates mapped to the
//                                       reference cube's corners).
// The 14-point rule of degree 5, or the 24-point rule of degree 6 when both sides are trilinear
// (a trilinear times a trilinear is of degree 6 on a parallelepiped). When neither side is
// trilinear the pair is tet4_mixed_mass_pair in closed form.

namespace sresample {
    namespace intersect {

        // One side of a pair: its element's nodes, the corners of its current tet among them, and
        // its basis.
        struct Hex8FluidSide {
            const double (*xe)[3];  // the element's nodes (4 or 8)
            const int *corner;      // the current tet's corners (element-local)
            int        nodes;       // 4 or 8
            bool       trilinear;   // a hex in the TRILINEAR basis

            // The basis at x (barycentric coordinates l in the current tet) into N[nodes]; returns
            // Newton's iterations (0 without).
            SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE int basis(const double x[3], const double l[4], double N[8]) const {
                if (!trilinear) {
                    for (int a = 0; a < nodes; ++a) N[a] = 0;
                    for (int v = 0; v < 4; ++v) N[corner[v]] += l[v];
                    return 0;
                }
                double s[3] = {0, 0, 0};
                for (int v = 0; v < 4; ++v) {
                    for (int d = 0; d < 3; ++d) s[d] += l[v] * hex8_corner(corner[v], d);
                }
                return hex8_inverse(xe, x, s, N);
            }
        };

        // B[i][a] += w Nf_i(x) Ns_a(x), lt and lf the point's barycentric coordinates in the solid's
        // and the fluid's current tets.
        struct Hex8FluidAccumulator {
            Hex8FluidSide solid, fluid;
            double (*B)[8];
            int max_it{0};

            SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void point(const double x[3], const double lt[4], const double lf[4], const double w) {
                double    Ns[8], Nf[8];
                const int its = solid.basis(x, lt, Ns);
                const int itf = fluid.basis(x, lf, Nf);
                max_it        = its > max_it ? its : max_it;
                max_it        = itf > max_it ? itf : max_it;
                for (int i = 0; i < fluid.nodes; ++i) {
                    const double wi = w * Nf[i];
                    for (int a = 0; a < solid.nodes; ++a) B[i][a] += wi * Ns[a];
                }
            }
        };

        // The rule: 14 points (degree 5) or 24 (degree 6).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE int hex8_fluid_rule_n(const bool six) { return six ? quadrature::tet24_n : quadrature::tet14_n; }

        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE double hex8_fluid_rule(const bool six, const int q, double l[4]) {
            if (six) {
                quadrature::tet24_l(q, l);
                return quadrature::tet24_w(q);
            }
            quadrature::tet14_l(q, l);
            return quadrature::tet14_w(q);
        }

        // The fan of the clipped polyhedron (in T's reference coordinates relative to c), as
        // Hex8FanVisit, with both sides' bases.
        struct Hex8FluidFanVisit {
            Hex8FluidAccumulator *acc;
            const double (*xt)[3];
            const double (*Jt)[3];     // rows: T's edges xt_k - xt_0
            const double (*Jfinv)[3];  // F's inverse map
            const double *xf0;
            const double *c;
            const double *p0;
            double        det;
            bool          six;

            SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void operator()(const double *a, const double *b, const double *cc) {
                double ea[3], eb[3], ec[3];
                for (int d = 0; d < 3; ++d) {
                    ea[d] = a[d] - p0[d];
                    eb[d] = b[d] - p0[d];
                    ec[d] = cc[d] - p0[d];
                }
                const double V = tet4_det3(ea, eb, ec) / 6;
                if (V == 0) return;
                const int nq = hex8_fluid_rule_n(six);
                for (int q = 0; q < nq; ++q) {
                    double       l[4];
                    const double w = hex8_fluid_rule(six, q, l);
                    double       xi[3], x[3], lt[4], lf[4];
                    for (int d = 0; d < 3; ++d) xi[d] = p0[d] + l[1] * ea[d] + l[2] * eb[d] + l[3] * ec[d] + c[d];
                    for (int d = 0; d < 3; ++d) x[d] = xt[0][d] + xi[0] * Jt[0][d] + xi[1] * Jt[1][d] + xi[2] * Jt[2][d];
                    lt[0] = 1 - xi[0] - xi[1] - xi[2];
                    lt[1] = xi[0];
                    lt[2] = xi[1];
                    lt[3] = xi[2];
                    double r[3];
                    for (int d = 0; d < 3; ++d) r[d] = x[d] - xf0[d];
                    for (int k = 0; k < 3; ++k) lf[k + 1] = Jfinv[k][0] * r[0] + Jfinv[k][1] * r[1] + Jfinv[k][2] * r[2];
                    lf[0] = 1 - lf[1] - lf[2] - lf[3];
                    acc->point(x, lt, lf, w * V * det);
                }
            }
        };

        // B += int_{T cap F} Nf_i Ns_a for the solid tet xt and the fluid tet xf (the accumulator's
        // sides at them). Returns the Tet4PairClass of (T, F).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE int hex8_fluid_part(const double xt[4][3], const double xf[4][3], Hex8FluidAccumulator *acc, const bool six) {
            double Jt[3][3], Jtinv[3][3];
            for (int d = 0; d < 3; ++d) {
                Jt[0][d] = xt[1][d] - xt[0][d];
                Jt[1][d] = xt[2][d] - xt[0][d];
                Jt[2][d] = xt[3][d] - xt[0][d];
            }
            const double det_t = fabs(tet4_inverse3(Jt[0], Jt[1], Jt[2], Jtinv));

            // F's vertices in T's reference coordinates and T's barycentric coordinates there.
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
            const int    nq    = hex8_fluid_rule_n(six);

            if (inside) {
                // F in T: the rule on F.
                for (int q = 0; q < nq; ++q) {
                    double       l[4], x[3] = {0, 0, 0}, lt[4] = {0, 0, 0, 0};
                    const double w = hex8_fluid_rule(six, q, l);
                    for (int j = 0; j < 4; ++j) {
                        for (int d = 0; d < 3; ++d) x[d] += l[j] * xf[j][d];
                        for (int k = 0; k < 4; ++k) lt[k] += l[j] * ls[k][j];
                    }
                    acc->point(x, lt, l, w * vol_f);
                }
                return TET4_PAIR_FLUID_INSIDE;
            }

            // F's barycentric coordinates at T's vertices: lfv[i][b] = lambda_i(xt_b).
            double lfv[4][4];
            bool   t_inside = true;
            for (int b = 0; b < 4; ++b) {
                double r[3];
                for (int d = 0; d < 3; ++d) r[d] = xt[b][d] - xf[0][d];
                double l[3];
                for (int k = 0; k < 3; ++k) l[k] = Jfinv[k][0] * r[0] + Jfinv[k][1] * r[1] + Jfinv[k][2] * r[2];
                lfv[1][b] = l[0];
                lfv[2][b] = l[1];
                lfv[3][b] = l[2];
                lfv[0][b] = 1 - l[0] - l[1] - l[2];
            }
            for (int i = 0; i < 4; ++i) {
                bool out = true;
                for (int b = 0; b < 4; ++b) {
                    out      = out && lfv[i][b] <= 0;
                    t_inside = t_inside && lfv[i][b] >= 0;
                }
                if (out) return TET4_PAIR_SEPARATED;
            }
            if (t_inside) {
                // T in F: the rule on T.
                for (int q = 0; q < nq; ++q) {
                    double       l[4], x[3] = {0, 0, 0}, lf[4] = {0, 0, 0, 0};
                    const double w = hex8_fluid_rule(six, q, l);
                    for (int b = 0; b < 4; ++b) {
                        for (int d = 0; d < 3; ++d) x[d] += l[b] * xt[b][d];
                        for (int i = 0; i < 4; ++i) lf[i] += l[b] * lfv[i][b];
                    }
                    acc->point(x, l, lf, w * det_t / 6);
                }
                return TET4_PAIR_SOLID_INSIDE;
            }

            // F in T's reference coordinates, relative to its centroid, clipped by T's faces.
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
            Hex8FluidFanVisit visit{acc, xt, Jt, Jfinv, xf[0], c, p0, det_t, six};
            tet4_polyhedron_fan(&poly, visit);
            return TET4_PAIR_CLIPPED;
        }

        // Whether the boxes of two tets are apart.
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE bool hex8_fluid_boxes_apart(const double a[4][3], const double b[4][3]) {
            for (int d = 0; d < 3; ++d) {
                double amin = a[0][d], amax = a[0][d], bmin = b[0][d], bmax = b[0][d];
                for (int v = 1; v < 4; ++v) {
                    amin = a[v][d] < amin ? a[v][d] : amin;
                    amax = a[v][d] > amax ? a[v][d] : amax;
                    bmin = b[v][d] < bmin ? b[v][d] : bmin;
                    bmax = b[v][d] > bmax ? b[v][d] : bmax;
                }
                if (amax < bmin || bmax < amin) return true;
            }
            return false;
        }

        // B[i][a], i < 8 the fluid hex's nodes, a < ns the solid's (a tet, ns = 4, or a hex, 8), for
        // the solid xs (ids: its split, if a hex) and the fluid hex xf (ids_f), in the bases asked
        // for. counts[c] += the (solid tet, fluid tet) pairs of Tet4PairClass c; *max_it Newton's
        // most iterations. Returns 0, or 1 if a hex's ids have no split (not reached).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE int hex8_fluid_mixed_mass_pair(const double    xs[8][3],
                                                                              const long long ids_s[8],
                                                                              const int       ns,
                                                                              const double    xf[8][3],
                                                                              const long long ids_f[8],
                                                                              double          B[8][8],
                                                                              const Hex8Basis solid_basis,
                                                                              const Hex8Basis fluid_basis,
                                                                              int             counts[5],
                                                                              int            *max_it) {
            for (int i = 0; i < 8; ++i) {
                for (int a = 0; a < 8; ++a) B[i][a] = 0;
            }
            *max_it = 0;
            int tets_s[6][4], tets_f[6][4];
            int n_s = 1;
            if (ns == 8) {
                n_s = hex8_split(hex8_split_pattern(ids_s), tets_s);
            } else {
                for (int v = 0; v < 4; ++v) tets_s[0][v] = v;
            }
            const int n_f = hex8_split(hex8_split_pattern(ids_f), tets_f);
            if (!n_s || !n_f) return 1;

            const bool s_tri = ns == 8 && solid_basis == Hex8Basis::TRILINEAR;
            const bool f_tri = fluid_basis == Hex8Basis::TRILINEAR;
            for (int m = 0; m < n_f; ++m) {
                double xfm[4][3];
                for (int v = 0; v < 4; ++v) {
                    for (int d = 0; d < 3; ++d) xfm[v][d] = xf[tets_f[m][v]][d];
                }
                for (int k = 0; k < n_s; ++k) {
                    double xt[4][3];
                    for (int v = 0; v < 4; ++v) {
                        for (int d = 0; d < 3; ++d) xt[v][d] = xs[tets_s[k][v]][d];
                    }
                    if (hex8_fluid_boxes_apart(xt, xfm)) {
                        counts[TET4_PAIR_SEPARATED]++;
                        continue;
                    }
                    int cls = 0;
                    if (!s_tri && !f_tri) {
                        // P1 on both tets: in closed form.
                        double b[4][4];
                        cls = tet4_mixed_mass_pair(xt, xfm, b);
                        for (int i = 0; i < 4; ++i) {
                            for (int v = 0; v < 4; ++v) B[tets_f[m][i]][tets_s[k][v]] += b[i][v];
                        }
                    } else {
                        Hex8FluidAccumulator acc{{xs, tets_s[k], ns, s_tri}, {xf, tets_f[m], 8, f_tri}, B, 0};
                        cls     = hex8_fluid_part(xt, xfm, &acc, s_tri && f_tri);
                        *max_it = acc.max_it > *max_it ? acc.max_it : *max_it;
                    }
                    counts[cls]++;
                }
            }
            return 0;
        }

        template <typename TGeom, typename TIdx>
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void hex8_fluid_load(const MeshView<TGeom, TIdx> &mesh,
                                                                    const ptrdiff_t              e,
                                                                    const int                    n,
                                                                    double                       x[8][3],
                                                                    long long                    ids[8]) {
            for (int v = 0; v < n; ++v) {
                const ptrdiff_t node = (ptrdiff_t)mesh.elements[v][e];
                ids[v]               = mesh.ids ? mesh.ids[node] : (long long)node;
                for (int d = 0; d < 3; ++d) x[v][d] = (double)mesh.points[d][node];
            }
        }

        // B of pair k into the structure of arrays B[(ns i + a) npairs + k]; adds the classes to
        // counts and Newton's iterations' maximum to *max_it.
        template <typename TReal, typename TGeom, typename TIdx>
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void hex8_fluid_pair_mixed_mass(const MeshView<TGeom, TIdx> &solid,
                                                                               const MeshView<TGeom, TIdx> &fluid,
                                                                               const ptrdiff_t              npairs,
                                                                               const ptrdiff_t              k,
                                                                               const int *const             pair_solid,
                                                                               const int *const             pair_fluid,
                                                                               TReal *const                 B,
                                                                               const Hex8Basis              solid_basis,
                                                                               const Hex8Basis              fluid_basis,
                                                                               int                          counts[5],
                                                                               int                         *max_it) {
            const int ns = nxe(solid.type);
            double    xs[8][3], xf[8][3], b[8][8];
            long long ids_s[8], ids_f[8];
            hex8_fluid_load(solid, pair_solid[k], ns, xs, ids_s);
            hex8_fluid_load(fluid, pair_fluid[k], 8, xf, ids_f);
            int it = 0;
            if (hex8_fluid_mixed_mass_pair(xs, ids_s, ns, xf, ids_f, b, solid_basis, fluid_basis, counts, &it)) counts[TET4_PAIR_OVERFLOW]++;
            *max_it = it > *max_it ? it : *max_it;
            for (int i = 0; i < 8; ++i) {
                for (int a = 0; a < ns; ++a) B[(ns * i + a) * npairs + k] = (TReal)b[i][a];
            }
        }

        template <typename TReal, typename TGeom, typename TIdx>
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void hex8_fluid_pair_apply(const MeshView<TGeom, TIdx> &solid,
                                                                          const MeshView<TGeom, TIdx> &fluid,
                                                                          const ptrdiff_t              npairs,
                                                                          const ptrdiff_t              k,
                                                                          const int *const             pair_solid,
                                                                          const int *const             pair_fluid,
                                                                          const TReal *const           B,
                                                                          const TReal *const           v0,
                                                                          const TReal *const           v1,
                                                                          const TReal *const           v2,
                                                                          TReal *const                 W,
                                                                          TReal *const                 U0,
                                                                          TReal *const                 U1,
                                                                          TReal *const                 U2) {
            const int       ns = nxe(solid.type);
            const ptrdiff_t s = pair_solid[k], f = pair_fluid[k];
            TReal           vs[8][3];
            for (int a = 0; a < 8; ++a) vs[a][0] = vs[a][1] = vs[a][2] = 0;
            if (U0) {
                for (int a = 0; a < ns; ++a) {
                    const ptrdiff_t node = (ptrdiff_t)solid.elements[a][s];
                    vs[a][0]             = v0[node];
                    vs[a][1]             = v1[node];
                    vs[a][2]             = v2[node];
                }
            }
            for (int i = 0; i < 8; ++i) {
                TReal w = 0, u[3] = {0, 0, 0};
                for (int a = 0; a < ns; ++a) {
                    const TReal bia = B[(ns * i + a) * npairs + k];
                    w += bia;
                    if (U0) {
                        for (int d = 0; d < 3; ++d) u[d] += bia * vs[a][d];
                    }
                }
                if (w == 0) continue;
                const ptrdiff_t node = (ptrdiff_t)fluid.elements[i][f];
                atomic_add(&W[node], w);
                if (U0) {
                    atomic_add(&U0[node], u[0]);
                    atomic_add(&U1[node], u[1]);
                    atomic_add(&U2[node], u[2]);
                }
            }
        }

        template <typename TReal, typename TGeom, typename TIdx>
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void hex8_fluid_pair_apply_transpose(const MeshView<TGeom, TIdx> &solid,
                                                                                    const MeshView<TGeom, TIdx> &fluid,
                                                                                    const ptrdiff_t              npairs,
                                                                                    const ptrdiff_t              k,
                                                                                    const int *const             pair_solid,
                                                                                    const int *const             pair_fluid,
                                                                                    const TReal *const           B,
                                                                                    const TReal *const           u0,
                                                                                    const TReal *const           u1,
                                                                                    const TReal *const           u2,
                                                                                    TReal *const                 c,
                                                                                    TReal *const                 g0,
                                                                                    TReal *const                 g1,
                                                                                    TReal *const                 g2) {
            const int       ns = nxe(solid.type);
            const ptrdiff_t s = pair_solid[k], f = pair_fluid[k];
            TReal           uf[8][3];
            for (int i = 0; i < 8; ++i) uf[i][0] = uf[i][1] = uf[i][2] = 0;
            if (g0) {
                for (int i = 0; i < 8; ++i) {
                    const ptrdiff_t node = (ptrdiff_t)fluid.elements[i][f];
                    uf[i][0]             = u0[node];
                    uf[i][1]             = u1[node];
                    uf[i][2]             = u2[node];
                }
            }
            for (int a = 0; a < ns; ++a) {
                TReal w = 0, g[3] = {0, 0, 0};
                for (int i = 0; i < 8; ++i) {
                    const TReal bia = B[(ns * i + a) * npairs + k];
                    w += bia;
                    if (g0) {
                        for (int d = 0; d < 3; ++d) g[d] += bia * uf[i][d];
                    }
                }
                if (w == 0) continue;
                const ptrdiff_t node = (ptrdiff_t)solid.elements[a][s];
                if (c) atomic_add(&c[node], w);
                if (g0) {
                    atomic_add(&g0[node], g[0]);
                    atomic_add(&g1[node], g[1]);
                    atomic_add(&g2[node], g[2]);
                }
            }
        }

    }  // namespace intersect
}  // namespace sresample

#endif  // SRESAMPLE_HEX8_FLUID_KERNEL_HPP
