#ifndef SRESAMPLE_TET4_INTERSECT_KERNEL_HPP
#define SRESAMPLE_TET4_INTERSECT_KERNEL_HPP

#include <math.h>

#include "sresample_base.hpp"

// The intersection of two tetrahedra (TET4, linear), exactly: the mixed mass matrix
//
//   B_ia = int_{T_s cap T_f} phi_i psi_a dx,
//
// phi_i the P1 basis of the "fluid" tet T_f, psi_a that of the "solid" tet T_s, for the host loop
// and the device kernel alike (everything by value or on the stack).
//
// The fluid tet is clipped by the solid tet's four faces in the solid's reference coordinates xi
// (x = x_s0 + J xi), where the faces are the planes lambda_k(xi) = 0 of its barycentric
// coordinates. The clipping is r3d's (Powell & Abel, J. Comput. Phys. 297, 2015): the polyhedron
// is a graph of degree-3 vertices, each plane inserts a vertex on every edge it cuts, relinks the
// new vertices around the cut face and drops the vertices outside; no face lists, no sorting, no
// tolerance in the topology (a vertex on a plane is kept; a zero-length edge contributes nothing).
// Its exact moments up to order 2 (volume, first and second) come from fanning every face to the
// origin into tetrahedra with closed-form moments, and B is a quadratic form in them, since
// phi_i and psi_a are affine. The coordinates are relative to the mapped fluid tet's centroid,
// so their magnitude is the tet's size (no cancellation against far-away origins).
//
// Before clipping, the cases that need none are decided from the barycentric coordinates of each
// tet's vertices in the other (exact sign tests): separated (B = 0), the fluid tet inside the
// solid one (B from the fluid tet's mass matrix) and the solid tet inside the fluid one (B from
// the solid tet's).

namespace sresample {
    namespace intersect {

        // The clipped polyhedron's capacity. Clipping a polyhedron with F faces inserts at most F
        // vertices (a plane cuts a convex face's boundary in at most two edges, each edge on two
        // faces) and adds at most one face, and drops at least one vertex: from a tet (4 vertices,
        // 4 faces) the four planes give at most 4+4, 7+5, 11+6 and 16+7 = 23 vertices before the
        // last compaction. Non-degenerate cases stay at 12 or fewer.
        static constexpr int tet4_clip_max_vertices = 24;

        // How a pair was integrated (the return value of tet4_mixed_mass_pair).
        enum Tet4PairClass {
            TET4_PAIR_SEPARATED    = 0,  // no common volume: B = 0
            TET4_PAIR_FLUID_INSIDE = 1,  // T_f in T_s
            TET4_PAIR_SOLID_INSIDE = 2,  // T_s in T_f
            TET4_PAIR_CLIPPED      = 3,  // the general case
            TET4_PAIR_OVERFLOW     = 4   // the polyhedron outgrew its capacity: B = 0 (a bug)
        };

        struct ClipPolyhedron {
            int           n{0};
            double        x[tet4_clip_max_vertices][3];
            unsigned char nb[tet4_clip_max_vertices][3];
        };

        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE double tet4_det3(const double a[3], const double b[3], const double c[3]) {
            return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
        }

        // inv = M^-1 for M = [c0 c1 c2] (columns), and det M.
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE double tet4_inverse3(const double c0[3],
                                                                   const double c1[3],
                                                                   const double c2[3],
                                                                   double       inv[3][3]) {
            const double det = tet4_det3(c0, c1, c2);
            const double s   = 1 / det;
            // Rows of the inverse: the cross products of the columns.
            inv[0][0] = (c1[1] * c2[2] - c1[2] * c2[1]) * s;
            inv[0][1] = (c1[2] * c2[0] - c1[0] * c2[2]) * s;
            inv[0][2] = (c1[0] * c2[1] - c1[1] * c2[0]) * s;
            inv[1][0] = (c2[1] * c0[2] - c2[2] * c0[1]) * s;
            inv[1][1] = (c2[2] * c0[0] - c2[0] * c0[2]) * s;
            inv[1][2] = (c2[0] * c0[1] - c2[1] * c0[0]) * s;
            inv[2][0] = (c0[1] * c1[2] - c0[2] * c1[1]) * s;
            inv[2][1] = (c0[2] * c1[0] - c0[0] * c1[2]) * s;
            inv[2][2] = (c0[0] * c1[1] - c0[1] * c1[0]) * s;
            return det;
        }

        // The tet's vertices as the polyhedron (r3d_init_tet's connectivity), positively oriented:
        // vertices 2 and 3 are swapped if det(v1 - v0, v2 - v0, v3 - v0) < 0.
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void tet4_polyhedron(const double v[4][3], ClipPolyhedron *const p) {
            double e1[3], e2[3], e3[3];
            for (int d = 0; d < 3; ++d) {
                e1[d] = v[1][d] - v[0][d];
                e2[d] = v[2][d] - v[0][d];
                e3[d] = v[3][d] - v[0][d];
            }
            const bool flip = tet4_det3(e1, e2, e3) < 0;
            for (int d = 0; d < 3; ++d) {
                p->x[0][d] = v[0][d];
                p->x[1][d] = v[1][d];
                p->x[2][d] = v[flip ? 3 : 2][d];
                p->x[3][d] = v[flip ? 2 : 3][d];
            }
            // r3d_init_tet: vertex k's neighbours, counterclockwise seen from outside.
            p->nb[0][0] = 1;
            p->nb[0][1] = 3;
            p->nb[0][2] = 2;
            p->nb[1][0] = 2;
            p->nb[1][1] = 3;
            p->nb[1][2] = 0;
            p->nb[2][0] = 0;
            p->nb[2][1] = 3;
            p->nb[2][2] = 1;
            p->nb[3][0] = 1;
            p->nb[3][1] = 2;
            p->nb[3][2] = 0;
            p->n        = 4;
        }

        // Keeps the part of the polyhedron where a + n . x >= 0. Returns false on overflow.
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE bool tet4_clip_plane(ClipPolyhedron *const p, const double a, const double n[3]) {
            const int onv = p->n;
            if (onv <= 0) return true;
            double        s[tet4_clip_max_vertices];
            unsigned char map[tet4_clip_max_vertices];  // 1 = clipped, then the new index
            double        smin = 1e300, smax = -1e300;
            for (int v = 0; v < onv; ++v) {
                s[v]   = a + n[0] * p->x[v][0] + n[1] * p->x[v][1] + n[2] * p->x[v][2];
                map[v] = s[v] < 0 ? 1 : 0;
                smin   = s[v] < smin ? s[v] : smin;
                smax   = s[v] > smax ? s[v] : smax;
            }
            if (smin >= 0) return true;  // all inside
            if (smax <= 0) {             // all outside (or a face on the plane): nothing left
                p->n = 0;
                return true;
            }

            // A new vertex on every edge from a kept vertex to a clipped one, interpolated from the
            // kept end: x = (x_in s_in' + x_out s_out') with weights -s_out, s_in > 0.
            int nv = onv;
            for (int vcur = 0; vcur < onv; ++vcur) {
                if (map[vcur]) continue;
                for (int k = 0; k < 3; ++k) {
                    const int vnext = p->nb[vcur][k];
                    if (!map[vnext]) continue;
                    if (nv >= tet4_clip_max_vertices) return false;
                    const double wa = -s[vnext], wb = s[vcur];
                    const double inv = 1 / (wa + wb);
                    for (int d = 0; d < 3; ++d) p->x[nv][d] = (p->x[vcur][d] * wa + p->x[vnext][d] * wb) * inv;
                    p->nb[nv][0]   = (unsigned char)vcur;
                    p->nb[vcur][k] = (unsigned char)nv;
                    map[nv]        = 0;
                    ++nv;
                }
            }

            // Link the new vertices around the cut face: from each, walk the face on its left
            // until the next new vertex.
            for (int vstart = onv; vstart < nv; ++vstart) {
                int vcur  = vstart;
                int vnext = p->nb[vcur][0];
                int guard = 0;
                do {
                    int k = 0;
                    while (k < 3 && p->nb[vnext][k] != vcur) ++k;
                    if (k == 3 || ++guard > 4 * tet4_clip_max_vertices) return false;  // a broken graph
                    vcur  = vnext;
                    vnext = p->nb[vcur][(k + 1) % 3];
                } while (vcur < onv);
                p->nb[vstart][2] = (unsigned char)vcur;
                p->nb[vcur][1]   = (unsigned char)vstart;
            }

            // Compact: drop the clipped vertices, renumber the neighbours.
            int m = 0;
            for (int v = 0; v < nv; ++v) {
                if (map[v]) continue;
                if (m != v) {
                    for (int d = 0; d < 3; ++d) p->x[m][d] = p->x[v][d];
                    for (int k = 0; k < 3; ++k) p->nb[m][k] = p->nb[v][k];
                }
                map[v] = (unsigned char)m++;
            }
            for (int v = 0; v < m; ++v) {
                for (int k = 0; k < 3; ++k) p->nb[v][k] = map[p->nb[v][k]];
            }
            p->n = m;
            return true;
        }

        // The polyhedron's moments: m[0] = vol, m[1..3] = int x, m[4..9] = int x x^T as xx, xy, xz,
        // yy, yz, zz. Every face fanned from its first vertex, each triangle with the origin a tet
        // (0, a, b, c) of volume a.(b x c)/6, first moment V (a+b+c)/4 and second moment
        // V/20 (a a^T + b b^T + c c^T + s s^T), s = a+b+c.
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void tet4_polyhedron_moments(const ClipPolyhedron *const p, double m[10]) {
            for (int k = 0; k < 10; ++k) m[k] = 0;
            const int     nv = p->n;
            unsigned char mark[tet4_clip_max_vertices][3];
            for (int v = 0; v < nv; ++v) mark[v][0] = mark[v][1] = mark[v][2] = 0;

            for (int vstart = 0; vstart < nv; ++vstart) {
                for (int kstart = 0; kstart < 3; ++kstart) {
                    if (mark[vstart][kstart]) continue;
                    // The face on the left of edge (vstart, nb[vstart][kstart]).
                    int vcur  = vstart;
                    int vnext = p->nb[vcur][kstart];
                    mark[vcur][kstart] = 1;
                    const double *a = p->x[vstart];
                    // Second edge.
                    int k = 0;
                    while (k < 3 && p->nb[vnext][k] != vcur) ++k;
                    vcur              = vnext;
                    const int knext   = (k + 1) % 3;
                    mark[vcur][knext] = 1;
                    vnext             = p->nb[vcur][knext];
                    int guard         = 0;
                    while (vnext != vstart && guard++ < 4 * tet4_clip_max_vertices) {
                        const double *c = p->x[vcur];
                        const double *b = p->x[vnext];
                        // 6V = a . (b x c) with b the next vertex and c the current (r3d's order).
                        const double V = tet4_det3(a, b, c) / 6;
                        double       sm[3];
                        for (int d = 0; d < 3; ++d) sm[d] = a[d] + b[d] + c[d];
                        m[0] += V;
                        for (int d = 0; d < 3; ++d) m[1 + d] += V * sm[d] / 4;
                        const double w = V / 20;
                        m[4] += w * (a[0] * a[0] + b[0] * b[0] + c[0] * c[0] + sm[0] * sm[0]);
                        m[5] += w * (a[0] * a[1] + b[0] * b[1] + c[0] * c[1] + sm[0] * sm[1]);
                        m[6] += w * (a[0] * a[2] + b[0] * b[2] + c[0] * c[2] + sm[0] * sm[2]);
                        m[7] += w * (a[1] * a[1] + b[1] * b[1] + c[1] * c[1] + sm[1] * sm[1]);
                        m[8] += w * (a[1] * a[2] + b[1] * b[2] + c[1] * c[2] + sm[1] * sm[2]);
                        m[9] += w * (a[2] * a[2] + b[2] * b[2] + c[2] * c[2] + sm[2] * sm[2]);
                        // Next edge.
                        k = 0;
                        while (k < 3 && p->nb[vnext][k] != vcur) ++k;
                        vcur              = vnext;
                        const int kn      = (k + 1) % 3;
                        mark[vcur][kn]    = 1;
                        vnext             = p->nb[vcur][kn];
                    }
                }
            }
        }

        // The polyhedron as tets (0, a, b, c): every face fanned from its first vertex, each triangle
        // with the origin, as tet4_polyhedron_moments; visit(a, b, c) for each, its signed volume
        // a . (b x c) / 6 (they add up to the polyhedron's; some are negative where the origin is
        // outside it).
        template <typename Visit>
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void tet4_polyhedron_fan(const ClipPolyhedron *const p, Visit &visit) {
            const int     nv = p->n;
            unsigned char mark[tet4_clip_max_vertices][3];
            for (int v = 0; v < nv; ++v) mark[v][0] = mark[v][1] = mark[v][2] = 0;

            for (int vstart = 0; vstart < nv; ++vstart) {
                for (int kstart = 0; kstart < 3; ++kstart) {
                    if (mark[vstart][kstart]) continue;
                    int vcur  = vstart;
                    int vnext = p->nb[vcur][kstart];
                    mark[vcur][kstart] = 1;
                    const double *a = p->x[vstart];
                    int k = 0;
                    while (k < 3 && p->nb[vnext][k] != vcur) ++k;
                    vcur              = vnext;
                    const int knext   = (k + 1) % 3;
                    mark[vcur][knext] = 1;
                    vnext             = p->nb[vcur][knext];
                    int guard         = 0;
                    while (vnext != vstart && guard++ < 4 * tet4_clip_max_vertices) {
                        visit(a, p->x[vnext], p->x[vcur]);
                        k = 0;
                        while (k < 3 && p->nb[vnext][k] != vcur) ++k;
                        vcur           = vnext;
                        const int kn   = (k + 1) % 3;
                        mark[vcur][kn] = 1;
                        vnext          = p->nb[vcur][kn];
                    }
                }
            }
        }

        // B[i][a] = int_{T_s cap T_f} phi_i psi_a for the fluid tet xf and the solid tet xs (vertex,
        // axis), both non-degenerate, either orientation. Returns the Tet4PairClass. force_clip
        // skips the containment shortcuts (for testing the clipping on those cases).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE int tet4_mixed_mass_pair(const double xs[4][3],
                                                                        const double xf[4][3],
                                                                        double       B[4][4],
                                                                        const bool   force_clip = false) {
            for (int i = 0; i < 4; ++i) {
                for (int a = 0; a < 4; ++a) B[i][a] = 0;
            }

            // The solid tet's affine map x = xs0 + J xi.
            double Js[3][3], Jinv[3][3];
            for (int d = 0; d < 3; ++d) {
                Js[0][d] = xs[1][d] - xs[0][d];
                Js[1][d] = xs[2][d] - xs[0][d];
                Js[2][d] = xs[3][d] - xs[0][d];
            }
            const double det_s = tet4_inverse3(Js[0], Js[1], Js[2], Jinv);
            const double vol_s = fabs(det_s) / 6;

            // The fluid vertices in xi, and the solid's barycentric coordinates there: ls[k][j] =
            // lambda_k(xf_j).
            double xi[4][3], ls[4][4];
            for (int j = 0; j < 4; ++j) {
                double r[3];
                for (int d = 0; d < 3; ++d) r[d] = xf[j][d] - xs[0][d];
                for (int k = 0; k < 3; ++k) xi[j][k] = Jinv[k][0] * r[0] + Jinv[k][1] * r[1] + Jinv[k][2] * r[2];
                ls[1][j] = xi[j][0];
                ls[2][j] = xi[j][1];
                ls[3][j] = xi[j][2];
                ls[0][j] = 1 - xi[j][0] - xi[j][1] - xi[j][2];
            }

            // Separated by a solid face: all fluid vertices on its outer side.
            bool inside = true;
            for (int k = 0; k < 4; ++k) {
                bool out = true;
                for (int j = 0; j < 4; ++j) {
                    out    = out && ls[k][j] <= 0;
                    inside = inside && ls[k][j] >= 0;
                }
                if (out) return TET4_PAIR_SEPARATED;
            }

            // The fluid tet's affine map.
            double Jf[3][3], Jfinv[3][3];
            for (int d = 0; d < 3; ++d) {
                Jf[0][d] = xf[1][d] - xf[0][d];
                Jf[1][d] = xf[2][d] - xf[0][d];
                Jf[2][d] = xf[3][d] - xf[0][d];
            }
            const double det_f = tet4_inverse3(Jf[0], Jf[1], Jf[2], Jfinv);
            const double vol_f = fabs(det_f) / 6;

            if (inside && !force_clip) {
                // T_f in T_s: psi_a = sum_j psi_a(xf_j) phi_j on T_f, so B = M_f Lambda^T with the
                // fluid mass matrix M_f = vol_f / 20 (1 + delta_ij).
                for (int i = 0; i < 4; ++i) {
                    for (int a = 0; a < 4; ++a) {
                        double sum = 0;
                        for (int j = 0; j < 4; ++j) sum += (i == j ? 2.0 : 1.0) * ls[a][j];
                        B[i][a] = vol_f / 20 * sum;
                    }
                }
                return TET4_PAIR_FLUID_INSIDE;
            }

            // The fluid's barycentric coordinates at the solid vertices: lf[i][b] = phi_i(xs_b).
            double lf[4][4];
            bool   s_inside = true;
            for (int b = 0; b < 4; ++b) {
                double r[3];
                for (int d = 0; d < 3; ++d) r[d] = xs[b][d] - xf[0][d];
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
                    s_inside = s_inside && lf[i][b] >= 0;
                }
                if (out) return TET4_PAIR_SEPARATED;
            }
            if (s_inside && !force_clip) {
                // T_s in T_f: phi_i = sum_b phi_i(xs_b) psi_b on T_s, B = Phi M_s.
                for (int i = 0; i < 4; ++i) {
                    for (int a = 0; a < 4; ++a) {
                        double sum = 0;
                        for (int b = 0; b < 4; ++b) sum += lf[i][b] * (a == b ? 2.0 : 1.0);
                        B[i][a] = vol_s / 20 * sum;
                    }
                }
                return TET4_PAIR_SOLID_INSIDE;
            }

            // The general case: the fluid tet in xi, relative to its centroid c, clipped by
            // lambda_k(xi) >= 0, i.e. by a_k + n_k . y >= 0 for y = xi - c.
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
                const double n0[3] = {-1, -1, -1};
                const double e[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
                bool ok = tet4_clip_plane(&poly, 1 - c[0] - c[1] - c[2], n0);
                for (int k = 0; ok && k < 3; ++k) ok = tet4_clip_plane(&poly, c[k], e[k]);
                if (!ok) return TET4_PAIR_OVERFLOW;
            }
            if (poly.n == 0) return TET4_PAIR_SEPARATED;

            double m[10];
            tet4_polyhedron_moments(&poly, m);
            const double M2[3][3] = {{m[4], m[5], m[6]}, {m[5], m[7], m[8]}, {m[6], m[8], m[9]}};

            // psi_a = alpha_a + h_a . y: lambda_0 = 1 - sum(xi), lambda_k = xi_k.
            double alpha[4], h[4][3];
            alpha[0] = 1 - c[0] - c[1] - c[2];
            for (int d = 0; d < 3; ++d) h[0][d] = -1;
            for (int k = 1; k < 4; ++k) {
                alpha[k] = c[k - 1];
                for (int d = 0; d < 3; ++d) h[k][d] = d == k - 1 ? 1 : 0;
            }
            // phi_i = beta_i + g_i . y, from the fluid tet's vertices y_j.
            double E[3][3], Einv[3][3];
            for (int d = 0; d < 3; ++d) {
                E[0][d] = y[1][d] - y[0][d];
                E[1][d] = y[2][d] - y[0][d];
                E[2][d] = y[3][d] - y[0][d];
            }
            tet4_inverse3(E[0], E[1], E[2], Einv);
            double beta[4], g[4][3];
            beta[0] = 1;
            for (int d = 0; d < 3; ++d) g[0][d] = 0;
            for (int i = 1; i < 4; ++i) {
                beta[i] = -(Einv[i - 1][0] * y[0][0] + Einv[i - 1][1] * y[0][1] + Einv[i - 1][2] * y[0][2]);
                for (int d = 0; d < 3; ++d) g[i][d] = Einv[i - 1][d];
                beta[0] -= beta[i];
                for (int d = 0; d < 3; ++d) g[0][d] -= g[i][d];
            }

            // B_ia = |det J_s| (beta_i alpha_a m0 + (beta_i h_a + alpha_a g_i) . m1 + g_i^T M2 h_a).
            const double scale = fabs(det_s);
            for (int i = 0; i < 4; ++i) {
                double M2g[3];
                for (int d = 0; d < 3; ++d) M2g[d] = M2[d][0] * g[i][0] + M2[d][1] * g[i][1] + M2[d][2] * g[i][2];
                for (int a = 0; a < 4; ++a) {
                    double v = beta[i] * alpha[a] * m[0];
                    for (int d = 0; d < 3; ++d) v += (beta[i] * h[a][d] + alpha[a] * g[i][d]) * m[1 + d] + M2g[d] * h[a][d];
                    B[i][a] = scale * v;
                }
            }
            return TET4_PAIR_CLIPPED;
        }

    }  // namespace intersect
}  // namespace sresample

#endif  // SRESAMPLE_TET4_INTERSECT_KERNEL_HPP
