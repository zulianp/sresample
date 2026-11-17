#ifndef SRESAMPLE_HEX8_PIECES_KERNEL_HPP
#define SRESAMPLE_HEX8_PIECES_KERNEL_HPP

#include <math.h>

#include "sresample_base.hpp"
#include "sresample_hex8_intersect_kernel.hpp"
#include "sresample_hex8_split.hpp"
#include "sresample_quadrature.hpp"
#include "sresample_tet4_intersect_kernel.hpp"

// The pair of the HEX8 transfer by convex pieces, for the host loops and the device kernels alike.
//
// A hex's region is the union of its split tets (sresample_hex8_split.hpp), bounded by its faces
// each cut along the diagonal the split chose. Its TRILINEAR basis is one smooth function on that
// region: the split's inner faces do not matter to it, only the region. A non-planar face's two
// diagonals give a convex fold (the face bulges out: the hull of the hex's nodes there) or a
// reflex one (it dents in), and the two choices differ by exactly the tet of the face's four nodes.
// So the region is
//
//   K  the hex with every face cut along its convex diagonal (convex: checked, all nodes inside
//      all its planes), the intersection of 6 to 12 half-spaces,
//   minus, for every face whose split diagonal is the reflex one, the notch: the tet of its nodes
//      (inside K, against that face),
//
// and B = B(K) - sum B(notch) (the notches disjoint: checked, vol K - sum vol notch = vol split).
// For a parallelepiped or a prism of a convex quad (planar faces) K is the region; a warped hex
// has thin notches on about half its non-planar faces. A side's pieces, each with a sign:
//
//   a TET4 element                   itself, its P1 basis;
//   a hex in the SPLIT basis         its split tets, each with the P1 basis on its four hex nodes;
//   a hex in the TRILINEAR basis     K (+) and its notches (-), all with the trilinear basis; if K
//                                    is not convex or the notches overlap (a badly shaped hex) the
//                                    pair is integrated split tet by split tet
//                                    (hex8_mixed_mass_pair_tets, hex8_fluid_mixed_mass_pair_tets).
//
// A pair of pieces contributes with the product of their signs: (K_s - N_s) cap (K_f - N_f) =
// K_s cap K_f - N_s cap K_f - K_s cap N_f + N_s cap N_f, as N is inside K on each side.
//
// Each (solid piece, fluid piece) pair is separated, one piece inside the other (integrated over
// the inner piece's tets), or clipped: the fluid piece (if a tet), the solid piece (if a tet) or
// the boxes' intersection (two hexes) clipped by the other pieces' planes (r3d, as
// sresample_tet4_intersect_kernel.hpp), fanned from one of its vertices and integrated with the
// rule. The region is the same as split tet by split tet, so the coupled volume and the sums
// sum_a B_ia, sum_i B_ia are the same; B differs by the rule's error where a hex is not a
// parallelepiped (another fan of the same region), and not at all otherwise (both exact).
//
// For the pump (fluid hexes prisms of convex quads, solid hexes structured) a pair is one clip of
// K against K plus a few thin notches, instead of up to 36 clips of split tets.
//
// Coordinates are relative to the fluid element's centroid (no cancellation against far origins).

namespace sresample {
    namespace intersect {

        // A convex clip of a tet or a box by up to 24 planes (12 per hex). Clipped polyhedra have about
        // 20 vertices (20000 random pairs: 99th percentile 20 for fluid hexes a quarter of the solid's,
        // 30 for hexes of equal size, at most 44); one that outgrows the capacity (it counts the
        // vertices before a clip compacts) falls back to the split tets (hex8_*_pair_tets), so the
        // capacity bounds the cost and the stack (on the device), not the result.
        static constexpr int hex8_piece_max_vertices = 48;
        using Hex8PieceClip                          = ClipPolyhedronN<hex8_piece_max_vertices>;

        // The trilinear map x(s) = sum_a N_a(s) x_a, set up once per hex: the nodes relative to node
        // 0 and the inverse Jacobian at the centre, the affine start of Newton (exact for a
        // parallelepiped: 0 iterations).
        struct Hex8Map {
            double x0[3];
            double xr[8][3];
            double xc[3];        // x(1/2, 1/2, 1/2) - x0
            double Jcinv[3][3];  // (dx/ds)^-1 there: ds = Jcinv dx
        };

        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void hex8_map_init(const double xh[8][3], Hex8Map *const m) {
            for (int d = 0; d < 3; ++d) m->x0[d] = xh[0][d];
            for (int a = 0; a < 8; ++a) {
                for (int d = 0; d < 3; ++d) m->xr[a][d] = xh[a][d] - xh[0][d];
            }
            const double s[3] = {0.5, 0.5, 0.5};
            double       N[8], dN[8][3], c[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
            hex8_basis(s, N, dN);
            for (int d = 0; d < 3; ++d) m->xc[d] = 0;
            for (int a = 0; a < 8; ++a) {
                for (int d = 0; d < 3; ++d) {
                    m->xc[d] += N[a] * m->xr[a][d];
                    for (int e = 0; e < 3; ++e) c[e][d] += dN[a][e] * m->xr[a][d];
                }
            }
            tet4_inverse3(c[0], c[1], c[2], m->Jcinv);
        }

        // hex8_inverse on a Hex8Map, from the affine start: s with x(s) = x, and the basis there in N.
        // Converged when a step is below 1e-7: Newton converges quadratically, so the error left is
        // about the step squared (1e-14), and hex8_inverse's confirming step below 1e-13 is saved.
        // Returns the iterations (hex8_newton_max_it: not converged).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE int hex8_map_inverse(const Hex8Map &m, const double x[3], double s[3], double N[8]) {
            double t[3], dN[8][3];
            for (int d = 0; d < 3; ++d) t[d] = x[d] - m.x0[d];
            for (int e = 0; e < 3; ++e) {
                s[e] = 0.5 + m.Jcinv[e][0] * (t[0] - m.xc[0]) + m.Jcinv[e][1] * (t[1] - m.xc[1]) + m.Jcinv[e][2] * (t[2] - m.xc[2]);
            }
            int it = 0;
            for (; it < hex8_newton_max_it; ++it) {
                hex8_basis(s, N, dN);
                double r[3] = {-t[0], -t[1], -t[2]}, c[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
                for (int a = 0; a < 8; ++a) {
                    for (int d = 0; d < 3; ++d) {
                        r[d] += N[a] * m.xr[a][d];
                        for (int e = 0; e < 3; ++e) c[e][d] += dN[a][e] * m.xr[a][d];
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
                if (step < 1e-7) break;
            }
            hex8_basis(s, N, nullptr);
            return it;
        }

        // A convex piece: its vertices among its element's nodes xe (a tet: vi; K: all 8, and as
        // vertex 8 their average c), its planes (a + n . x >= 0 inside, n of unit length), the tets
        // that tile it (vertex numbers as above), and the basis on it. Indices, not copies: the
        // device keeps it on its stack.
        struct Hex8Piece {
            const double (*xe)[3]{nullptr};
            double        c[3];
            unsigned char nv{0};  // 4 or 8 (the vertices the plane tests use)
            unsigned char vi[4];
            unsigned char np{0};
            unsigned char nt{0};
            unsigned char tets[12][4];
            double        pa[12];
            double        pn[12][3];

            double         sign{1};           // -1 for a notch (subtracted from K)
            unsigned char  nodes{4};          // the element's nodes (the basis' length)
            bool           trilinear{false};  // the hex's trilinear basis (map), else the tet's P1 (vi)
            const Hex8Map *map{nullptr};
            double         Jinv[3][3];  // the tet's barycentric coordinates: l_{k+1} = Jinv[k] . (x - x_vi[0])
        };

        // Vertex v of the piece (8: K's node average).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE const double *hex8_piece_vertex(const Hex8Piece &p, const int v) {
            if (v == 8) return p.c;
            return p.xe[p.nv == 4 ? p.vi[v] : v];
        }

        // The plane through p, q, r, oriented so that inside is on its side: false if the triangle is
        // degenerate (no plane).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE bool hex8_piece_plane(const double p[3],
                                                                     const double q[3],
                                                                     const double r[3],
                                                                     const double inside[3],
                                                                     double      *a,
                                                                     double       n[3]) {
            const double u[3] = {q[0] - p[0], q[1] - p[1], q[2] - p[2]};
            const double v[3] = {r[0] - p[0], r[1] - p[1], r[2] - p[2]};
            n[0]              = u[1] * v[2] - u[2] * v[1];
            n[1]              = u[2] * v[0] - u[0] * v[2];
            n[2]              = u[0] * v[1] - u[1] * v[0];
            const double len  = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            const double uu   = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
            const double vv   = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            if (!(len > 1e-14 * (uu > vv ? uu : vv))) return false;
            for (int d = 0; d < 3; ++d) n[d] /= len;
            *a = -(n[0] * p[0] + n[1] * p[1] + n[2] * p[2]);
            if (*a + n[0] * inside[0] + n[1] * inside[1] + n[2] * inside[2] < 0) {
                *a = -*a;
                for (int d = 0; d < 3; ++d) n[d] = -n[d];
            }
            return true;
        }

        // A tet piece, the element's nodes vi[4] of xe: with map, a notch of that hex (its trilinear
        // basis, sign -1); without, the P1 basis on those nodes (nodes: the element's).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void hex8_piece_tet(const double (*xe)[3],
                                                                   const int      vi[4],
                                                                   const int      nodes,
                                                                   const Hex8Map *map,
                                                                   Hex8Piece *const p) {
            p->xe        = xe;
            p->nv        = 4;
            p->sign      = map ? -1 : 1;
            p->nodes     = (unsigned char)(map ? 8 : nodes);
            p->trilinear = map != nullptr;
            p->map       = map;
            double c[3]  = {0, 0, 0};
            for (int v = 0; v < 4; ++v) {
                p->vi[v] = (unsigned char)vi[v];
                for (int d = 0; d < 3; ++d) c[d] += xe[vi[v]][d] / 4;
            }
            p->nt = 1;
            for (int v = 0; v < 4; ++v) p->tets[0][v] = (unsigned char)v;
            constexpr int F[4][3] = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
            p->np                 = 0;
            for (int f = 0; f < 4; ++f) {
                if (hex8_piece_plane(xe[vi[F[f][0]]], xe[vi[F[f][1]]], xe[vi[F[f][2]]], c, &p->pa[p->np], p->pn[p->np])) p->np++;
            }
            if (map) return;
            double J[3][3];
            for (int d = 0; d < 3; ++d) {
                J[0][d] = xe[vi[1]][d] - xe[vi[0]][d];
                J[1][d] = xe[vi[2]][d] - xe[vi[0]][d];
                J[2][d] = xe[vi[3]][d] - xe[vi[0]][d];
            }
            tet4_inverse3(J[0], J[1], J[2], p->Jinv);
        }

        // |volume| of the tet (a, b, c, d).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE double hex8_tet_volume(const double a[3], const double b[3], const double c[3], const double d[3]) {
            const double e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
            const double e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
            const double e3[3] = {d[0] - a[0], d[1] - a[1], d[2] - a[2]};
            return fabs(tet4_det3(e1, e2, e3)) / 6;
        }

        // A hex in the TRILINEAR basis as its pieces: K, and the notches (their nodes in notch[k],
        // n_notch of them, subtracted). False if K is not convex, the notches overlap, or the ids
        // have no split: the pair is then integrated split tet by split tet.
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE bool hex8_piece_hex(const double    xh[8][3],
                                                                   const long long ids[8],
                                                                   const Hex8Map  *map,
                                                                   Hex8Piece *const K,
                                                                   int              notch[6][4],
                                                                   int             *n_notch) {
            const int pattern = hex8_split_pattern(ids);
            int       split[6][4];
            const int n_split = hex8_split(pattern, split);
            if (!n_split) return false;
            K->xe        = xh;
            K->nv        = 8;
            K->sign      = 1;
            K->nodes     = 8;
            K->trilinear = true;
            K->map       = map;
            double c[3]  = {0, 0, 0}, lo[3], hi[3];
            for (int d = 0; d < 3; ++d) lo[d] = hi[d] = xh[0][d];
            for (int a = 0; a < 8; ++a) {
                for (int d = 0; d < 3; ++d) {
                    c[d] += xh[a][d] / 8;
                    lo[d] = xh[a][d] < lo[d] ? xh[a][d] : lo[d];
                    hi[d] = xh[a][d] > hi[d] ? xh[a][d] : hi[d];
                }
            }
            for (int d = 0; d < 3; ++d) K->c[d] = c[d];
            const double h   = sqrt((hi[0] - lo[0]) * (hi[0] - lo[0]) + (hi[1] - lo[1]) * (hi[1] - lo[1]) + (hi[2] - lo[2]) * (hi[2] - lo[2]));
            const double tol = 1e-12 * h;

            // The faces (Exodus order, as hex8_split_pattern): diagonal a-c is the convex fold if d is
            // inside the plane of (a, b, c); planar faces (d within tol of it) give one plane and keep
            // the split's diagonal. K's pattern: every face cut along its convex diagonal.
            constexpr int F[6][4] = {{0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {0, 4, 7, 3}, {0, 3, 2, 1}, {4, 5, 6, 7}};
            int           kpattern = pattern;
            K->np                  = 0;
            *n_notch               = 0;
            for (int f = 0; f < 6; ++f) {
                const int a = F[f][0], b = F[f][1], cc = F[f][2], dd = F[f][3];
                double    pa, pn[3];
                if (!hex8_piece_plane(xh[a], xh[b], xh[cc], c, &pa, pn)) return false;
                const double s = pa + pn[0] * xh[dd][0] + pn[1] * xh[dd][1] + pn[2] * xh[dd][2];
                if (fabs(s) <= tol) {
                    K->pa[K->np] = pa;
                    for (int d = 0; d < 3; ++d) K->pn[K->np][d] = pn[d];
                    K->np++;
                    continue;
                }
                // The convex diagonal: a-c (bit clear) if d is inside, else b-d (bit set).
                const bool bd = s < 0;
                if (bd) {
                    if (!hex8_piece_plane(xh[a], xh[b], xh[dd], c, &K->pa[K->np], K->pn[K->np])) return false;
                    K->np++;
                    if (!hex8_piece_plane(xh[b], xh[cc], xh[dd], c, &K->pa[K->np], K->pn[K->np])) return false;
                    K->np++;
                } else {
                    K->pa[K->np] = pa;
                    for (int d = 0; d < 3; ++d) K->pn[K->np][d] = pn[d];
                    K->np++;
                    if (!hex8_piece_plane(xh[a], xh[cc], xh[dd], c, &K->pa[K->np], K->pn[K->np])) return false;
                    K->np++;
                }
                const bool split_bd = (pattern >> f) & 1;
                if (split_bd != bd) {
                    // The split's diagonal is the reflex one: the notch.
                    notch[*n_notch][0] = a;
                    notch[*n_notch][1] = b;
                    notch[*n_notch][2] = cc;
                    notch[*n_notch][3] = dd;
                    (*n_notch)++;
                }
                kpattern = bd ? (kpattern | (1 << f)) : (kpattern & ~(1 << f));
            }
            // K convex: every node inside every plane (within tol).
            for (int k = 0; k < K->np; ++k) {
                for (int v = 0; v < 8; ++v) {
                    if (K->pa[k] + K->pn[k][0] * xh[v][0] + K->pn[k][1] * xh[v][1] + K->pn[k][2] * xh[v][2] < -tol) return false;
                }
            }
            // K's tets: the split of its pattern, or (a pattern without one) the fan of its 12
            // triangles from the node average x[8]. The notches disjoint: vol K - sum vol notch is
            // the split's volume (overlapping notches would remove less than their sum).
            int       ktets[6][4];
            const int nk = hex8_split(kpattern, ktets);
            K->nt        = 0;
            if (nk) {
                for (int t = 0; t < nk; ++t) {
                    for (int v = 0; v < 4; ++v) K->tets[t][v] = (unsigned char)ktets[t][v];
                }
                K->nt = (unsigned char)nk;
            } else {
                for (int f = 0; f < 6; ++f) {
                    const int  a = F[f][0], b = F[f][1], cc = F[f][2], dd = F[f][3];
                    const bool bd = (kpattern >> f) & 1;
                    const int  t1[3] = {a, b, bd ? dd : cc}, t2[3] = {bd ? b : a, cc, dd};
                    for (int v = 0; v < 3; ++v) {
                        K->tets[K->nt][v]     = (unsigned char)t1[v];
                        K->tets[K->nt + 1][v] = (unsigned char)t2[v];
                    }
                    K->tets[K->nt][3]     = 8;
                    K->tets[K->nt + 1][3] = 8;
                    K->nt                 = (unsigned char)(K->nt + 2);
                }
            }
            double vk = 0, vn = 0, vs = 0;
            for (int t = 0; t < K->nt; ++t) {
                vk += hex8_tet_volume(hex8_piece_vertex(*K, K->tets[t][0]), hex8_piece_vertex(*K, K->tets[t][1]), hex8_piece_vertex(*K, K->tets[t][2]),
                                      hex8_piece_vertex(*K, K->tets[t][3]));
            }
            for (int t = 0; t < *n_notch; ++t) vn += hex8_tet_volume(xh[notch[t][0]], xh[notch[t][1]], xh[notch[t][2]], xh[notch[t][3]]);
            for (int t = 0; t < n_split; ++t) vs += hex8_tet_volume(xh[split[t][0]], xh[split[t][1]], xh[split[t][2]], xh[split[t][3]]);
            return fabs(vk - vn - vs) <= 1e-12 * vs;
        }

        // The basis of the piece's element at x into N[nodes]; returns Newton's iterations.
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE int hex8_piece_basis(const Hex8Piece &p, const double x[3], double N[8]) {
            if (p.trilinear) {
                double s[3];
                return hex8_map_inverse(*p.map, x, s, N);
            }
            const double *x0   = p.xe[p.vi[0]];
            const double  r[3] = {x[0] - x0[0], x[1] - x0[1], x[2] - x0[2]};
            double        l[4];
            for (int k = 0; k < 3; ++k) l[k + 1] = p.Jinv[k][0] * r[0] + p.Jinv[k][1] * r[1] + p.Jinv[k][2] * r[2];
            l[0] = 1 - l[1] - l[2] - l[3];
            for (int a = 0; a < p.nodes; ++a) N[a] = 0;
            for (int v = 0; v < 4; ++v) N[p.vi[v]] += l[v];
            return 0;
        }

        // The tets a pair of pieces is integrated over: their vertices x (a clipped polyhedron's, or
        // own: the inner piece's vertices), each tet four of them, and the pair's sign (the product of
        // the pieces'). A fan of at most hex8_piece_max_vertices vertices has fewer than twice as
        // many triangles. A fan's tets have signed volumes: a polyhedron cut by two nearly coincident
        // planes (a solid face's fold of 1e-9 of its size) has slightly negative ones, which cancel
        // against others (taken as |V| they added up to 4e-7 of a hex on the pump); a piece's own
        // tets, either orientation, |V|.
        static constexpr int hex8_tet_list_max = 2 * hex8_piece_max_vertices;

        struct Hex8TetList {
            const double (*x)[3]{nullptr};
            double        own[9][3];
            int           n{0};
            unsigned char t[hex8_tet_list_max][4];
            double        sign{1};
            bool          fan{false};  // the tets of a fan (signed volumes), else a piece's own (|V|)
        };

        // The rule: 24 points of degree 6 (six: both sides trilinear) or 14 of degree 5.
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE int hex8_rule_n(const bool six) { return six ? quadrature::tet24_n : quadrature::tet14_n; }

        // Point q of tet t of the list into x; returns its weight: the rule's times the tet's volume
        // (signed for a fan's tet, |V| for a piece's own) times the list's sign (0: no volume).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE double hex8_tet_list_point(const Hex8TetList &L, const int t, const int q, const bool six, double x[3]) {
            const double *v0 = L.x[L.t[t][0]], *v1 = L.x[L.t[t][1]], *v2 = L.x[L.t[t][2]], *v3 = L.x[L.t[t][3]];
            double        e0[3], e1[3], e2[3];
            for (int d = 0; d < 3; ++d) {
                e0[d] = v1[d] - v0[d];
                e1[d] = v2[d] - v0[d];
                e2[d] = v3[d] - v0[d];
            }
            const double det = tet4_det3(e0, e1, e2) / 6;
            const double V   = L.fan ? det : fabs(det);
            double       l[4], w;
            if (six) {
                quadrature::tet24_l(q, l);
                w = quadrature::tet24_w(q);
            } else {
                quadrature::tet14_l(q, l);
                w = quadrature::tet14_w(q);
            }
            for (int d = 0; d < 3; ++d) x[d] = v0[d] + l[1] * e0[d] + l[2] * e1[d] + l[3] * e2[d];
            return w * V * L.sign;
        }

        // The piece's own tets as the list.
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void hex8_tet_list_piece(const Hex8Piece &p, Hex8TetList *const L) {
            for (int v = 0; v < p.nv; ++v) {
                const double *x = hex8_piece_vertex(p, v);
                for (int d = 0; d < 3; ++d) L->own[v][d] = x[d];
            }
            if (p.nv == 8) {
                for (int d = 0; d < 3; ++d) L->own[8][d] = p.c[d];
            }
            L->x   = L->own;
            L->n   = p.nt;
            L->fan = false;
            for (int t = 0; t < p.nt; ++t) {
                for (int v = 0; v < 4; ++v) L->t[t][v] = p.tets[t][v];
            }
        }

        // A clipped polyhedron (convex) as the list: every face not through vertex 0, fanned from its
        // first vertex, a tet with vertex 0 (the faces through it would give tets of no volume). The
        // face walk is tet4_polyhedron_fan's. False if the list is full (not reached: see its bound).
        template <int N>
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE bool hex8_tet_list_fan(const ClipPolyhedronN<N> *const p, Hex8TetList *const L) {
            const int     nv = p->n;
            unsigned char mark[N][3];
            for (int v = 0; v < nv; ++v) mark[v][0] = mark[v][1] = mark[v][2] = 0;
            L->x   = p->x;
            L->n   = 0;
            L->fan = true;
            for (int vstart = 0; vstart < nv; ++vstart) {
                for (int kstart = 0; kstart < 3; ++kstart) {
                    if (mark[vstart][kstart]) continue;
                    // The face on the left of edge (vstart, nb[vstart][kstart]): its vertices.
                    unsigned char face[N];
                    int           nf    = 0;
                    int           vcur  = vstart;
                    int           k     = kstart;
                    int           guard = 0;
                    bool          apex  = false;
                    do {
                        mark[vcur][k]   = 1;
                        face[nf++]      = (unsigned char)vcur;
                        apex            = apex || vcur == 0;
                        const int vnext = p->nb[vcur][k];
                        int       j     = 0;
                        while (j < 3 && p->nb[vnext][j] != vcur) ++j;
                        vcur = vnext;
                        k    = (j + 1) % 3;
                    } while (vcur != vstart && nf < N && ++guard < 4 * N);
                    if (apex || nf < 3) continue;
                    // Triangles (face[0], face[t + 1], face[t]) in tet4_polyhedron_fan's order.
                    for (int t = 1; t + 1 < nf; ++t) {
                        if (L->n >= hex8_tet_list_max) return false;
                        L->t[L->n][0] = 0;
                        L->t[L->n][1] = face[0];
                        L->t[L->n][2] = face[t + 1];
                        L->t[L->n][3] = face[t];
                        L->n++;
                    }
                }
            }
            return true;
        }

        // Whether every vertex of q is outside one plane of p (q outside p).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE bool hex8_piece_outside(const Hex8Piece &p, const Hex8Piece &q) {
            for (int k = 0; k < p.np; ++k) {
                bool out = true;
                for (int v = 0; out && v < q.nv; ++v) {
                    const double *x = hex8_piece_vertex(q, v);
                    out             = p.pa[k] + p.pn[k][0] * x[0] + p.pn[k][1] * x[1] + p.pn[k][2] * x[2] <= 0;
                }
                if (out) return true;
            }
            return false;
        }

        // Whether every vertex of q is inside every plane of p (q inside p).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE bool hex8_piece_inside(const Hex8Piece &p, const Hex8Piece &q) {
            for (int k = 0; k < p.np; ++k) {
                for (int v = 0; v < q.nv; ++v) {
                    const double *x = hex8_piece_vertex(q, v);
                    if (p.pa[k] + p.pn[k][0] * x[0] + p.pn[k][1] * x[1] + p.pn[k][2] * x[2] < 0) return false;
                }
            }
            return true;
        }

        // The box [lo, hi] as the polyhedron (r3d_init_box's connectivity: vertex k's neighbours
        // counterclockwise seen from outside; corners in Exodus order).
        template <int N>
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void hex8_box_polyhedron(const double lo[3], const double hi[3], ClipPolyhedronN<N> *const p) {
            constexpr unsigned char nb[8][3] = {{1, 4, 3}, {2, 5, 0}, {3, 6, 1}, {0, 7, 2}, {7, 0, 5}, {4, 1, 6}, {5, 2, 7}, {6, 3, 4}};
            for (int a = 0; a < 8; ++a) {
                for (int d = 0; d < 3; ++d) p->x[a][d] = hex8_corner(a, d) > 0.5 ? hi[d] : lo[d];
                for (int k = 0; k < 3; ++k) p->nb[a][k] = nb[a][k];
            }
            p->n = 8;
        }

        // The pieces S (solid) and F (fluid) as the tets of S cap F, into L (with poly, the clip's
        // workspace). Returns the Tet4PairClass; TET4_PAIR_OVERFLOW if the clip outgrew its capacity
        // (the caller starts over split tet by split tet). L's sign is the caller's.
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE int hex8_piece_pair_tets(const Hex8Piece &S,
                                                                        const Hex8Piece &F,
                                                                        Hex8PieceClip *const poly,
                                                                        Hex8TetList *const   L) {
            L->n = 0;
            double lo[3], hi[3];
            for (int d = 0; d < 3; ++d) {
                double smin = 1e300, smax = -1e300, fmin = 1e300, fmax = -1e300;
                for (int v = 0; v < S.nv; ++v) {
                    const double x = hex8_piece_vertex(S, v)[d];
                    smin           = x < smin ? x : smin;
                    smax           = x > smax ? x : smax;
                }
                for (int v = 0; v < F.nv; ++v) {
                    const double x = hex8_piece_vertex(F, v)[d];
                    fmin           = x < fmin ? x : fmin;
                    fmax           = x > fmax ? x : fmax;
                }
                if (smax < fmin || fmax < smin) return TET4_PAIR_SEPARATED;
                lo[d] = smin > fmin ? smin : fmin;
                hi[d] = smax < fmax ? smax : fmax;
            }
            if (hex8_piece_outside(S, F) || hex8_piece_outside(F, S)) return TET4_PAIR_SEPARATED;
            if (hex8_piece_inside(S, F)) {
                hex8_tet_list_piece(F, L);
                return TET4_PAIR_FLUID_INSIDE;
            }
            if (hex8_piece_inside(F, S)) {
                hex8_tet_list_piece(S, L);
                return TET4_PAIR_SOLID_INSIDE;
            }

            // A tet piece is the start (exact), clipped by the other piece's planes; two hexes: the
            // boxes' intersection, clipped by both.
            const Hex8Piece *clip_by[2] = {nullptr, nullptr};
            if (F.nv == 4 || S.nv == 4) {
                const Hex8Piece &start = F.nv == 4 ? F : S;
                double           t[4][3];
                for (int v = 0; v < 4; ++v) {
                    for (int d = 0; d < 3; ++d) t[v][d] = hex8_piece_vertex(start, v)[d];
                }
                tet4_polyhedron(t, poly);
                clip_by[0] = F.nv == 4 ? &S : &F;
            } else {
                hex8_box_polyhedron(lo, hi, poly);
                clip_by[0] = &F;
                clip_by[1] = &S;
            }
            for (int c = 0; c < 2 && clip_by[c]; ++c) {
                const Hex8Piece &p = *clip_by[c];
                for (int k = 0; k < p.np && poly->n > 0; ++k) {
                    if (!tet4_clip_plane(poly, p.pa[k], p.pn[k])) return TET4_PAIR_OVERFLOW;
                }
            }
            if (poly->n == 0) return TET4_PAIR_SEPARATED;
            if (!hex8_tet_list_fan(poly, L)) return TET4_PAIR_OVERFLOW;
            return TET4_PAIR_CLIPPED;
        }

        // B[i][a] += w Nf_i Ns_a at the list's points (the host: one thread a pair).
        struct Hex8PieceAccumulator {
            const Hex8Piece *solid;
            const Hex8Piece *fluid;
            double (*B)[8];
            bool six;
            int  max_it{0};

            SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void point(const double x[3], const double w) {
                double    Ns[8], Nf[8];
                const int its = hex8_piece_basis(*solid, x, Ns);
                const int itf = hex8_piece_basis(*fluid, x, Nf);
                max_it        = its > max_it ? its : max_it;
                max_it        = itf > max_it ? itf : max_it;
                for (int i = 0; i < fluid->nodes; ++i) {
                    const double wi = w * Nf[i];
                    for (int a = 0; a < solid->nodes; ++a) B[i][a] += wi * Ns[a];
                }
            }

            SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void list(const Hex8TetList &L) {
                const int nq = hex8_rule_n(six);
                for (int t = 0; t < L.n; ++t) {
                    for (int q = 0; q < nq; ++q) {
                        double       x[3];
                        const double w = hex8_tet_list_point(L, t, q, six, x);
                        if (w != 0) point(x, w);
                    }
                }
            }
        };

        // One side of a pair as its pieces: piece 0 K and the notches after it (a TRILINEAR hex), or
        // tets (a SPLIT hex's split, a TET4 element): the element's nodes x (local coordinates).
        struct Hex8PieceSet {
            int            n{0};
            bool           trilinear{false};
            int            nodes{4};
            const double (*x)[3]{nullptr};
            const Hex8Map *map{nullptr};
            Hex8Piece      K;
            int            tets[6][4];  // the notches (trilinear) or the tets, element-local nodes

            // Piece k: K itself, or a tet built into scratch.
            SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE const Hex8Piece *piece(const int k, Hex8Piece *const scratch) const {
                if (trilinear && k == 0) return &K;
                hex8_piece_tet(x, tets[trilinear ? k - 1 : k], nodes, trilinear ? map : nullptr, scratch);
                return scratch;
            }
        };

        // A TRILINEAR hex as its pieces (false: K not convex, or no split).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE bool hex8_piece_set_trilinear(const double    x[8][3],
                                                                             const long long ids[8],
                                                                             const Hex8Map  *map,
                                                                             Hex8PieceSet   *set) {
            set->trilinear = true;
            set->nodes     = 8;
            set->x         = x;
            set->map       = map;
            int n_notch    = 0;
            if (!hex8_piece_hex(x, ids, map, &set->K, set->tets, &n_notch)) return false;
            set->n = 1 + n_notch;
            return true;
        }

        // A hex in the SPLIT basis (its split tets) or a TET4 element (nodes 4: itself).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE bool hex8_piece_set_tets(const double x[8][3], const long long ids[8], const int nodes, Hex8PieceSet *set) {
            set->trilinear = false;
            set->nodes     = nodes;
            set->x         = x;
            set->map       = nullptr;
            if (nodes == 4) {
                for (int v = 0; v < 4; ++v) set->tets[0][v] = v;
                set->n = 1;
                return true;
            }
            set->n = hex8_split(hex8_split_pattern(ids), set->tets);
            return set->n > 0;
        }

        // B[i][a] += over all (solid piece, fluid piece) pairs; counts per piece pair, Newton's most
        // iterations into *max_it. Returns false on an overflow (B partly accumulated).
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE bool hex8_piece_sets(const Hex8PieceSet &S,
                                                                    const Hex8PieceSet &F,
                                                                    double (*B)[8],
                                                                    const bool six,
                                                                    int        counts[5],
                                                                    int       *max_it) {
            Hex8Piece     ps, pf;
            Hex8PieceClip poly;
            Hex8TetList   L;
            for (int m = 0; m < F.n; ++m) {
                const Hex8Piece *fp = F.piece(m, &pf);
                for (int k = 0; k < S.n; ++k) {
                    const Hex8Piece *sp  = S.piece(k, &ps);
                    const int        cls = hex8_piece_pair_tets(*sp, *fp, &poly, &L);
                    if (cls == TET4_PAIR_OVERFLOW) return false;
                    counts[cls]++;
                    L.sign = sp->sign * fp->sign;
                    Hex8PieceAccumulator acc{sp, fp, B, six, 0};
                    acc.list(L);
                    *max_it = acc.max_it > *max_it ? acc.max_it : *max_it;
                }
            }
            return true;
        }

        // B[i][a] for the hex xs (ids: its split) and the fluid tet xf, in the basis asked for, as
        // hex8_mixed_mass_pair_tets (sresample_hex8_intersect_kernel.hpp), which it falls back to for
        // the SPLIT basis (closed form per split tet), a non-convex K, or an overflow. counts[c] +=
        // the piece pairs of Tet4PairClass c.
        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE int hex8_mixed_mass_pair(const double    xs[8][3],
                                                                        const long long ids[8],
                                                                        const double    xf[4][3],
                                                                        double          B[4][8],
                                                                        const Hex8Basis basis,
                                                                        int             counts[5],
                                                                        int            *max_it) {
            if (basis == Hex8Basis::SPLIT) return hex8_mixed_mass_pair_tets(xs, ids, xf, B, basis, counts, max_it);

            // Relative to the fluid tet's centroid.
            double c[3] = {0, 0, 0}, xsl[8][3], xfl[8][3];
            for (int v = 0; v < 4; ++v) {
                for (int d = 0; d < 3; ++d) c[d] += xf[v][d] / 4;
            }
            for (int v = 0; v < 8; ++v) {
                for (int d = 0; d < 3; ++d) {
                    xsl[v][d] = xs[v][d] - c[d];
                    xfl[v][d] = v < 4 ? xf[v][d] - c[d] : 0;
                }
            }
            Hex8Map      map;
            Hex8PieceSet S, F;
            hex8_map_init(xsl, &map);
            if (!hex8_piece_set_trilinear(xsl, ids, &map, &S)) return hex8_mixed_mass_pair_tets(xs, ids, xf, B, basis, counts, max_it);
            hex8_piece_set_tets(xfl, ids, 4, &F);

            for (int i = 0; i < 4; ++i) {
                for (int a = 0; a < 8; ++a) B[i][a] = 0;
            }
            int local[5] = {0, 0, 0, 0, 0}, most = 0;
            if (!hex8_piece_sets(S, F, B, false, local, &most)) return hex8_mixed_mass_pair_tets(xs, ids, xf, B, basis, counts, max_it);
            for (int q = 0; q < 5; ++q) counts[q] += local[q];
            *max_it = most;
            return 0;
        }

    }  // namespace intersect
}  // namespace sresample

#endif  // SRESAMPLE_HEX8_PIECES_KERNEL_HPP
