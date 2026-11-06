// tet4_mixed_mass_pair (src/intersect/sresample_tet4_intersect_kernel.hpp): the exact mixed mass
// matrix of two intersecting tetrahedra, against closed forms, Monte Carlo and the partition of a
// solid tet by a fluid mesh that tiles it.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "sresample_test.hpp"
#include "sresample_tet4_intersect_kernel.hpp"

using namespace sresample::intersect;

namespace {

    const double unit_tet[4][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    double tet_volume(const double x[4][3]) {
        double e[3][3];
        for (int k = 0; k < 3; ++k) {
            for (int d = 0; d < 3; ++d) e[k][d] = x[k + 1][d] - x[0][d];
        }
        return fabs(tet4_det3(e[0], e[1], e[2])) / 6;
    }

    double sum_all(const double B[4][4]) {
        double s = 0;
        for (int i = 0; i < 4; ++i) {
            for (int a = 0; a < 4; ++a) s += B[i][a];
        }
        return s;
    }

    // A reproducible uniform number in [0, 1).
    unsigned long long rng_state = 0x9E3779B97F4A7C15ULL;
    double             uniform() {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        return (double)(rng_state >> 11) * (1.0 / 9007199254740992.0);
    }

    void random_tet(const double center[3], const double size, double x[4][3]) {
        for (;;) {
            for (int k = 0; k < 4; ++k) {
                for (int d = 0; d < 3; ++d) x[k][d] = center[d] + size * (uniform() - 0.5);
            }
            if (tet_volume(x) > 0.01 * size * size * size) return;
        }
    }

    // Barycentric coordinates of p in tet x.
    void barycentric(const double x[4][3], const double p[3], double l[4]) {
        double J[3][3], inv[3][3];
        for (int k = 0; k < 3; ++k) {
            for (int d = 0; d < 3; ++d) J[k][d] = x[k + 1][d] - x[0][d];
        }
        tet4_inverse3(J[0], J[1], J[2], inv);
        double r[3];
        for (int d = 0; d < 3; ++d) r[d] = p[d] - x[0][d];
        l[0] = 1;
        for (int k = 0; k < 3; ++k) {
            l[k + 1] = inv[k][0] * r[0] + inv[k][1] * r[1] + inv[k][2] * r[2];
            l[0] -= l[k + 1];
        }
    }

    int test_unit_moments() {
        ClipPolyhedron p;
        tet4_polyhedron(unit_tet, &p);
        double m[10];
        tet4_polyhedron_moments(&p, m);
        SRESAMPLE_TEST_APPROXEQ(m[0], 1.0 / 6, 1e-15);
        for (int d = 0; d < 3; ++d) SRESAMPLE_TEST_APPROXEQ(m[1 + d], 1.0 / 24, 1e-15);
        SRESAMPLE_TEST_APPROXEQ(m[4], 1.0 / 60, 1e-15);   // xx
        SRESAMPLE_TEST_APPROXEQ(m[5], 1.0 / 120, 1e-15);  // xy
        SRESAMPLE_TEST_APPROXEQ(m[9], 1.0 / 60, 1e-15);   // zz

        // Clipped by x <= 1/2: the corner tet at the origin minus nothing... the part x <= 1/2 of the
        // unit tet has volume 1/6 - (1/2)^3 / 6.
        const double n[3] = {-1, 0, 0};
        SRESAMPLE_TEST_ASSERT(tet4_clip_plane(&p, 0.5, n));
        tet4_polyhedron_moments(&p, m);
        SRESAMPLE_TEST_APPROXEQ(m[0], 1.0 / 6 - 0.125 / 6, 1e-15);
        return 0;
    }

    // Identical tets: B is the mass matrix, by the shortcut and by clipping.
    int test_identical() {
        double x[4][3];
        const double c[3] = {0.3, -0.2, 0.7};
        random_tet(c, 1.0, x);
        const double V = tet_volume(x);
        for (int force = 0; force < 2; ++force) {
            double B[4][4];
            // Round-off decides whether the shortcut or the clipping is taken: either is exact.
            const int cls = tet4_mixed_mass_pair(x, x, B, force == 1);
            SRESAMPLE_TEST_ASSERT(cls == TET4_PAIR_CLIPPED || cls == TET4_PAIR_FLUID_INSIDE || cls == TET4_PAIR_SOLID_INSIDE);
            for (int i = 0; i < 4; ++i) {
                for (int a = 0; a < 4; ++a) SRESAMPLE_TEST_APPROXEQ(B[i][a], V / 20 * (i == a ? 2 : 1), 1e-14 * V);
            }
        }
        return 0;
    }

    // The unit tet against its copy shifted by t along x: common volume (1 - t)^3 / 6.
    int test_translated() {
        for (int k = 1; k < 10; ++k) {
            const double t = k / 10.0;
            double       xf[4][3];
            for (int v = 0; v < 4; ++v) {
                for (int d = 0; d < 3; ++d) xf[v][d] = unit_tet[v][d] + (d == 0 ? t : 0);
            }
            double B[4][4];
            const int cls = tet4_mixed_mass_pair(unit_tet, xf, B);
            SRESAMPLE_TEST_ASSERT(cls == TET4_PAIR_CLIPPED);
            SRESAMPLE_TEST_APPROXEQ(sum_all(B), (1 - t) * (1 - t) * (1 - t) / 6, 1e-15);
        }
        return 0;
    }

    // Containment both ways: the shortcut equals the clipping.
    int test_containment() {
        double xs[4][3], xf[4][3];
        const double c[3] = {0.1, 0.2, 0.3};
        random_tet(c, 1.0, xs);
        // xf: a small tet at the centroid of xs.
        double g[3] = {0, 0, 0};
        for (int v = 0; v < 4; ++v) {
            for (int d = 0; d < 3; ++d) g[d] += xs[v][d] / 4;
        }
        for (int v = 0; v < 4; ++v) {
            for (int d = 0; d < 3; ++d) xf[v][d] = g[d] + 0.05 * (xs[v][d] - g[d]) + 0.01 * (uniform() - 0.5);
        }
        double B0[4][4], B1[4][4];
        SRESAMPLE_TEST_ASSERT(tet4_mixed_mass_pair(xs, xf, B0) == TET4_PAIR_FLUID_INSIDE);
        SRESAMPLE_TEST_ASSERT(tet4_mixed_mass_pair(xs, xf, B1, true) == TET4_PAIR_CLIPPED);
        const double scale = tet_volume(xf);
        for (int i = 0; i < 4; ++i) {
            for (int a = 0; a < 4; ++a) SRESAMPLE_TEST_APPROXEQ(B0[i][a], B1[i][a], 1e-12 * scale);
        }
        // Swapped roles: the solid inside the fluid.
        SRESAMPLE_TEST_ASSERT(tet4_mixed_mass_pair(xf, xs, B0) == TET4_PAIR_SOLID_INSIDE);
        SRESAMPLE_TEST_ASSERT(tet4_mixed_mass_pair(xf, xs, B1, true) == TET4_PAIR_CLIPPED);
        for (int i = 0; i < 4; ++i) {
            for (int a = 0; a < 4; ++a) SRESAMPLE_TEST_APPROXEQ(B0[i][a], B1[i][a], 1e-12 * scale);
        }
        return 0;
    }

    // Random overlapping pairs against Monte Carlo over the solid tet.
    int test_monte_carlo() {
        for (int trial = 0; trial < 5; ++trial) {
            double       xs[4][3], xf[4][3];
            const double c0[3] = {0, 0, 0}, c1[3] = {0.2 * (uniform() - 0.5), 0.2 * (uniform() - 0.5), 0.2 * (uniform() - 0.5)};
            random_tet(c0, 1.0, xs);
            random_tet(c1, 1.0, xf);
            double B[4][4];
            tet4_mixed_mass_pair(xs, xf, B, true);

            const int n = 2000000;
            double    mc[4][4] = {{0}};
            const double V = tet_volume(xs);
            for (int s = 0; s < n; ++s) {
                // A uniform point of xs: sorted uniforms give uniform barycentric coordinates.
                double u[3] = {uniform(), uniform(), uniform()};
                for (int a = 0; a < 3; ++a) {
                    for (int b = a + 1; b < 3; ++b) {
                        if (u[b] < u[a]) {
                            const double t = u[a];
                            u[a]           = u[b];
                            u[b]           = t;
                        }
                    }
                }
                const double l[4] = {u[0], u[1] - u[0], u[2] - u[1], 1 - u[2]};
                double       p[3] = {0, 0, 0};
                for (int a = 0; a < 4; ++a) {
                    for (int d = 0; d < 3; ++d) p[d] += l[a] * xs[a][d];
                }
                double lf[4];
                barycentric(xf, p, lf);
                if (lf[0] < 0 || lf[1] < 0 || lf[2] < 0 || lf[3] < 0) continue;
                for (int i = 0; i < 4; ++i) {
                    for (int a = 0; a < 4; ++a) mc[i][a] += lf[i] * l[a];
                }
            }
            for (int i = 0; i < 4; ++i) {
                for (int a = 0; a < 4; ++a) SRESAMPLE_TEST_APPROXEQ(B[i][a], V * mc[i][a] / n, 3e-3 * V);
            }
        }
        return 0;
    }

    // The unit cube tiled by n^3 Kuhn cubes (6 tets each), the inner nodes perturbed: for a solid
    // tet inside the cube, sum_f sum_i B^f_ia = int psi_a = V / 4 and
    // sum_f sum_i B^f_ia x_I(f, i) = int x psi_a = V / 20 (sum_b x_b + x_a), exactly.
    int test_partition() {
        const int n  = 4;
        const int nn = n + 1;
        double   *X  = (double *)malloc(sizeof(double) * 3 * nn * nn * nn);
        for (int k = 0; k < nn; ++k) {
            for (int j = 0; j < nn; ++j) {
                for (int i = 0; i < nn; ++i) {
                    const int    node  = (k * nn + j) * nn + i;
                    const int    idx[3] = {i, j, k};
                    for (int d = 0; d < 3; ++d) {
                        const bool inner = idx[d] > 0 && idx[d] < n;
                        X[3 * node + d]  = (idx[d] + (inner ? 0.3 * (uniform() - 0.5) : 0.0)) / n;
                    }
                }
            }
        }
        // Kuhn: the 6 paths from corner 0 to corner 7 of each cube.
        const int perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

        for (int trial = 0; trial < 4; ++trial) {
            double xs[4][3];
            const double c[3] = {0.5, 0.5, 0.5};
            random_tet(c, 0.7, xs);
            const double V = tet_volume(xs);

            double sum_psi[4] = {0, 0, 0, 0}, sum_xpsi[4][3] = {{0}};
            double sum_B[4][4] = {{0}};
            for (int k = 0; k < n; ++k) {
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < n; ++i) {
                        for (int t = 0; t < 6; ++t) {
                            int node[4];
                            int pos[3] = {i, j, k};
                            node[0]    = (pos[2] * nn + pos[1]) * nn + pos[0];
                            for (int s = 0; s < 3; ++s) {
                                pos[perms[t][s]]++;
                                node[s + 1] = (pos[2] * nn + pos[1]) * nn + pos[0];
                            }
                            double xf[4][3];
                            for (int v = 0; v < 4; ++v) {
                                for (int d = 0; d < 3; ++d) xf[v][d] = X[3 * node[v] + d];
                            }
                            double B[4][4];
                            const int cls = tet4_mixed_mass_pair(xs, xf, B);
                            SRESAMPLE_TEST_ASSERT(cls != TET4_PAIR_OVERFLOW);
                            for (int a = 0; a < 4; ++a) {
                                for (int v = 0; v < 4; ++v) {
                                    sum_psi[a] += B[v][a];
                                    for (int d = 0; d < 3; ++d) sum_xpsi[a][d] += B[v][a] * xf[v][d];
                                }
                                for (int v = 0; v < 4; ++v) sum_B[v][a] += B[v][a];
                            }
                        }
                    }
                }
            }
            for (int a = 0; a < 4; ++a) {
                SRESAMPLE_TEST_APPROXEQ(sum_psi[a], V / 4, 1e-13 * V);
                for (int d = 0; d < 3; ++d) {
                    double exact = xs[a][d];
                    for (int b = 0; b < 4; ++b) exact += xs[b][d];
                    exact *= V / 20;
                    SRESAMPLE_TEST_APPROXEQ(sum_xpsi[a][d], exact, 1e-13 * V);
                }
            }
        }
        free(X);
        return 0;
    }

    // The worst case for degeneracies: an unperturbed grid of 4^3 Kuhn cubes and a solid tet that
    // is a Kuhn tet of the 2x2x2 coarser grid, so its faces and vertices lie on fluid faces and
    // nodes everywhere: still sum_f B = M_s exactly, and no fluid tet outside it gets anything.
    int test_partition_aligned() {
        const int n = 4, nn = n + 1;
        const int perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
        for (int ts = 0; ts < 6; ++ts) {
            // The solid: Kuhn tet ts of the cube [0, 1/2]^3 scaled by 2: corner 0 to corner 7 of [0,1/2]^3... of size 1/2.
            double xs[4][3];
            int    pos[3] = {0, 0, 0};
            for (int d = 0; d < 3; ++d) xs[0][d] = 0.25;
            for (int s = 0; s < 3; ++s) {
                pos[perms[ts][s]]++;
                for (int d = 0; d < 3; ++d) xs[s + 1][d] = 0.25 + 0.5 * pos[d];
            }
            const double V = tet_volume(xs);
            double       sum_B[4] = {0, 0, 0, 0}, total = 0;
            for (int k = 0; k < n; ++k) {
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < n; ++i) {
                        for (int t = 0; t < 6; ++t) {
                            int    q[3] = {i, j, k};
                            double xf[4][3];
                            for (int d = 0; d < 3; ++d) xf[0][d] = (double)q[d] / n;
                            for (int s = 0; s < 3; ++s) {
                                q[perms[t][s]]++;
                                for (int d = 0; d < 3; ++d) xf[s + 1][d] = (double)q[d] / n;
                            }
                            for (int force = 0; force < 2; ++force) {
                                double    B[4][4];
                                const int cls = tet4_mixed_mass_pair(xs, xf, B, force == 1);
                                SRESAMPLE_TEST_ASSERT(cls != TET4_PAIR_OVERFLOW);
                                for (int a = 0; a < 4; ++a) {
                                    for (int v = 0; v < 4; ++v) {
                                        SRESAMPLE_TEST_ASSERT(B[v][a] == B[v][a]);
                                        if (!force) sum_B[a] += B[v][a];
                                        else total += B[v][a];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            (void)nn;
            for (int a = 0; a < 4; ++a) SRESAMPLE_TEST_APPROXEQ(sum_B[a], V / 4, 1e-14 * V);
            SRESAMPLE_TEST_APPROXEQ(total, V, 1e-14 * V);
        }
        return 0;
    }

    // Tets that touch: a shared face, a shared edge, a shared vertex, coplanar faces; no volume, no
    // NaN, and a tet against its face neighbour exactly zero.
    int test_touching() {
        // The unit tet's neighbour across the face x + y + z = 1.
        const double nb_face[4][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}};
        // Across the vertex (0,0,0): the mirror image.
        const double nb_vertex[4][3] = {{0, 0, 0}, {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}};
        // Sharing the edge x-axis segment: rotated to y < 0, z < 0.
        const double nb_edge[4][3] = {{0, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 0, -1}};
        // Coplanar faces z = 0, overlapping in the plane, on opposite sides.
        const double nb_coplanar[4][3] = {{0.1, 0.1, 0}, {0.6, 0.1, 0}, {0.1, 0.6, 0}, {0.2, 0.2, -1}};
        const double(*cases[4])[3] = {nb_face, nb_vertex, nb_edge, nb_coplanar};
        for (int c = 0; c < 4; ++c) {
            for (int force = 0; force < 2; ++force) {
                double B[4][4];
                tet4_mixed_mass_pair(unit_tet, cases[c], B, force == 1);
                for (int i = 0; i < 4; ++i) {
                    for (int a = 0; a < 4; ++a) {
                        SRESAMPLE_TEST_ASSERT(B[i][a] == B[i][a]);  // not NaN
                        SRESAMPLE_TEST_APPROXEQ(B[i][a], 0, 1e-16);
                    }
                }
            }
        }
        // Coplanar faces on the same side: the overlap is the part of the small tet inside.
        const double same_side[4][3] = {{0.1, 0.1, 0}, {0.3, 0.1, 0}, {0.1, 0.3, 0}, {0.15, 0.15, 0.2}};
        double       B[4][4];
        SRESAMPLE_TEST_ASSERT(tet4_mixed_mass_pair(unit_tet, same_side, B) == TET4_PAIR_FLUID_INSIDE);
        SRESAMPLE_TEST_APPROXEQ(sum_all(B), tet_volume(same_side), 1e-16);
        tet4_mixed_mass_pair(unit_tet, same_side, B, true);
        SRESAMPLE_TEST_APPROXEQ(sum_all(B), tet_volume(same_side), 1e-16);
        return 0;
    }

    // Scale invariance: the same configuration at 1e-5 of the size has 1e-15 of the volume.
    int test_scale() {
        double xs[4][3], xf[4][3], ys[4][3], yf[4][3];
        const double c0[3] = {0, 0, 0}, c1[3] = {0.1, 0.05, -0.05};
        random_tet(c0, 1.0, xs);
        random_tet(c1, 1.0, xf);
        const double s = 1e-5, o[3] = {0.012, -0.003, 0.02};  // metres, away from the origin
        for (int v = 0; v < 4; ++v) {
            for (int d = 0; d < 3; ++d) {
                ys[v][d] = o[d] + s * xs[v][d];
                yf[v][d] = o[d] + s * xf[v][d];
            }
        }
        double B0[4][4], B1[4][4];
        tet4_mixed_mass_pair(xs, xf, B0);
        tet4_mixed_mass_pair(ys, yf, B1);
        const double s3 = s * s * s;
        for (int i = 0; i < 4; ++i) {
            for (int a = 0; a < 4; ++a) SRESAMPLE_TEST_APPROXEQ(B1[i][a] / s3, B0[i][a], 1e-9 * fabs(sum_all(B0)));
        }
        return 0;
    }

}  // namespace

int main() {
    int err = 0;
    SRESAMPLE_RUN_TEST(test_unit_moments);
    SRESAMPLE_RUN_TEST(test_identical);
    SRESAMPLE_RUN_TEST(test_translated);
    SRESAMPLE_RUN_TEST(test_containment);
    SRESAMPLE_RUN_TEST(test_monte_carlo);
    SRESAMPLE_RUN_TEST(test_partition);
    SRESAMPLE_RUN_TEST(test_partition_aligned);
    SRESAMPLE_RUN_TEST(test_touching);
    SRESAMPLE_RUN_TEST(test_scale);
    return err ? EXIT_FAILURE : EXIT_SUCCESS;
}
