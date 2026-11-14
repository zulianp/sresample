// The HEX8-TET4 intersection (sresample_hex8_intersect_kernel.hpp), on the host:
//   - the split (sresample_hex8_split.hpp) under all 8! numberings of the cube: positive tets of
//     volume 1, each face cut along the diagonal through its smallest node, inner triangles paired;
//   - Newton's inverse of a warped hex's trilinear map;
//   - a parallelepiped hex with the fluid tet inside it: TRILINEAR against a conical-product Gauss
//     rule (independent of the kernel's tet rule), exact for the polynomial phi_i psi_a;
//   - the containment branches against independent rules (a fluid tet inside a split tet, the hex
//     inside a fluid tet);
//   - a parallelepiped hex cut by fluid tets: the column sums against int psi_a = vol / 8;
//   - a warped hex cut by fluid tets: TRILINEAR's row sums equal SPLIT's (the same region), SPLIT
//     equals tet4_mixed_mass_pair on the split tets.

#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "sresample_hex8_intersect_kernel.hpp"
#include "sresample_quadrature.hpp"
#include "sresample_test.hpp"

using namespace sresample;
using namespace sresample::intersect;

namespace {

    unsigned long long rng_state = 0x2545F4914F6CDD1DULL;
    double             uniform() {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        return (double)(rng_state >> 11) * (1.0 / 9007199254740992.0);
    }

    const double CUBE[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    const int    FACES[6][4] = {{0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {0, 4, 7, 3}, {0, 3, 2, 1}, {4, 5, 6, 7}};

    double tet_det(const double a[3], const double b[3], const double c[3], const double d[3]) {
        double e[3][3];
        for (int k = 0; k < 3; ++k) {
            e[0][k] = b[k] - a[k];
            e[1][k] = c[k] - a[k];
            e[2][k] = d[k] - a[k];
        }
        return tet4_det3(e[0], e[1], e[2]);
    }

    int test_split_numberings() {
        int  perm[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        long n5 = 0, n6 = 0;
        do {
            long long ids[8];
            for (int a = 0; a < 8; ++a) ids[a] = perm[a];
            int       t[6][4];
            const int n = hex8_split(hex8_split_pattern(ids), t);
            SRESAMPLE_TEST_ASSERT(n == 5 || n == 6);
            n5 += n == 5;
            n6 += n == 6;
            double vol = 0;
            for (int k = 0; k < n; ++k) {
                const double d = tet_det(CUBE[t[k][0]], CUBE[t[k][1]], CUBE[t[k][2]], CUBE[t[k][3]]);
                SRESAMPLE_TEST_ASSERT(d > 0);
                vol += d / 6;
            }
            SRESAMPLE_TEST_APPROXEQ(vol, 1.0, 1e-15);
            // Triangles: on a face (2 per face, the diagonal through its smallest node) or paired.
            int tri[24][3], ntri = 0;
            for (int k = 0; k < n; ++k) {
                const int F[4][3] = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
                for (int f = 0; f < 4; ++f) {
                    int *q = tri[ntri++];
                    for (int j = 0; j < 3; ++j) q[j] = t[k][F[f][j]];
                    std::sort(q, q + 3);
                }
            }
            int outer = 0;
            for (int a = 0; a < ntri; ++a) {
                int same = 0;
                for (int b = 0; b < ntri; ++b) same += tri[a][0] == tri[b][0] && tri[a][1] == tri[b][1] && tri[a][2] == tri[b][2];
                SRESAMPLE_TEST_ASSERT(same == 1 || same == 2);
                if (same == 2) continue;
                ++outer;
                // An outer triangle: on a face, containing the face's diagonal through its smallest node.
                bool ok = false;
                for (int f = 0; f < 6 && !ok; ++f) {
                    int on = 0;
                    for (int j = 0; j < 3; ++j) {
                        for (int v = 0; v < 4; ++v) on += tri[a][j] == FACES[f][v];
                    }
                    if (on < 3) continue;
                    int lo = 0;
                    for (int v = 1; v < 4; ++v) {
                        if (ids[FACES[f][v]] < ids[FACES[f][lo]]) lo = v;
                    }
                    const int p = FACES[f][lo], r = FACES[f][(lo + 2) % 4];
                    bool      hp = false, hr = false;
                    for (int j = 0; j < 3; ++j) {
                        hp = hp || tri[a][j] == p;
                        hr = hr || tri[a][j] == r;
                    }
                    ok = hp && hr;
                }
                SRESAMPLE_TEST_ASSERT(ok);
            }
            SRESAMPLE_TEST_EQ(outer, 12);
        } while (std::next_permutation(perm, perm + 8));
        printf("  40320 numberings: %ld split in 5, %ld in 6\n", n5, n6);
        return 0;
    }

    void warped_hex(double xs[8][3], const double amp) {
        for (int a = 0; a < 8; ++a) {
            for (int d = 0; d < 3; ++d) xs[a][d] = 0.3 + 0.8 * CUBE[a][d] + amp * (uniform() - 0.5);
        }
    }

    int test_newton() {
        double xs[8][3];
        warped_hex(xs, 0.25);
        double worst = 0;
        for (int t = 0; t < 200; ++t) {
            const double s[3] = {uniform(), uniform(), uniform()};
            double       N[8], x[3] = {0, 0, 0};
            hex8_basis(s, N, nullptr);
            for (int a = 0; a < 8; ++a) {
                for (int d = 0; d < 3; ++d) x[d] += N[a] * xs[a][d];
            }
            double    r[3] = {0.5, 0.5, 0.5};
            const int it   = hex8_inverse(xs, x, r, N);
            SRESAMPLE_TEST_ASSERT(it < hex8_newton_max_it);
            for (int d = 0; d < 3; ++d) worst = fmax(worst, fabs(r[d] - s[d]));
        }
        SRESAMPLE_TEST_ASSERT(worst < 1e-12);

        // A hex of 0.1 mm at 1 cm from the origin (metres, as LDP24's meshes): converged, not
        // stalled on the coordinates' round-off.
        double small[8][3];
        for (int a = 0; a < 8; ++a) {
            for (int d = 0; d < 3; ++d) small[a][d] = (d == 0 ? -0.012 : 0.004) + 1e-4 * (xs[a][d] - 0.3) / 0.8;
        }
        for (int t = 0; t < 200; ++t) {
            const double s[3] = {uniform(), uniform(), uniform()};
            double       N[8], x[3] = {0, 0, 0};
            hex8_basis(s, N, nullptr);
            for (int a = 0; a < 8; ++a) {
                for (int d = 0; d < 3; ++d) x[d] += N[a] * small[a][d];
            }
            double    r[3] = {0.5, 0.5, 0.5};
            const int it   = hex8_inverse(small, x, r, N);
            SRESAMPLE_TEST_ASSERT(it < hex8_newton_max_it);
            for (int d = 0; d < 3; ++d) SRESAMPLE_TEST_ASSERT(fabs(r[d] - s[d]) < 1e-10);
        }
        return 0;
    }

    // A parallelepiped hex x = x0 + A s.
    void parallelepiped(double xs[8][3], double x0[3], double A[3][3]) {
        const double M[3][3] = {{0.9, 0.15, -0.1}, {0.05, 0.7, 0.2}, {-0.12, 0.1, 1.1}};
        for (int d = 0; d < 3; ++d) {
            x0[d] = 0.1 * (d + 1);
            for (int e = 0; e < 3; ++e) A[d][e] = M[d][e];
        }
        for (int a = 0; a < 8; ++a) {
            for (int d = 0; d < 3; ++d) xs[a][d] = x0[d] + A[d][0] * CUBE[a][0] + A[d][1] * CUBE[a][1] + A[d][2] * CUBE[a][2];
        }
    }

    // int_T f by the conical product of Gauss-Legendre rules (Duffy): independent of tet14.
    template <typename F>
    void duffy(const double v[4][3], F &&f) {
        const double vol6 = fabs(tet_det(v[0], v[1], v[2], v[3]));
        for (int i = 0; i < quadrature::line12_n; ++i) {
            for (int j = 0; j < quadrature::line12_n; ++j) {
                for (int k = 0; k < quadrature::line12_n; ++k) {
                    const double u1 = quadrature::line12_x(i), u2 = quadrature::line12_x(j), u3 = quadrature::line12_x(k);
                    const double w  = quadrature::line12_w(i) * quadrature::line12_w(j) * quadrature::line12_w(k) * u1 * u1 * u2 * vol6;
                    // Barycentric: l1 = u1 (1 - u2), l2 = u1 u2 (1 - u3), l3 = u1 u2 u3, l0 = 1 - u1.
                    const double l[4] = {1 - u1, u1 * (1 - u2), u1 * u2 * (1 - u3), u1 * u2 * u3};
                    double       x[3] = {0, 0, 0};
                    for (int a = 0; a < 4; ++a) {
                        for (int d = 0; d < 3; ++d) x[d] += l[a] * v[a][d];
                    }
                    f(x, l, w);
                }
            }
        }
    }

    int test_parallelepiped_inside() {
        double xs[8][3], x0[3], A[3][3];
        parallelepiped(xs, x0, A);
        long long ids[8] = {5, 2, 7, 0, 3, 6, 1, 4};
        // A fluid tet well inside: around the hex's centre.
        double xf[4][3];
        const double sf[4][3] = {{0.3, 0.3, 0.25}, {0.7, 0.35, 0.3}, {0.4, 0.75, 0.35}, {0.45, 0.4, 0.8}};
        for (int j = 0; j < 4; ++j) {
            for (int d = 0; d < 3; ++d) xf[j][d] = x0[d] + A[d][0] * sf[j][0] + A[d][1] * sf[j][1] + A[d][2] * sf[j][2];
        }
        double B[4][8], R[4][8] = {{0}};
        int    counts[5] = {0, 0, 0, 0, 0}, it = 0;
        SRESAMPLE_TEST_EQ(hex8_mixed_mass_pair(xs, ids, xf, B, Hex8Basis::TRILINEAR, counts, &it), 0);
        // The reference: s(x) = A^-1 (x - x0) is affine, psi_a(s) a trilinear polynomial.
        double Ainv[3][3];
        const double c0[3] = {A[0][0], A[1][0], A[2][0]}, c1[3] = {A[0][1], A[1][1], A[2][1]}, c2[3] = {A[0][2], A[1][2], A[2][2]};
        tet4_inverse3(c0, c1, c2, Ainv);
        duffy(xf, [&](const double x[3], const double l[4], const double w) {
            double s[3], N[8];
            for (int e = 0; e < 3; ++e) s[e] = Ainv[e][0] * (x[0] - x0[0]) + Ainv[e][1] * (x[1] - x0[1]) + Ainv[e][2] * (x[2] - x0[2]);
            hex8_basis(s, N, nullptr);
            for (int i = 0; i < 4; ++i) {
                for (int a = 0; a < 8; ++a) R[i][a] += w * l[i] * N[a];
            }
        });
        double scale = 0;
        for (int i = 0; i < 4; ++i) {
            for (int a = 0; a < 8; ++a) scale = fmax(scale, fabs(R[i][a]));
        }
        for (int i = 0; i < 4; ++i) {
            for (int a = 0; a < 8; ++a) SRESAMPLE_TEST_APPROXEQ(B[i][a], R[i][a], 1e-13 * scale);
        }
        return 0;
    }

    // The containment branches: a small fluid tet inside one split tet (FLUID_INSIDE), and a large
    // fluid tet around the whole hex (SOLID_INSIDE: B = int_hex phi_i psi_a, by the tensor Gauss rule
    // on the reference cube, exact on a parallelepiped).
    int test_containment() {
        double xs[8][3], x0[3], A[3][3];
        parallelepiped(xs, x0, A);
        long long ids[8] = {2, 6, 0, 4, 7, 1, 5, 3};
        int       t[6][4];
        hex8_split(hex8_split_pattern(ids), t);
        double c[3] = {0, 0, 0};
        for (int v = 0; v < 4; ++v) {
            for (int d = 0; d < 3; ++d) c[d] += xs[t[0][v]][d] / 4;
        }
        double small[4][3];
        for (int v = 0; v < 4; ++v) {
            for (int d = 0; d < 3; ++d) small[v][d] = c[d] + 0.05 * (xs[t[0][v]][d] - c[d]);
        }
        double B[4][8], R[4][8] = {{0}};
        int    counts[5] = {0, 0, 0, 0, 0}, it = 0;
        hex8_mixed_mass_pair(xs, ids, small, B, Hex8Basis::TRILINEAR, counts, &it);
        SRESAMPLE_TEST_EQ(counts[TET4_PAIR_FLUID_INSIDE], 1);
        double Ainv[3][3];
        const double c0[3] = {A[0][0], A[1][0], A[2][0]}, c1[3] = {A[0][1], A[1][1], A[2][1]}, c2[3] = {A[0][2], A[1][2], A[2][2]};
        const double det = fabs(tet4_inverse3(c0, c1, c2, Ainv));
        duffy(small, [&](const double x[3], const double l[4], const double w) {
            double s[3], N[8];
            for (int e = 0; e < 3; ++e) s[e] = Ainv[e][0] * (x[0] - x0[0]) + Ainv[e][1] * (x[1] - x0[1]) + Ainv[e][2] * (x[2] - x0[2]);
            hex8_basis(s, N, nullptr);
            for (int i = 0; i < 4; ++i) {
                for (int a = 0; a < 8; ++a) R[i][a] += w * l[i] * N[a];
            }
        });
        for (int i = 0; i < 4; ++i) {
            for (int a = 0; a < 8; ++a) SRESAMPLE_TEST_APPROXEQ(B[i][a], R[i][a], 1e-13 * fabs(R[0][0]) * 8);
        }

        const double big[4][3] = {{-3, -3, -3}, {21, -3, -3}, {-3, 21, -3}, {-3, -3, 21}};
        for (int k = 0; k < 5; ++k) counts[k] = 0;
        hex8_mixed_mass_pair(xs, ids, big, B, Hex8Basis::TRILINEAR, counts, &it);
        SRESAMPLE_TEST_EQ(counts[TET4_PAIR_SOLID_INSIDE], hex8_split(hex8_split_pattern(ids), t));
        // The big tet's barycentric coordinates: x = big0 + 24 (l1, l2, l3).
        double Q[4][8] = {{0}};
        for (int i = 0; i < quadrature::line12_n; ++i) {
            for (int j = 0; j < quadrature::line12_n; ++j) {
                for (int k = 0; k < quadrature::line12_n; ++k) {
                    const double s[3] = {quadrature::line12_x(i), quadrature::line12_x(j), quadrature::line12_x(k)};
                    const double w    = quadrature::line12_w(i) * quadrature::line12_w(j) * quadrature::line12_w(k) * det;
                    double       N[8], x[3];
                    hex8_basis(s, N, nullptr);
                    for (int d = 0; d < 3; ++d) x[d] = x0[d] + A[d][0] * s[0] + A[d][1] * s[1] + A[d][2] * s[2];
                    const double l[4] = {1 - (x[0] + x[1] + x[2] + 9) / 24, (x[0] + 3) / 24, (x[1] + 3) / 24, (x[2] + 3) / 24};
                    for (int ii = 0; ii < 4; ++ii) {
                        for (int a = 0; a < 8; ++a) Q[ii][a] += w * l[ii] * N[a];
                    }
                }
            }
        }
        for (int i = 0; i < 4; ++i) {
            for (int a = 0; a < 8; ++a) SRESAMPLE_TEST_APPROXEQ(B[i][a], Q[i][a], 1e-13 * det);
        }
        return 0;
    }

    // Fluid tets: a Kuhn split of n^3 cubes of [lo, hi]^3, perturbed.
    int kuhn_tets(const int n, const double lo, const double hi, double (*t)[4][3]) {
        const int perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
        int       m           = 0;
        const double h        = (hi - lo) / n;
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    for (int p = 0; p < 6; ++p) {
                        int pos[3] = {i, j, k};
                        for (int v = 0; v < 4; ++v) {
                            if (v) pos[perms[p][v - 1]]++;
                            for (int d = 0; d < 3; ++d) t[m][v][d] = lo + h * pos[d];
                        }
                        ++m;
                    }
                }
            }
        }
        return m;
    }

    int test_parallelepiped_cut() {
        double xs[8][3], x0[3], A[3][3];
        parallelepiped(xs, x0, A);
        long long ids[8] = {3, 1, 4, 7, 0, 2, 6, 5};
        static double t[6 * 125][4][3];
        const int     nt = kuhn_tets(5, -0.5, 2.0, t);
        double        col[8] = {0}, row_t = 0, row_s = 0;
        for (int e = 0; e < nt; ++e) {
            double B[4][8], S[4][8];
            int    counts[5] = {0, 0, 0, 0, 0}, it = 0;
            hex8_mixed_mass_pair(xs, ids, t[e], B, Hex8Basis::TRILINEAR, counts, &it);
            hex8_mixed_mass_pair(xs, ids, t[e], S, Hex8Basis::SPLIT, counts, &it);
            SRESAMPLE_TEST_EQ(counts[TET4_PAIR_OVERFLOW], 0);
            for (int i = 0; i < 4; ++i) {
                double rt = 0, rs = 0;
                for (int a = 0; a < 8; ++a) {
                    col[a] += B[i][a];
                    rt += B[i][a];
                    rs += S[i][a];
                }
                SRESAMPLE_TEST_APPROXEQ(rt, rs, 1e-14);
                row_t += rt;
                row_s += rs;
            }
        }
        const double vol = fabs(tet4_det3(A[0], A[1], A[2]));
        for (int a = 0; a < 8; ++a) SRESAMPLE_TEST_APPROXEQ(col[a], vol / 8, 1e-13 * vol);
        SRESAMPLE_TEST_APPROXEQ(row_t, vol, 1e-13 * vol);
        return 0;
    }

    // n fluid cells per side over [lo, hi]: 5 over 1.5 (tets smaller than the hex) or 2 over 2.1
    // (larger: their centroids often outside the clipped regions, the case the fan's apex is for).
    int warped_cut(const int nk, const double lo, const double hi) {
        double xs[8][3];
        warped_hex(xs, 0.2);
        long long ids[8] = {6, 0, 5, 2, 7, 3, 1, 4};
        static double t[6 * 125][4][3];
        const int     nt   = kuhn_tets(nk, lo, hi, t);
        double        sumt = 0;
        for (int e = 0; e < nt; ++e) {
            double B[4][8], S[4][8];
            int    counts[5] = {0, 0, 0, 0, 0}, it = 0;
            hex8_mixed_mass_pair(xs, ids, t[e], B, Hex8Basis::TRILINEAR, counts, &it);
            SRESAMPLE_TEST_ASSERT(it < hex8_newton_max_it);
            hex8_mixed_mass_pair(xs, ids, t[e], S, Hex8Basis::SPLIT, counts, &it);
            for (int i = 0; i < 4; ++i) {
                double rt = 0, rs = 0;
                for (int a = 0; a < 8; ++a) {
                    rt += B[i][a];
                    rs += S[i][a];
                }
                SRESAMPLE_TEST_APPROXEQ(rt, rs, 1e-14);
                sumt += rt;
            }
            // SPLIT is tet4_mixed_mass_pair on the split tets.
            int    tets[6][4];
            const int n = hex8_split(hex8_split_pattern(ids), tets);
            double R[4][8] = {{0}};
            for (int k = 0; k < n; ++k) {
                double xt[4][3], b[4][4];
                for (int v = 0; v < 4; ++v) {
                    for (int d = 0; d < 3; ++d) xt[v][d] = xs[tets[k][v]][d];
                }
                tet4_mixed_mass_pair(xt, t[e], b);
                for (int i = 0; i < 4; ++i) {
                    for (int v = 0; v < 4; ++v) R[i][tets[k][v]] += b[i][v];
                }
            }
            for (int i = 0; i < 4; ++i) {
                for (int a = 0; a < 8; ++a) SRESAMPLE_TEST_EQ(S[i][a], R[i][a]);
            }
        }
        SRESAMPLE_TEST_APPROXEQ(sumt, hex8_split_volume(xs, ids), 1e-13);
        return 0;
    }

    int test_warped_cut() { return warped_cut(5, 0.0, 1.5) + warped_cut(2, -0.3, 1.8); }

}  // namespace

int main() {
    int err = 0;
    SRESAMPLE_RUN_TEST(test_split_numberings);
    SRESAMPLE_RUN_TEST(test_newton);
    SRESAMPLE_RUN_TEST(test_parallelepiped_inside);
    SRESAMPLE_RUN_TEST(test_containment);
    SRESAMPLE_RUN_TEST(test_parallelepiped_cut);
    SRESAMPLE_RUN_TEST(test_warped_cut);
    return err ? EXIT_FAILURE : EXIT_SUCCESS;
}
