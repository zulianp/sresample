// The cost of one pair of the exact transfer, on the host, one thread: tet4_mixed_mass_pair,
// hex8_mixed_mass_pair (HEX8 solid, TET4 fluid) and hex8_fluid_mixed_mass_pair (HEX8 fluid), in
// every basis combination, called directly on 2000 overlapping element pairs per case.
//
//   cube      unit cubes, the fluid shifted by up to 0.3 in each direction;
//   warped    the same, every node moved by up to 8 % of the edge;
//   prism     the pump's fluid hexes: a perturbed quad in (x, z) extruded along y (planar faces,
//             not a parallelepiped), against a warped solid;
//   fine      fluid prisms of a quarter of the solid's size, anywhere in the solid's box (the pump:
//             about 80 fluid hexes per solid hex).
//
// Prints the microseconds per pair of the pair functions (by convex pieces where they apply) and
// of the split-tet ones they fall back to (hex8_*_pair_tets, "tets"), the largest |B - B_tets|
// relative to max |B| on the same pairs, and the classes per pair. Not a test: built with the
// tests, run by hand (bench_hex8_pairs [pairs]).

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sresample_hex8_fluid_kernel.hpp"
#include "sresample_hex8_pieces_kernel.hpp"
#include "sresample_tet4_intersect_kernel.hpp"

using namespace sresample;
using namespace sresample::intersect;

namespace {

    unsigned long long rng_state = 0x9E3779B97F4A7C15ULL;
    double             uniform() {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        return (double)(rng_state >> 11) * (1.0 / 9007199254740992.0);
    }

    double now() {
        struct timespec t;
        clock_gettime(CLOCK_MONOTONIC, &t);
        return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
    }

    // A hex of edge h at origin o (Exodus order); every node moved by up to warp * h; prism: the
    // move is the same on both y levels and has no y component (vertical sides, flat levels).
    void make_hex(const double o[3], const double h, const double warp, const bool prism, double x[8][3]) {
        double m[4][3];
        for (int c = 0; c < 4; ++c) {
            for (int d = 0; d < 3; ++d) m[c][d] = warp * h * (2 * uniform() - 1);
        }
        for (int a = 0; a < 8; ++a) {
            for (int d = 0; d < 3; ++d) {
                double move = warp * h * (2 * uniform() - 1);
                if (prism) {
                    // Levels in y (d = 1): corners a and a + 4 share their (x, z) move.
                    move = d == 1 ? 0 : m[a % 4][d];
                }
                x[a][d] = o[d] + h * hex8_corner(a, d) + move;
            }
        }
    }

    void random_ids(long long ids[8]) {
        for (int a = 0; a < 8; ++a) ids[a] = a;
        for (int a = 7; a > 0; --a) {
            const int       b = (int)(uniform() * (a + 1)) % (a + 1);
            const long long t = ids[a];
            ids[a]            = ids[b];
            ids[b]            = t;
        }
    }

    struct Pairs {
        int        n{0};
        double (*xs)[8][3]{nullptr};
        double (*xf)[8][3]{nullptr};
        long long (*ids_s)[8]{nullptr};
        long long (*ids_f)[8]{nullptr};
    };

    void make_pairs(const char *kind, const int n, Pairs *p) {
        p->n     = n;
        p->xs    = (double(*)[8][3])malloc(sizeof(double[8][3]) * n);
        p->xf    = (double(*)[8][3])malloc(sizeof(double[8][3]) * n);
        p->ids_s = (long long(*)[8])malloc(sizeof(long long[8]) * n);
        p->ids_f = (long long(*)[8])malloc(sizeof(long long[8]) * n);
        const bool cube = !strcmp(kind, "cube"), fine = !strcmp(kind, "fine"), prism = !strcmp(kind, "prism");
        for (int k = 0; k < n; ++k) {
            const double o0[3] = {0, 0, 0};
            make_hex(o0, 1, cube ? 0 : 0.08, false, p->xs[k]);
            double o[3];
            if (fine) {
                for (int d = 0; d < 3; ++d) o[d] = -0.2 + 1.15 * uniform();
                make_hex(o, 0.25, 0.08, true, p->xf[k]);
            } else {
                for (int d = 0; d < 3; ++d) o[d] = 0.3 * (2 * uniform() - 1);
                make_hex(o, 1, cube ? 0 : 0.08, prism, p->xf[k]);
            }
            random_ids(p->ids_s[k]);
            random_ids(p->ids_f[k]);
        }
    }

    void free_pairs(Pairs *p) {
        free(p->xs);
        free(p->xf);
        free(p->ids_s);
        free(p->ids_f);
    }

    // The first split tet of each hex: the TET4xTET4 reference cost.
    void first_tet(const double x[8][3], const long long ids[8], double t[4][3]) {
        int tets[6][4];
        hex8_split(hex8_split_pattern(ids), tets);
        for (int v = 0; v < 4; ++v) {
            for (int d = 0; d < 3; ++d) t[v][d] = x[tets[0][v]][d];
        }
    }

    enum Variant { TET_TET, HEX_TET_SPLIT, HEX_TET_TRI, HEX_HEX_SS, HEX_HEX_TS, HEX_HEX_TT, N_VARIANTS };
    const char *variant_name[N_VARIANTS] = {"TET4xTET4 (closed form)", "HEX8xTET4, SPLIT",         "HEX8xTET4, TRILINEAR",
                                            "HEX8xHEX8, SPLIT/SPLIT",  "HEX8xHEX8, TRILINEAR/SPLIT", "HEX8xHEX8, TRILINEAR/TRILINEAR"};

    // Pair k of the variant into b (8 x 8, fluid rows), by pieces or (tets) split tet by split tet.
    void pair(const Pairs &p, const int k, const int variant, const bool tets, double b8[8][8], int counts[5]) {
        int it = 0;
        for (int i = 0; i < 8; ++i) {
            for (int a = 0; a < 8; ++a) b8[i][a] = 0;
        }
        if (variant == TET_TET) {
            double ts[4][3], tf[4][3], b[4][4];
            first_tet(p.xs[k], p.ids_s[k], ts);
            first_tet(p.xf[k], p.ids_f[k], tf);
            counts[tet4_mixed_mass_pair(ts, tf, b)]++;
            for (int i = 0; i < 4; ++i) {
                for (int a = 0; a < 4; ++a) b8[i][a] = b[i][a];
            }
        } else if (variant == HEX_TET_SPLIT || variant == HEX_TET_TRI) {
            double          tf[4][3], b[4][8];
            const Hex8Basis basis = variant == HEX_TET_SPLIT ? Hex8Basis::SPLIT : Hex8Basis::TRILINEAR;
            first_tet(p.xf[k], p.ids_f[k], tf);
            if (tets) {
                hex8_mixed_mass_pair_tets(p.xs[k], p.ids_s[k], tf, b, basis, counts, &it);
            } else {
                hex8_mixed_mass_pair(p.xs[k], p.ids_s[k], tf, b, basis, counts, &it);
            }
            for (int i = 0; i < 4; ++i) {
                for (int a = 0; a < 8; ++a) b8[i][a] = b[i][a];
            }
        } else {
            const Hex8Basis sb = variant == HEX_HEX_SS ? Hex8Basis::SPLIT : Hex8Basis::TRILINEAR;
            const Hex8Basis fb = variant == HEX_HEX_TT ? Hex8Basis::TRILINEAR : Hex8Basis::SPLIT;
            if (tets) {
                hex8_fluid_mixed_mass_pair_tets(p.xs[k], p.ids_s[k], 8, p.xf[k], p.ids_f[k], b8, sb, fb, counts, &it);
            } else {
                hex8_fluid_mixed_mass_pair(p.xs[k], p.ids_s[k], 8, p.xf[k], p.ids_f[k], b8, sb, fb, counts, &it);
            }
        }
    }

    // max |B - B_tets| / max |B_tets| over the pairs.
    double difference(const Pairs &p, const int variant) {
        double err = 0, scale = 0;
        for (int k = 0; k < p.n; ++k) {
            double b[8][8], r[8][8];
            int    c[5] = {0, 0, 0, 0, 0};
            pair(p, k, variant, false, b, c);
            pair(p, k, variant, true, r, c);
            for (int i = 0; i < 8; ++i) {
                for (int a = 0; a < 8; ++a) {
                    err   = fmax(err, fabs(b[i][a] - r[i][a]));
                    scale = fmax(scale, fabs(r[i][a]));
                }
            }
        }
        return scale > 0 ? err / scale : 0;
    }

    // One sweep over the pairs; returns a checksum (keeps the work alive) and adds the classes.
    double sweep(const Pairs &p, const int variant, const bool tets, long long classes[5]) {
        double sum = 0;
        for (int k = 0; k < p.n; ++k) {
            int    counts[5] = {0, 0, 0, 0, 0};
            double b[8][8];
            pair(p, k, variant, tets, b, counts);
            sum += b[0][0];
            for (int c = 0; c < 5; ++c) classes[c] += counts[c];
        }
        return sum;
    }

}  // namespace

int main(int argc, char **argv) {
    const int   n        = argc > 1 ? atoi(argv[1]) : 2000;
    const char *kinds[4] = {"cube", "warped", "prism", "fine"};
    double      t[2][N_VARIANTS][4], diff[N_VARIANTS][4];
    long long   cls[2][N_VARIANTS][4][5];
    double      check = 0;
    for (int c = 0; c < 4; ++c) {
        Pairs p;
        make_pairs(kinds[c], n, &p);
        for (int v = 0; v < N_VARIANTS; ++v) {
            diff[v][c] = difference(p, v);
            for (int tets = 0; tets < 2; ++tets) {
                long long scratch[5] = {0, 0, 0, 0, 0};
                check += sweep(p, v, tets, scratch);  // warm-up
                // Repeat until 0.2 s have passed.
                int          reps = 0;
                const double t0   = now();
                double       t1   = t0;
                while (t1 - t0 < 0.2) {
                    for (int q = 0; q < 5; ++q) scratch[q] = 0;
                    check += sweep(p, v, tets, scratch);
                    ++reps;
                    t1 = now();
                }
                for (int q = 0; q < 5; ++q) cls[tets][v][c][q] = scratch[q];
                t[tets][v][c] = 1e6 * (t1 - t0) / ((double)reps * n);
            }
        }
        free_pairs(&p);
    }
    printf("microseconds per pair, one thread, %d pairs per case: pieces (split tets)\n\n", n);
    printf("%-34s %20s %20s %20s %20s\n", "pair", kinds[0], kinds[1], kinds[2], kinds[3]);
    for (int v = 0; v < N_VARIANTS; ++v) {
        printf("%-34s", variant_name[v]);
        for (int c = 0; c < 4; ++c) printf(" %9.2f (%8.2f)", t[0][v][c], t[1][v][c]);
        printf("\n");
    }
    printf("\nmax |B - B_tets| / max |B_tets|\n");
    for (int v = 0; v < N_VARIANTS; ++v) {
        printf("%-34s", variant_name[v]);
        for (int c = 0; c < 4; ++c) printf(" %20.2e", diff[v][c]);
        printf("\n");
    }
    printf("\nclasses per pair, pieces | split tets (separated/fluid inside/solid inside/clipped/overflow)\n");
    for (int v = 0; v < N_VARIANTS; ++v) {
        printf("%-34s", variant_name[v]);
        for (int c = 0; c < 4; ++c) {
            printf("  %.1f/%.1f/%.1f/%.1f/%.0f | %.1f/%.1f/%.1f/%.1f/%.0f", (double)cls[0][v][c][0] / n, (double)cls[0][v][c][1] / n,
                   (double)cls[0][v][c][2] / n, (double)cls[0][v][c][3] / n, (double)cls[0][v][c][4], (double)cls[1][v][c][0] / n,
                   (double)cls[1][v][c][1] / n, (double)cls[1][v][c][2] / n, (double)cls[1][v][c][3] / n, (double)cls[1][v][c][4]);
        }
        printf("\n");
    }
    printf("\n(checksum %g)\n", check);
    return 0;
}
