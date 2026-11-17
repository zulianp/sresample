// HEX8 x HEX8 pairs dumped from an application (512 bytes a pair: the solid's and the fluid's nodes,
// then their global ids; a leading long long: the number of pairs), TRILINEAR / TRILINEAR, by
// convex pieces and split tet by split tet, on the host (OpenMP): the time of each, the largest
// |B - B_tets| / max |B_tets|, and every pair whose volume sum_ia B_ia differs by more than 1e-10
// of the solid hex's (those first 20 printed with their classes, all of them written to
// <file>.worst for a closer look). Not a test: run by hand (bench_hex8_pump_pairs <file>).

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sresample_hex8_fluid_kernel.hpp"
#include "sresample_hex8_pieces_kernel.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace sresample;
using namespace sresample::intersect;

namespace {
    double now() {
        struct timespec t;
        clock_gettime(CLOCK_MONOTONIC, &t);
        return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
    }

    struct Pair {
        double    xs[8][3];
        double    xf[8][3];
        long long ids_s[8];
        long long ids_f[8];
    };
}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <pairs file>\n", argv[0]);
        return 1;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    long long n = 0;
    if (fread(&n, sizeof(n), 1, f) != 1) return 1;
    Pair *p = (Pair *)malloc(sizeof(Pair) * n);
    if (fread(p, sizeof(Pair), n, f) != (size_t)n) return 1;
    fclose(f);
    double(*vb)[2]  = (double(*)[2])malloc(sizeof(double[2]) * n);
    double *dmax    = (double *)malloc(sizeof(double) * n);
    double  t[2]    = {0, 0};
    for (int tets = 0; tets < 2; ++tets) {
        const double t0 = now();
#pragma omp parallel for schedule(dynamic, 64)
        for (long long k = 0; k < n; ++k) {
            double b[8][8];
            int    c[5] = {0, 0, 0, 0, 0}, it = 0;
            if (tets) {
                hex8_fluid_mixed_mass_pair_tets(p[k].xs, p[k].ids_s, 8, p[k].xf, p[k].ids_f, b, Hex8Basis::TRILINEAR, Hex8Basis::TRILINEAR, c, &it);
            } else {
                hex8_fluid_mixed_mass_pair(p[k].xs, p[k].ids_s, 8, p[k].xf, p[k].ids_f, b, Hex8Basis::TRILINEAR, Hex8Basis::TRILINEAR, c, &it);
            }
            double v = 0;
            for (int i = 0; i < 8; ++i) {
                for (int a = 0; a < 8; ++a) v += b[i][a];
            }
            vb[k][tets] = v;
        }
        t[tets] = now() - t0;
    }
    // The solid hexes' split volumes, for the relative criterion.
    double sum_d = 0, sum_v = 0, worst = 0;
    long long bad = 0;
    FILE     *w   = nullptr;
    {
        char name[1024];
        snprintf(name, sizeof(name), "%s.worst", argv[1]);
        w = fopen(name, "wb");
    }
    for (long long k = 0; k < n; ++k) {
        const double vs = hex8_split_volume(p[k].xs, p[k].ids_s);
        const double d  = vb[k][0] - vb[k][1];
        sum_d += d;
        sum_v += vb[k][1];
        dmax[k] = fabs(d) / vs;
        worst   = fmax(worst, dmax[k]);
        if (dmax[k] > 1e-10) {
            if (bad < 20) {
                Hex8Map      ms, mf;
                Hex8PieceSet S, F;
                double       xs[8][3], xf[8][3], c[3] = {0, 0, 0};
                for (int a = 0; a < 8; ++a) {
                    for (int dd = 0; dd < 3; ++dd) c[dd] += p[k].xf[a][dd] / 8;
                }
                for (int a = 0; a < 8; ++a) {
                    for (int dd = 0; dd < 3; ++dd) {
                        xs[a][dd] = p[k].xs[a][dd] - c[dd];
                        xf[a][dd] = p[k].xf[a][dd] - c[dd];
                    }
                }
                hex8_map_init(xs, &ms);
                hex8_map_init(xf, &mf);
                const bool oks = hex8_piece_set_trilinear(xs, p[k].ids_s, &ms, &S);
                const bool okf = hex8_piece_set_trilinear(xf, p[k].ids_f, &mf, &F);
                int        cp[5] = {0, 0, 0, 0, 0}, ct[5] = {0, 0, 0, 0, 0}, it = 0;
                double     b[8][8];
                hex8_fluid_mixed_mass_pair(p[k].xs, p[k].ids_s, 8, p[k].xf, p[k].ids_f, b, Hex8Basis::TRILINEAR, Hex8Basis::TRILINEAR, cp, &it);
                hex8_fluid_mixed_mass_pair_tets(p[k].xs, p[k].ids_s, 8, p[k].xf, p[k].ids_f, b, Hex8Basis::TRILINEAR, Hex8Basis::TRILINEAR, ct, &it);
                printf("pair %lld: vol pieces %.15e tets %.15e (|d|/vol_s %.2e); solid pieces %d (ok %d, np %d), fluid pieces %d (ok %d, np %d); "
                       "classes pieces %d/%d/%d/%d/%d tets %d/%d/%d/%d/%d\n",
                       k, vb[k][0], vb[k][1], dmax[k], S.n, oks, S.K.np, F.n, okf, F.K.np, cp[0], cp[1], cp[2], cp[3], cp[4], ct[0], ct[1], ct[2],
                       ct[3], ct[4]);
            }
            if (w) fwrite(&p[k], sizeof(Pair), 1, w);
            bad++;
        }
    }
    if (w) fclose(w);
    printf("%lld pairs: pieces %.2f s, split tets %.2f s (%.1fx, %d threads); sum vol pieces - tets %.3e of %.6e (%.2e); "
           "%lld pairs differ by more than 1e-10 of their solid hex (worst %.2e)\n",
           n, t[0], t[1], t[1] / t[0],
#ifdef _OPENMP
           omp_get_max_threads(),
#else
           1,
#endif
           sum_d, sum_v, sum_d / sum_v, bad, worst);
    free(p);
    free(vb);
    free(dmax);
    return 0;
}
