// The HEX8 transfer's B on the device: hex8_fluid_mixed_mass (HEX8 fluid) and hex8_mixed_mass (TET4
// fluid), TRILINEAR, by convex pieces and split tet by split tet (Hex8TransferOptions::pieces), on
// the pairs of bench_hex8_pairs.cpp (cube, warped, prism, fine), each pair its own two elements.
// Prints milliseconds per call, nanoseconds per pair, and the largest |B - B_tets| / max |B_tets|.
// Not a test: built with the tests, run by hand (bench_hex8_device [pairs]); needs CUDA.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sresample_base.hpp"

#ifdef SRESAMPLE_ENABLE_CUDA
#include <cuda_runtime.h>
#include <time.h>

#include "sresample_hex8_fluid.hpp"
#include "sresample_hex8_split.hpp"
#include "sresample_hex8_transfer.hpp"

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

    double corner(const int a, const int d) {
        const int c[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
        return c[a][d];
    }

    void make_hex(const double o[3], const double h, const double warp, const bool prism, double x[8][3]) {
        double m[4][3];
        for (int c = 0; c < 4; ++c) {
            for (int d = 0; d < 3; ++d) m[c][d] = warp * h * (2 * uniform() - 1);
        }
        for (int a = 0; a < 8; ++a) {
            for (int d = 0; d < 3; ++d) {
                const double move = prism ? (d == 1 ? 0 : m[a % 4][d]) : warp * h * (2 * uniform() - 1);
                x[a][d]           = o[d] + h * corner(a, d) + move;
            }
        }
    }

    // A mesh of n elements of nxe nodes each, its own nodes (on the device; ids too).
    struct DeviceMesh {
        int        nxe{8};
        ptrdiff_t  n{0};
        int       *e[8]{};
        double    *x[3]{};
        long long *ids{nullptr};
        int      **d_e{nullptr};
        double   **d_x{nullptr};

        void upload(const int nxe_, const ptrdiff_t n_, const double (*xh)[8][3], const long long (*idh)[8]) {
            nxe                 = nxe_;
            n                   = n_;
            const ptrdiff_t nn  = n * nxe;
            int            *he  = (int *)malloc(sizeof(int) * n);
            double         *hx  = (double *)malloc(sizeof(double) * nn);
            long long      *hid = (long long *)malloc(sizeof(long long) * nn);
            for (int v = 0; v < nxe; ++v) {
                for (ptrdiff_t k = 0; k < n; ++k) he[k] = (int)(k * nxe + v);
                cudaMalloc(&e[v], sizeof(int) * n);
                cudaMemcpy(e[v], he, sizeof(int) * n, cudaMemcpyHostToDevice);
            }
            for (int d = 0; d < 3; ++d) {
                for (ptrdiff_t k = 0; k < n; ++k) {
                    for (int v = 0; v < nxe; ++v) hx[k * nxe + v] = xh[k][v][d];
                }
                cudaMalloc(&x[d], sizeof(double) * nn);
                cudaMemcpy(x[d], hx, sizeof(double) * nn, cudaMemcpyHostToDevice);
            }
            for (ptrdiff_t k = 0; k < n; ++k) {
                for (int v = 0; v < nxe; ++v) hid[k * nxe + v] = k * 8 + idh[k][v];
            }
            cudaMalloc(&ids, sizeof(long long) * nn);
            cudaMemcpy(ids, hid, sizeof(long long) * nn, cudaMemcpyHostToDevice);
            cudaMalloc(&d_e, sizeof(int *) * 8);
            cudaMemcpy(d_e, e, sizeof(int *) * 8, cudaMemcpyHostToDevice);
            cudaMalloc(&d_x, sizeof(double *) * 3);
            cudaMemcpy(d_x, x, sizeof(double *) * 3, cudaMemcpyHostToDevice);
            free(he);
            free(hx);
            free(hid);
        }

        MeshView<double, int> view(const ElemType type) const {
            MeshView<double, int> m;
            m.type      = type;
            m.nelements = n;
            m.nnodes    = n * nxe;
            m.elements  = d_e;
            m.points    = d_x;
            m.ids       = ids;
            return m;
        }

        void release() {
            for (int v = 0; v < nxe; ++v) cudaFree(e[v]);
            for (int d = 0; d < 3; ++d) cudaFree(x[d]);
            cudaFree(ids);
            cudaFree(d_e);
            cudaFree(d_x);
        }
    };

    double now() {
        struct timespec t;
        clock_gettime(CLOCK_MONOTONIC, &t);
        return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
    }

}  // namespace

int main(int argc, char **argv) {
    const ptrdiff_t n        = argc > 1 ? atol(argv[1]) : 400000;
    const char     *kinds[4] = {"cube", "warped", "prism", "fine"};
    double (*xs)[8][3]       = (double(*)[8][3])malloc(sizeof(double[8][3]) * n);
    double (*xf)[8][3]       = (double(*)[8][3])malloc(sizeof(double[8][3]) * n);
    double (*xt)[8][3]       = (double(*)[8][3])malloc(sizeof(double[8][3]) * n);
    long long (*ids_s)[8]    = (long long(*)[8])malloc(sizeof(long long[8]) * n);
    long long (*ids_f)[8]    = (long long(*)[8])malloc(sizeof(long long[8]) * n);
    long long (*ids_t)[8]    = (long long(*)[8])malloc(sizeof(long long[8]) * n);
    int      *h_pairs        = (int *)malloc(sizeof(int) * n);
    double   *hb[2]          = {(double *)malloc(sizeof(double) * 64 * n), (double *)malloc(sizeof(double) * 64 * n)};
    for (ptrdiff_t k = 0; k < n; ++k) h_pairs[k] = (int)k;
    int *d_pairs = nullptr;
    cudaMalloc(&d_pairs, sizeof(int) * n);
    cudaMemcpy(d_pairs, h_pairs, sizeof(int) * n, cudaMemcpyHostToDevice);
    double *d_B = nullptr;
    cudaMalloc(&d_B, sizeof(double) * 64 * n);

    printf("%ld pairs per case, TRILINEAR, ms per call (ns per pair): pieces | split tets | speedup | max|dB|/max|B|\n", (long)n);
    for (int c = 0; c < 4; ++c) {
        const bool cube = !strcmp(kinds[c], "cube"), fine = !strcmp(kinds[c], "fine"), prism = !strcmp(kinds[c], "prism");
        for (ptrdiff_t k = 0; k < n; ++k) {
            const double o0[3] = {0, 0, 0};
            make_hex(o0, 1, cube ? 0 : 0.08, false, xs[k]);
            double o[3];
            if (fine) {
                for (int d = 0; d < 3; ++d) o[d] = -0.2 + 1.15 * uniform();
                make_hex(o, 0.25, 0.08, true, xf[k]);
            } else {
                for (int d = 0; d < 3; ++d) o[d] = 0.3 * (2 * uniform() - 1);
                make_hex(o, 1, cube ? 0 : 0.08, prism, xf[k]);
            }
            for (int a = 0; a < 8; ++a) ids_s[k][a] = ids_f[k][a] = a;
            for (int a = 7; a > 0; --a) {
                int       b = (int)(uniform() * (a + 1)) % (a + 1);
                long long t = ids_s[k][a];
                ids_s[k][a] = ids_s[k][b];
                ids_s[k][b] = t;
                b           = (int)(uniform() * (a + 1)) % (a + 1);
                t           = ids_f[k][a];
                ids_f[k][a] = ids_f[k][b];
                ids_f[k][b] = t;
            }
            // The TET4 fluid: the fluid hex's first split tet.
            int tets[6][4];
            hex8_split(hex8_split_pattern(ids_f[k]), tets);
            for (int v = 0; v < 4; ++v) {
                ids_t[k][v] = v;
                for (int d = 0; d < 3; ++d) xt[k][v][d] = xf[k][tets[0][v]][d];
            }
        }
        DeviceMesh S, F, T;
        S.upload(8, n, xs, ids_s);
        F.upload(8, n, xf, ids_f);
        T.upload(4, n, xt, ids_t);
        for (int fluid_tet = 0; fluid_tet < 2; ++fluid_tet) {
            double t[2];
            for (int tets = 0; tets < 2; ++tets) {
                Hex8TransferOptions opt;
                opt.pieces = tets ? 0 : 2;
                Tet4PairCounts counts;
                int            max_it = 0;
                double         best   = 1e300;
                for (int rep = 0; rep < 3; ++rep) {
                    cudaDeviceSynchronize();
                    const double t0 = now();
                    const int    rc = fluid_tet ? hex8_mixed_mass<double>(S.view(ElemType::HEX8), T.view(ElemType::TET4), n, d_pairs, d_pairs, d_B,
                                                                       opt, &counts, &max_it, Exec::device())
                                                : hex8_fluid_mixed_mass<double>(S.view(ElemType::HEX8), F.view(ElemType::HEX8), n, d_pairs,
                                                                             d_pairs, d_B, opt, &counts, &max_it, Exec::device());
                    cudaDeviceSynchronize();
                    best = fmin(best, now() - t0);
                    if (rc) printf("  call failed (%d)\n", rc);
                }
                t[tets] = best;
                cudaMemcpy(hb[tets], d_B, sizeof(double) * (fluid_tet ? 32 : 64) * n, cudaMemcpyDeviceToHost);
            }
            double err = 0, scale = 0;
            for (ptrdiff_t q = 0; q < (fluid_tet ? 32 : 64) * n; ++q) {
                err   = fmax(err, fabs(hb[0][q] - hb[1][q]));
                scale = fmax(scale, fabs(hb[1][q]));
            }
            printf("  %-7s %-11s %9.2f (%7.1f) | %9.2f (%7.1f) | %5.1fx | %.2e\n", kinds[c], fluid_tet ? "HEX8xTET4" : "HEX8xHEX8", 1e3 * t[0],
                   1e9 * t[0] / n, 1e3 * t[1], 1e9 * t[1] / n, t[1] / t[0], err / scale);
        }
        S.release();
        F.release();
        T.release();
    }
    cudaFree(d_pairs);
    cudaFree(d_B);
    return 0;
}

#else

int main() {
    printf("bench_hex8_device: sresample was built without CUDA\n");
    return 0;
}

#endif
