// The exact HEX8-TET4 transfer end to end (search + intersect): a hex block inside a perturbed
// Kuhn fluid mesh, its nodes numbered at random (all the splits' patterns); element_boxes, the cell
// list, the mixed mass matrices and their application both ways, against identities:
//   affine block (parallelepipeds), TRILINEAR: sum_i W_i = |solid|, c_a = sum_e |H_e| / 8,
//     U = v W for a uniform v, g_a = int x psi_a for u(x) = x (closed form on parallelepipeds);
//   warped block, both bases: sum_i W_i = sum_e hex8_split_volumes, U = v W, sum_a c_a = sum_i W_i;
//     SPLIT: c_a and g_a the split tets' lumped volumes and int x psi_a in closed form.
// On the host, and on the device when sresample has CUDA.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sresample_base.hpp"

#ifdef SRESAMPLE_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include "sresample_cell_list.hpp"
#include "sresample_hex8_intersect_kernel.hpp"
#include "sresample_hex8_transfer.hpp"
#include "sresample_test.hpp"
#include "sresample_tet4_transfer.hpp"

using namespace sresample;

namespace {

    unsigned long long rng_state = 0x9E3779B97F4A7C15ULL;
    double             uniform() {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        return (double)(rng_state >> 11) * (1.0 / 9007199254740992.0);
    }

    struct Mesh {
        int       nxe{4};
        ptrdiff_t ne{0}, nn{0};
        int      *e[8]{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        double   *x[3]{nullptr, nullptr, nullptr};
        void      free_all() {
            for (int v = 0; v < nxe; ++v) free(e[v]);
            for (int d = 0; d < 3; ++d) free(x[d]);
        }
    };

    // The fluid: a Kuhn mesh of n^3 cubes over [0, 1]^3, inner nodes perturbed.
    Mesh kuhn(const int n, const double perturb) {
        Mesh      m;
        const int nn = n + 1;
        m.nxe        = 4;
        m.nn         = (ptrdiff_t)nn * nn * nn;
        m.ne         = (ptrdiff_t)6 * n * n * n;
        for (int d = 0; d < 3; ++d) m.x[d] = (double *)malloc(sizeof(double) * m.nn);
        for (int v = 0; v < 4; ++v) m.e[v] = (int *)malloc(sizeof(int) * m.ne);
        for (int k = 0; k < nn; ++k) {
            for (int j = 0; j < nn; ++j) {
                for (int i = 0; i < nn; ++i) {
                    const int idx[3] = {i, j, k};
                    for (int d = 0; d < 3; ++d) {
                        const bool inner                      = idx[d] > 0 && idx[d] < n;
                        m.x[d][(k * nn + j) * nn + i] = (idx[d] + (inner ? perturb * (uniform() - 0.5) : 0.0)) / n;
                    }
                }
            }
        }
        const int perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
        ptrdiff_t e           = 0;
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    for (int t = 0; t < 6; ++t) {
                        int pos[3] = {i, j, k};
                        m.e[0][e]  = (pos[2] * nn + pos[1]) * nn + pos[0];
                        for (int s = 0; s < 3; ++s) {
                            pos[perms[t][s]]++;
                            m.e[s + 1][e] = (pos[2] * nn + pos[1]) * nn + pos[0];
                        }
                        ++e;
                    }
                }
            }
        }
        return m;
    }

    // The solid: n^3 hexes, x = X0 + A p for p in [0, 1]^3 (affine) plus a smooth warp if warped,
    // the nodes numbered by a random permutation.
    const double X0[3]   = {0.22, 0.25, 0.2};
    const double A[3][3] = {{0.5, 0.06, -0.04}, {0.03, 0.45, 0.08}, {-0.05, 0.04, 0.55}};

    void map_point(const double p[3], const bool warped, double x[3]) {
        for (int d = 0; d < 3; ++d) x[d] = X0[d] + A[d][0] * p[0] + A[d][1] * p[1] + A[d][2] * p[2];
        if (warped) {
            x[0] += 0.03 * sin(3 * p[1] + 2 * p[2]);
            x[1] += 0.03 * p[0] * p[2];
            x[2] += 0.02 * cos(2 * p[0]);
        }
    }

    Mesh hex_block(const int n, const bool warped) {
        Mesh      m;
        const int nn = n + 1;
        m.nxe        = 8;
        m.nn         = (ptrdiff_t)nn * nn * nn;
        m.ne         = (ptrdiff_t)n * n * n;
        int *id      = (int *)malloc(sizeof(int) * m.nn);
        for (ptrdiff_t i = 0; i < m.nn; ++i) id[i] = (int)i;
        for (ptrdiff_t i = m.nn - 1; i > 0; --i) {
            const ptrdiff_t j = (ptrdiff_t)(uniform() * (double)(i + 1)) % (i + 1);
            const int       t = id[i];
            id[i]             = id[j];
            id[j]             = t;
        }
        for (int d = 0; d < 3; ++d) m.x[d] = (double *)malloc(sizeof(double) * m.nn);
        for (int v = 0; v < 8; ++v) m.e[v] = (int *)malloc(sizeof(int) * m.ne);
        for (int k = 0; k < nn; ++k) {
            for (int j = 0; j < nn; ++j) {
                for (int i = 0; i < nn; ++i) {
                    const double p[3] = {(double)i / n, (double)j / n, (double)k / n};
                    double       x[3];
                    map_point(p, warped, x);
                    for (int d = 0; d < 3; ++d) m.x[d][id[(k * nn + j) * nn + i]] = x[d];
                }
            }
        }
        const int corner[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
        ptrdiff_t e            = 0;
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    for (int c = 0; c < 8; ++c) {
                        m.e[c][e] = id[((k + corner[c][2]) * nn + j + corner[c][1]) * nn + i + corner[c][0]];
                    }
                    ++e;
                }
            }
        }
        free(id);
        return m;
    }

    struct MeshOn {
        MeshView<double, int> view;
        int                  *e[8]{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        double               *x[3]{nullptr, nullptr, nullptr};
        int const           **ep{nullptr};
        double const        **xp{nullptr};
        bool                  device{false};
        int                   nxe{4};
        void                  free_all() {
#ifdef SRESAMPLE_ENABLE_CUDA
            if (device) {
                for (int v = 0; v < nxe; ++v) cudaFree(e[v]);
                for (int d = 0; d < 3; ++d) cudaFree(x[d]);
                cudaFree(ep);
                cudaFree(xp);
                return;
            }
#endif
            free(ep);
            free(xp);
        }
    };

    MeshOn mesh_on(const Mesh &m, const bool device) {
        MeshOn o;
        o.device = device;
        o.nxe    = m.nxe;
        if (!device) {
            o.ep = (int const **)malloc(sizeof(int *) * m.nxe);
            o.xp = (double const **)malloc(sizeof(double *) * 3);
            for (int v = 0; v < m.nxe; ++v) o.ep[v] = m.e[v];
            for (int d = 0; d < 3; ++d) o.xp[d] = m.x[d];
        } else {
#ifdef SRESAMPLE_ENABLE_CUDA
            const int    *eh[8];
            const double *xh[3];
            for (int v = 0; v < m.nxe; ++v) {
                cudaMalloc(&o.e[v], sizeof(int) * m.ne);
                cudaMemcpy(o.e[v], m.e[v], sizeof(int) * m.ne, cudaMemcpyHostToDevice);
                eh[v] = o.e[v];
            }
            for (int d = 0; d < 3; ++d) {
                cudaMalloc(&o.x[d], sizeof(double) * m.nn);
                cudaMemcpy(o.x[d], m.x[d], sizeof(double) * m.nn, cudaMemcpyHostToDevice);
                xh[d] = o.x[d];
            }
            cudaMalloc(&o.ep, sizeof(int *) * m.nxe);
            cudaMalloc(&o.xp, sizeof(double *) * 3);
            cudaMemcpy(o.ep, eh, sizeof(int *) * m.nxe, cudaMemcpyHostToDevice);
            cudaMemcpy(o.xp, xh, sizeof(xh), cudaMemcpyHostToDevice);
#endif
        }
        o.view.type      = m.nxe == 8 ? ElemType::HEX8 : ElemType::TET4;
        o.view.nelements = m.ne;
        o.view.nnodes    = m.nn;
        o.view.elements  = o.ep;
        o.view.points    = o.xp;
        return o;
    }

    void *alloc(const bool device, const size_t bytes) {
#ifdef SRESAMPLE_ENABLE_CUDA
        if (device) {
            void *p = nullptr;
            cudaMalloc(&p, bytes);
            cudaMemset(p, 0, bytes);
            return p;
        }
#endif
        (void)device;
        return calloc(1, bytes);
    }
    void release(const bool device, void *p) {
#ifdef SRESAMPLE_ENABLE_CUDA
        if (device) {
            cudaFree(p);
            return;
        }
#endif
        (void)device;
        free(p);
    }
    void copy(const bool device, void *dst, const void *src, const size_t bytes, const bool to_host) {
#ifdef SRESAMPLE_ENABLE_CUDA
        if (device) {
            cudaMemcpy(dst, src, bytes, to_host ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice);
            return;
        }
#endif
        (void)device;
        (void)to_host;
        memcpy(dst, src, bytes);
    }

    int check(const bool device, const bool warped, const int basis) {
        const Exec exec  = device ? Exec::device() : Exec::host();
        Mesh       fluid = kuhn(9, 0.3);
        Mesh       solid = hex_block(3, warped);
        MeshOn     f = mesh_on(fluid, device), s = mesh_on(solid, device);

        const double origin[3] = {0.5, 0.5, 0.5};
        float       *fb[6], *sb[6];
        for (int k = 0; k < 6; ++k) {
            fb[k] = (float *)alloc(device, sizeof(float) * fluid.ne);
            sb[k] = (float *)alloc(device, sizeof(float) * solid.ne);
        }
        SRESAMPLE_TEST_EQ(search::element_boxes<double>(f.view, 0, fluid.ne, origin, fb, fb + 3, exec), 0);
        SRESAMPLE_TEST_EQ(search::element_boxes<double>(s.view, 0, solid.ne, origin, sb, sb + 3, exec), 0);
        search::Boxes fboxes, sboxes;
        fboxes.n = fluid.ne;
        sboxes.n = solid.ne;
        for (int d = 0; d < 3; ++d) {
            fboxes.lo[d] = fb[d];
            fboxes.hi[d] = fb[3 + d];
            sboxes.lo[d] = sb[d];
            sboxes.hi[d] = sb[3 + d];
        }
        search::CellList list;
        SRESAMPLE_TEST_EQ(search::cell_list_build(fboxes, search::CellListOptions(), &list, exec), 0);
        ptrdiff_t *ptr = (ptrdiff_t *)alloc(device, sizeof(ptrdiff_t) * (solid.ne + 1));
        SRESAMPLE_TEST_EQ(search::cell_list_count(list, sboxes, ptr, exec), 0);
        ptrdiff_t np = 0;
        SRESAMPLE_TEST_EQ(search::exclusive_scan(ptr, solid.ne, &np, exec), 0);
        int *ps = (int *)alloc(device, sizeof(int) * np), *pf = (int *)alloc(device, sizeof(int) * np);
        SRESAMPLE_TEST_EQ(search::cell_list_fill(list, sboxes, ptr, ps, pf, exec), 0);

        double                        *B = (double *)alloc(device, sizeof(double) * 32 * np);
        intersect::Hex8TransferOptions options;
        options.basis = basis;
        intersect::Tet4PairCounts counts;
        int                       max_it = 0;
        SRESAMPLE_TEST_EQ(intersect::hex8_mixed_mass<double>(s.view, f.view, np, ps, pf, B, options, &counts, &max_it, exec), 0);
        printf("  %s, %s, %s: %td pairs; split tets: %td separated, %td fluid inside, %td solid inside, %td clipped, %td overflow; "
               "Newton <= %d iterations\n",
               device ? "device" : "host", warped ? "warped" : "affine", basis ? "split" : "trilinear", np, counts.separated,
               counts.fluid_inside, counts.solid_inside, counts.clipped, counts.overflow, max_it);
        SRESAMPLE_TEST_EQ(counts.overflow, 0);
        SRESAMPLE_TEST_ASSERT(max_it < intersect::hex8_newton_max_it);

        const double vel[3] = {0.3, -1.2, 2.5};
        double      *v[3], *U[3], *u[3], *g[3];
        double      *W   = (double *)alloc(device, sizeof(double) * fluid.nn);
        double      *c   = (double *)alloc(device, sizeof(double) * solid.nn);
        double      *vol = (double *)alloc(device, sizeof(double) * solid.ne);
        double      *h   = (double *)malloc(sizeof(double) * (fluid.nn > solid.nn ? fluid.nn : solid.nn));
        for (int d = 0; d < 3; ++d) {
            v[d] = (double *)alloc(device, sizeof(double) * solid.nn);
            for (ptrdiff_t a = 0; a < solid.nn; ++a) h[a] = vel[d];
            copy(device, v[d], h, sizeof(double) * solid.nn, false);
            U[d] = (double *)alloc(device, sizeof(double) * fluid.nn);
            u[d] = (double *)alloc(device, sizeof(double) * fluid.nn);
            copy(device, u[d], fluid.x[d], sizeof(double) * fluid.nn, false);
            g[d] = (double *)alloc(device, sizeof(double) * solid.nn);
        }
        SRESAMPLE_TEST_EQ(intersect::hex8_mixed_mass_apply<double>(s.view, f.view, np, ps, pf, B, v, W, U, exec), 0);
        SRESAMPLE_TEST_EQ(intersect::hex8_mixed_mass_apply_transpose<double>(s.view, f.view, np, ps, pf, B, u, c, g, exec), 0);
        SRESAMPLE_TEST_EQ(intersect::hex8_split_volumes<double>(s.view, vol, exec), 0);
#ifdef SRESAMPLE_ENABLE_CUDA
        if (device) cudaDeviceSynchronize();
#endif

        double *hW = (double *)malloc(sizeof(double) * fluid.nn);
        double *hc = (double *)malloc(sizeof(double) * solid.nn);
        double *hv = (double *)malloc(sizeof(double) * solid.ne);
        copy(device, hW, W, sizeof(double) * fluid.nn, true);
        copy(device, hc, c, sizeof(double) * solid.nn, true);
        copy(device, hv, vol, sizeof(double) * solid.ne, true);
        double sumW = 0, sumc = 0, total = 0;
        for (ptrdiff_t i = 0; i < fluid.nn; ++i) sumW += hW[i];
        for (ptrdiff_t a = 0; a < solid.nn; ++a) sumc += hc[a];
        for (ptrdiff_t e = 0; e < solid.ne; ++e) total += hv[e];
        const double tol = 1e-13 * total;
        SRESAMPLE_TEST_APPROXEQ(sumW, total, tol);
        SRESAMPLE_TEST_APPROXEQ(sumc, total, tol);
        for (int d = 0; d < 3; ++d) {
            copy(device, h, U[d], sizeof(double) * fluid.nn, true);
            for (ptrdiff_t i = 0; i < fluid.nn; ++i) SRESAMPLE_TEST_APPROXEQ(h[i], vel[d] * hW[i], tol);
        }

        // The exact c and g, per element: closed forms for parallelepipeds (TRILINEAR), on the split
        // tets (SPLIT).
        double *ec = (double *)calloc(solid.nn, sizeof(double)), *eg[3];
        for (int d = 0; d < 3; ++d) eg[d] = (double *)calloc(solid.nn, sizeof(double));
        const bool exact = basis == 1 || !warped;
        if (exact) {
            for (ptrdiff_t e = 0; e < solid.ne; ++e) {
                double    xs[8][3];
                long long ids[8];
                for (int a = 0; a < 8; ++a) {
                    ids[a] = solid.e[a][e];
                    for (int d = 0; d < 3; ++d) xs[a][d] = solid.x[d][solid.e[a][e]];
                }
                if (basis == 1) {
                    int       t[6][4];
                    const int n = intersect::hex8_split(intersect::hex8_split_pattern(ids), t);
                    for (int k = 0; k < n; ++k) {
                        double ed[3][3];
                        for (int j = 0; j < 3; ++j) {
                            for (int d = 0; d < 3; ++d) ed[j][d] = xs[t[k][j + 1]][d] - xs[t[k][0]][d];
                        }
                        const double V = fabs(intersect::tet4_det3(ed[0], ed[1], ed[2])) / 6;
                        for (int a = 0; a < 4; ++a) {
                            const int node = solid.e[t[k][a]][e];
                            ec[node] += V / 4;
                            for (int d = 0; d < 3; ++d) {
                                double sum = xs[t[k][a]][d];
                                for (int b = 0; b < 4; ++b) sum += xs[t[k][b]][d];
                                eg[d][node] += V / 20 * sum;
                            }
                        }
                    }
                } else {
                    // x = x_0 + E s, E's columns the hex's edges from node 0; int N_a = 1/8,
                    // int s_d N_a = (corner_ad ? 1/3 : 1/6) / 4.
                    double E[3][3];
                    for (int d = 0; d < 3; ++d) {
                        E[d][0] = xs[1][d] - xs[0][d];
                        E[d][1] = xs[3][d] - xs[0][d];
                        E[d][2] = xs[4][d] - xs[0][d];
                    }
                    const double c0[3] = {E[0][0], E[1][0], E[2][0]}, c1[3] = {E[0][1], E[1][1], E[2][1]},
                                 c2[3] = {E[0][2], E[1][2], E[2][2]};
                    const double det = fabs(intersect::tet4_det3(c0, c1, c2));
                    for (int a = 0; a < 8; ++a) {
                        const int node = solid.e[a][e];
                        ec[node] += det / 8;
                        for (int d = 0; d < 3; ++d) {
                            double m = xs[0][d] / 8;
                            for (int q = 0; q < 3; ++q) m += E[d][q] * (intersect::hex8_corner(a, q) > 0.5 ? 1.0 / 3 : 1.0 / 6) / 4;
                            eg[d][node] += det * m;
                        }
                    }
                }
            }
            for (ptrdiff_t a = 0; a < solid.nn; ++a) SRESAMPLE_TEST_APPROXEQ(hc[a], ec[a], tol);
            for (int d = 0; d < 3; ++d) {
                copy(device, h, g[d], sizeof(double) * solid.nn, true);
                for (ptrdiff_t a = 0; a < solid.nn; ++a) SRESAMPLE_TEST_APPROXEQ(h[a], eg[d][a], tol);
            }
        }

        search::cell_list_destroy(&list);
        for (int k = 0; k < 6; ++k) {
            release(device, fb[k]);
            release(device, sb[k]);
        }
        for (int d = 0; d < 3; ++d) {
            release(device, v[d]);
            release(device, U[d]);
            release(device, u[d]);
            release(device, g[d]);
            free(eg[d]);
        }
        release(device, ptr);
        release(device, ps);
        release(device, pf);
        release(device, B);
        release(device, W);
        release(device, c);
        release(device, vol);
        free(h);
        free(hW);
        free(hc);
        free(hv);
        free(ec);
        f.free_all();
        s.free_all();
        fluid.free_all();
        solid.free_all();
        return 0;
    }

    int run(const bool device) {
        int err = 0;
        err += check(device, false, 0);
        err += check(device, true, 0);
        err += check(device, false, 1);
        err += check(device, true, 1);
        return err;
    }

    int test_host() { return run(false); }
#ifdef SRESAMPLE_ENABLE_CUDA
    int test_device() { return run(true); }
#endif

}  // namespace

int main() {
    int err = 0;
    SRESAMPLE_RUN_TEST(test_host);
#ifdef SRESAMPLE_ENABLE_CUDA
    SRESAMPLE_RUN_TEST(test_device);
#endif
    return err ? EXIT_FAILURE : EXIT_SUCCESS;
}
