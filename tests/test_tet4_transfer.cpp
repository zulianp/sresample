// The exact TET4 transfer end to end (search + intersect): a deformed solid mesh inside a perturbed
// fluid mesh; tet4_boxes, the cell list, the mixed mass matrices and their application both ways,
// against identities that hold exactly:
//   sum_i W_i = |solid|, c_a = the solid's lumped volumes, U = v W for a uniform velocity v,
//   g_a = int x psi_a for the linear field u(x) = x (the fluid's P1 interpolates it exactly).
// On the host, and on the device when sresample has CUDA.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sresample_base.hpp"  // sresample_config.hpp: SRESAMPLE_ENABLE_CUDA

#ifdef SRESAMPLE_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include "sresample_cell_list.hpp"
#include "sresample_test.hpp"
#include "sresample_tet4_intersect_kernel.hpp"
#include "sresample_tet4_transfer.hpp"

using namespace sresample;

namespace {

    unsigned long long rng_state = 0x853C49E6748FEA9BULL;
    double             uniform() {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        return (double)(rng_state >> 11) * (1.0 / 9007199254740992.0);
    }

    // A Kuhn mesh of n^3 cubes over [x0, x0 + L]^3, inner nodes perturbed by `perturb` cells, mapped by
    // a smooth deformation if `deform`.
    struct Mesh {
        ptrdiff_t ne{0}, nn{0};
        int      *e[4]{nullptr, nullptr, nullptr, nullptr};
        double   *x[3]{nullptr, nullptr, nullptr};
        void      free_all() {
            for (int v = 0; v < 4; ++v) free(e[v]);
            for (int d = 0; d < 3; ++d) free(x[d]);
        }
    };

    Mesh kuhn(const int n, const double x0, const double L, const double perturb, const bool deform) {
        Mesh       m;
        const int  nn = n + 1;
        m.nn          = (ptrdiff_t)nn * nn * nn;
        m.ne          = (ptrdiff_t)6 * n * n * n;
        for (int d = 0; d < 3; ++d) m.x[d] = (double *)malloc(sizeof(double) * m.nn);
        for (int v = 0; v < 4; ++v) m.e[v] = (int *)malloc(sizeof(int) * m.ne);
        for (int k = 0; k < nn; ++k) {
            for (int j = 0; j < nn; ++j) {
                for (int i = 0; i < nn; ++i) {
                    const int node   = (k * nn + j) * nn + i;
                    const int idx[3] = {i, j, k};
                    double    p[3];
                    for (int d = 0; d < 3; ++d) {
                        const bool inner = idx[d] > 0 && idx[d] < n;
                        p[d]             = (idx[d] + (inner ? perturb * (uniform() - 0.5) : 0.0)) / n;
                    }
                    if (deform) {
                        const double q[3] = {p[0] + 0.1 * sin(2 * p[1]), p[1] + 0.1 * p[2] * p[0], p[2] - 0.05 * p[0]};
                        for (int d = 0; d < 3; ++d) p[d] = q[d];
                    }
                    for (int d = 0; d < 3; ++d) m.x[d][node] = x0 + L * p[d];
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

    double tet_volume(const Mesh &m, const ptrdiff_t e, double xs[4][3]) {
        for (int v = 0; v < 4; ++v) {
            for (int d = 0; d < 3; ++d) xs[v][d] = m.x[d][m.e[v][e]];
        }
        double J[3][3];
        for (int k = 0; k < 3; ++k) {
            for (int d = 0; d < 3; ++d) J[k][d] = xs[k + 1][d] - xs[0][d];
        }
        return fabs(intersect::tet4_det3(J[0], J[1], J[2])) / 6;
    }

    // Copies of a mesh's arrays (and of its arrays of pointers) in the memory of `device`.
    struct MeshOn {
        MeshView<double, int> view;
        bool                  device{false};
        int                  *e[4]{nullptr, nullptr, nullptr, nullptr};
        double               *x[3]{nullptr, nullptr, nullptr};
        int const           **ep{nullptr};
        double const        **xp{nullptr};
    };

    MeshOn mesh_on(const Mesh &m, const bool device) {
        MeshOn o;
        o.device = device;
        if (!device) {
            for (int v = 0; v < 4; ++v) o.e[v] = m.e[v];
            for (int d = 0; d < 3; ++d) o.x[d] = m.x[d];
            o.ep = (int const **)malloc(sizeof(int *) * 4);
            o.xp = (double const **)malloc(sizeof(double *) * 3);
            for (int v = 0; v < 4; ++v) o.ep[v] = o.e[v];
            for (int d = 0; d < 3; ++d) o.xp[d] = o.x[d];
        } else {
#ifdef SRESAMPLE_ENABLE_CUDA
            const int    *eh[4];
            const double *xh[3];
            for (int v = 0; v < 4; ++v) {
                cudaMalloc(&o.e[v], sizeof(int) * m.ne);
                cudaMemcpy(o.e[v], m.e[v], sizeof(int) * m.ne, cudaMemcpyHostToDevice);
                eh[v] = o.e[v];
            }
            for (int d = 0; d < 3; ++d) {
                cudaMalloc(&o.x[d], sizeof(double) * m.nn);
                cudaMemcpy(o.x[d], m.x[d], sizeof(double) * m.nn, cudaMemcpyHostToDevice);
                xh[d] = o.x[d];
            }
            cudaMalloc(&o.ep, sizeof(int *) * 4);
            cudaMalloc(&o.xp, sizeof(double *) * 3);
            cudaMemcpy(o.ep, eh, sizeof(eh), cudaMemcpyHostToDevice);
            cudaMemcpy(o.xp, xh, sizeof(xh), cudaMemcpyHostToDevice);
#endif
        }
        o.view.type      = ElemType::TET4;
        o.view.nelements = m.ne;
        o.view.nnodes    = m.nn;
        o.view.elements  = o.ep;
        o.view.points    = o.xp;
        return o;
    }

    // Arrays in host or device memory.
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
    void to_host(const bool device, void *dst, const void *src, const size_t bytes) {
#ifdef SRESAMPLE_ENABLE_CUDA
        if (device) {
            cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
            return;
        }
#endif
        (void)device;
        memcpy(dst, src, bytes);
    }
    void to_space(const bool device, void *dst, const void *src, const size_t bytes) {
#ifdef SRESAMPLE_ENABLE_CUDA
        if (device) {
            cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
            return;
        }
#endif
        (void)device;
        memcpy(dst, src, bytes);
    }

    int check(const bool device) {
        const Exec exec  = device ? Exec::device() : Exec::host();
        Mesh       fluid = kuhn(9, 0.0, 1.0, 0.3, false);   // [0, 1]^3
        Mesh       solid = kuhn(3, 0.2, 0.55, 0.3, true);   // deformed, inside it
        MeshOn     f = mesh_on(fluid, device), s = mesh_on(solid, device);

        // Broadphase: the fluid's boxes in a cell list, the solid's as queries.
        const double origin[3] = {0.5, 0.5, 0.5};
        float       *fb[6], *sb[6];
        for (int k = 0; k < 6; ++k) {
            fb[k] = (float *)alloc(device, sizeof(float) * fluid.ne);
            sb[k] = (float *)alloc(device, sizeof(float) * solid.ne);
        }
        SRESAMPLE_TEST_EQ(search::tet4_boxes<double>(f.view, 0, fluid.ne, origin, fb, fb + 3, exec), 0);
        SRESAMPLE_TEST_EQ(search::tet4_boxes<double>(s.view, 0, solid.ne, origin, sb, sb + 3, exec), 0);
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

        // The mixed mass matrices.
        double                    *B = (double *)alloc(device, sizeof(double) * 16 * np);
        intersect::Tet4PairCounts  counts;
        SRESAMPLE_TEST_EQ(intersect::tet4_mixed_mass<double>(s.view, f.view, np, ps, pf, B, &counts, exec), 0);
        printf("  %s: %td candidate pairs: %td separated, %td fluid inside, %td solid inside, %td clipped, %td overflow\n",
               device ? "device" : "host", np, counts.separated, counts.fluid_inside, counts.solid_inside, counts.clipped,
               counts.overflow);
        SRESAMPLE_TEST_EQ(counts.overflow, 0);

        // Fluid side with a uniform velocity, solid side with u(x) = x.
        const double vel[3] = {0.3, -1.2, 2.5};
        double      *v[3], *U[3], *u[3], *g[3];
        double      *W = (double *)alloc(device, sizeof(double) * fluid.nn);
        double      *c = (double *)alloc(device, sizeof(double) * solid.nn);
        double      *h = (double *)malloc(sizeof(double) * (fluid.nn > solid.nn ? fluid.nn : solid.nn));
        for (int d = 0; d < 3; ++d) {
            v[d] = (double *)alloc(device, sizeof(double) * solid.nn);
            for (ptrdiff_t a = 0; a < solid.nn; ++a) h[a] = vel[d];
            to_space(device, v[d], h, sizeof(double) * solid.nn);
            U[d] = (double *)alloc(device, sizeof(double) * fluid.nn);
            u[d] = (double *)alloc(device, sizeof(double) * fluid.nn);
            to_space(device, u[d], fluid.x[d], sizeof(double) * fluid.nn);
            g[d] = (double *)alloc(device, sizeof(double) * solid.nn);
        }
        SRESAMPLE_TEST_EQ(intersect::tet4_mixed_mass_apply<double>(s.view, f.view, np, ps, pf, B, v, W, U, exec), 0);
        SRESAMPLE_TEST_EQ(intersect::tet4_mixed_mass_apply_transpose<double>(s.view, f.view, np, ps, pf, B, u, c, g, exec), 0);
#ifdef SRESAMPLE_ENABLE_CUDA
        if (device) cudaDeviceSynchronize();
#endif

        // The exact values.
        double  vol = 0;
        double *lumped = (double *)calloc(solid.nn, sizeof(double));
        double *xpsi[3];
        for (int d = 0; d < 3; ++d) xpsi[d] = (double *)calloc(solid.nn, sizeof(double));
        for (ptrdiff_t e = 0; e < solid.ne; ++e) {
            double    xs[4][3];
            const double V = tet_volume(solid, e, xs);
            vol += V;
            for (int a = 0; a < 4; ++a) {
                lumped[solid.e[a][e]] += V / 4;
                for (int d = 0; d < 3; ++d) {
                    double sum = xs[a][d];
                    for (int b = 0; b < 4; ++b) sum += xs[b][d];
                    xpsi[d][solid.e[a][e]] += V / 20 * sum;
                }
            }
        }

        double *hW = (double *)malloc(sizeof(double) * fluid.nn);
        to_host(device, hW, W, sizeof(double) * fluid.nn);
        double sumW = 0;
        for (ptrdiff_t i = 0; i < fluid.nn; ++i) sumW += hW[i];
        SRESAMPLE_TEST_APPROXEQ(sumW, vol, 1e-13 * vol);
        for (int d = 0; d < 3; ++d) {
            to_host(device, h, U[d], sizeof(double) * fluid.nn);
            for (ptrdiff_t i = 0; i < fluid.nn; ++i) SRESAMPLE_TEST_APPROXEQ(h[i], vel[d] * hW[i], 1e-13 * vol);
        }
        to_host(device, h, c, sizeof(double) * solid.nn);
        for (ptrdiff_t a = 0; a < solid.nn; ++a) SRESAMPLE_TEST_APPROXEQ(h[a], lumped[a], 1e-13 * vol);
        for (int d = 0; d < 3; ++d) {
            to_host(device, h, g[d], sizeof(double) * solid.nn);
            for (ptrdiff_t a = 0; a < solid.nn; ++a) SRESAMPLE_TEST_APPROXEQ(h[a], xpsi[d][a], 1e-13 * vol);
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
            free(xpsi[d]);
        }
        release(device, ptr);
        release(device, ps);
        release(device, pf);
        release(device, B);
        release(device, W);
        release(device, c);
        free(h);
        free(hW);
        free(lumped);
        fluid.free_all();
        solid.free_all();
        return 0;
    }

    int test_host() { return check(false); }
#ifdef SRESAMPLE_ENABLE_CUDA
    int test_device() { return check(true); }
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
