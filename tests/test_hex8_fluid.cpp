// The exact transfer with a HEX8 fluid (sresample_hex8_fluid.hpp).
//
// One pair, both hexes parallelepipeds and TRILINEAR, one inside the other (both containment
// branches): B against a 4^3 tensor Gauss rule on the inner hex (exact: the integrand is a cubic in
// each reference coordinate). The degree-6 tet rule is exact here; the degree-5 one is not.
//
// End to end (search + intersect): a hex fluid over [0, 1]^3 (inner nodes perturbed: warped hexes;
// nodes numbered at random, with global ids given through MeshView::ids, another random
// numbering), and a solid inside it, a HEX8 block (affine or warped) or a TET4 (Kuhn) block, in
// every basis combination, against identities:
//   sum_i W_i = sum_a c_a = |split(solid)| (the fluid's split tets tile the cube);
//   U = v W for a uniform v;
//   u = x at the fluid nodes: sum_a g_a = int_solid x (both bases reproduce linear fields);
//   v = x at the solid nodes: sum_i U_i = int_solid x;
//   duality: sum_i u_i U_i = sum_a v_a g_a for random u, v.
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
#include "sresample_hex8_fluid.hpp"
#include "sresample_hex8_fluid_kernel.hpp"
#include "sresample_test.hpp"

using namespace sresample;

namespace {

    unsigned long long rng_state = 0x2545F4914F6CDD1DULL;
    double             uniform() {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        return (double)(rng_state >> 11) * (1.0 / 9007199254740992.0);
    }

    void permutation(const ptrdiff_t n, long long *p) {
        for (ptrdiff_t i = 0; i < n; ++i) p[i] = i;
        for (ptrdiff_t i = n - 1; i > 0; --i) {
            const ptrdiff_t j = (ptrdiff_t)(uniform() * (double)(i + 1)) % (i + 1);
            const long long t = p[i];
            p[i]              = p[j];
            p[j]              = t;
        }
    }

    // ---- one pair -------------------------------------------------------------------------

    // x = x0 + E s, s in [0, 1]^3 (E's columns the edges), corners in Exodus order.
    void parallelepiped(const double x0[3], const double E[3][3], double x[8][3]) {
        for (int a = 0; a < 8; ++a) {
            for (int d = 0; d < 3; ++d) {
                x[a][d] = x0[d];
                for (int q = 0; q < 3; ++q) x[a][d] += E[d][q] * intersect::hex8_corner(a, q);
            }
        }
    }

    // B_ia = int_inner N^f_i N^s_a by a 4^3 Gauss rule on the inner hex; both affine.
    void reference(const double xf[8][3], const double xs[8][3], const bool fluid_inner, double B[8][8]) {
        const double g[4] = {0.0694318442029737, 0.3300094782075719, 0.6699905217924281, 0.9305681557970263};
        const double w[4] = {0.1739274225687269, 0.3260725774312731, 0.3260725774312731, 0.1739274225687269};
        const double(*xi)[3] = fluid_inner ? xf : xs;
        const double(*xo)[3] = fluid_inner ? xs : xf;
        double Ei[3][3], Eo[3][3], Eoinv[3][3];
        for (int d = 0; d < 3; ++d) {
            Ei[d][0] = xi[1][d] - xi[0][d];
            Ei[d][1] = xi[3][d] - xi[0][d];
            Ei[d][2] = xi[4][d] - xi[0][d];
            Eo[d][0] = xo[1][d] - xo[0][d];
            Eo[d][1] = xo[3][d] - xo[0][d];
            Eo[d][2] = xo[4][d] - xo[0][d];
        }
        const double ci0[3] = {Ei[0][0], Ei[1][0], Ei[2][0]}, ci1[3] = {Ei[0][1], Ei[1][1], Ei[2][1]}, ci2[3] = {Ei[0][2], Ei[1][2], Ei[2][2]};
        const double det = fabs(intersect::tet4_det3(ci0, ci1, ci2));
        // Eoinv = Eo^-1 (tet4_inverse3 takes the columns of the matrix).
        const double co0[3] = {Eo[0][0], Eo[1][0], Eo[2][0]}, co1[3] = {Eo[0][1], Eo[1][1], Eo[2][1]}, co2[3] = {Eo[0][2], Eo[1][2], Eo[2][2]};
        intersect::tet4_inverse3(co0, co1, co2, Eoinv);
        for (int i = 0; i < 8; ++i) {
            for (int a = 0; a < 8; ++a) B[i][a] = 0;
        }
        for (int p = 0; p < 4; ++p) {
            for (int q = 0; q < 4; ++q) {
                for (int r = 0; r < 4; ++r) {
                    const double s[3] = {g[p], g[q], g[r]};
                    double       x[3], so[3], r0[3], Ni[8], No[8];
                    for (int d = 0; d < 3; ++d) x[d] = xi[0][d] + Ei[d][0] * s[0] + Ei[d][1] * s[1] + Ei[d][2] * s[2];
                    for (int d = 0; d < 3; ++d) r0[d] = x[d] - xo[0][d];
                    for (int k = 0; k < 3; ++k) so[k] = Eoinv[k][0] * r0[0] + Eoinv[k][1] * r0[1] + Eoinv[k][2] * r0[2];
                    intersect::hex8_basis(s, Ni, nullptr);
                    intersect::hex8_basis(so, No, nullptr);
                    const double* Nf = fluid_inner ? Ni : No;
                    const double* Ns = fluid_inner ? No : Ni;
                    const double  wq = w[p] * w[q] * w[r] * det;
                    for (int i = 0; i < 8; ++i) {
                        for (int a = 0; a < 8; ++a) B[i][a] += wq * Nf[i] * Ns[a];
                    }
                }
            }
        }
    }

    int test_pair() {
        const double E_out[3][3] = {{1.0, 0.1, -0.05}, {0.05, 0.9, 0.1}, {-0.1, 0.05, 1.1}};
        const double E_in[3][3]  = {{0.35, 0.05, 0.02}, {-0.03, 0.3, 0.04}, {0.05, -0.02, 0.4}};
        const double o_out[3] = {0, 0, 0}, o_in[3] = {0.3, 0.32, 0.31};
        double       x_out[8][3], x_in[8][3];
        parallelepiped(o_out, E_out, x_out);
        parallelepiped(o_in, E_in, x_in);
        long long ids_a[8], ids_b[8];
        permutation(8, ids_a);
        permutation(8, ids_b);
        for (int fluid_inner = 0; fluid_inner < 2; ++fluid_inner) {
            const double(*xf)[3] = fluid_inner ? x_in : x_out;
            const double(*xs)[3] = fluid_inner ? x_out : x_in;
            double B[8][8], R[8][8];
            int    counts[5] = {0, 0, 0, 0, 0}, max_it = 0;
            SRESAMPLE_TEST_EQ(intersect::hex8_fluid_mixed_mass_pair(xs, ids_a, 8, xf, ids_b, B, intersect::Hex8Basis::TRILINEAR,
                                                                    intersect::Hex8Basis::TRILINEAR, counts, &max_it),
                              0);
            reference(xf, xs, fluid_inner, R);
            double err = 0, scale = 0;
            for (int i = 0; i < 8; ++i) {
                for (int a = 0; a < 8; ++a) {
                    err   = fmax(err, fabs(B[i][a] - R[i][a]));
                    scale = fmax(scale, fabs(R[i][a]));
                }
            }
            printf("  pair, %s inside: max |B - B_ref| = %.3g (max |B_ref| %.3g), Newton <= %d\n", fluid_inner ? "fluid" : "solid", err,
                   scale, max_it);
            SRESAMPLE_TEST_ASSERT(err < 1e-12 * scale);
            SRESAMPLE_TEST_ASSERT(max_it < intersect::hex8_newton_max_it);
            SRESAMPLE_TEST_EQ(counts[intersect::TET4_PAIR_OVERFLOW], 0);
        }
        return 0;
    }

    // ---- end to end -----------------------------------------------------------------------

    struct Mesh {
        int        nxe{8};
        ptrdiff_t  ne{0}, nn{0};
        int       *e[8]{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        double    *x[3]{nullptr, nullptr, nullptr};
        long long *ids{nullptr};
        void       free_all() {
            for (int v = 0; v < nxe; ++v) free(e[v]);
            for (int d = 0; d < 3; ++d) free(x[d]);
            free(ids);
        }
    };

    const double X0[3]   = {0.22, 0.25, 0.2};
    const double A[3][3] = {{0.5, 0.06, -0.04}, {0.03, 0.45, 0.08}, {-0.05, 0.04, 0.55}};

    // A structured block of n^3 cells: p in [0, 1]^3 mapped by `map`; hexes, or 6 Kuhn tets per
    // cell. Nodes numbered by a random permutation.
    template <class Map>
    Mesh block(const int n, const bool tets, Map map) {
        Mesh      m;
        const int nn = n + 1;
        m.nxe        = tets ? 4 : 8;
        m.nn         = (ptrdiff_t)nn * nn * nn;
        m.ne         = (ptrdiff_t)(tets ? 6 : 1) * n * n * n;
        long long *id = (long long *)malloc(sizeof(long long) * m.nn);
        permutation(m.nn, id);
        for (int d = 0; d < 3; ++d) m.x[d] = (double *)malloc(sizeof(double) * m.nn);
        for (int v = 0; v < m.nxe; ++v) m.e[v] = (int *)malloc(sizeof(int) * m.ne);
        for (int k = 0; k < nn; ++k) {
            for (int j = 0; j < nn; ++j) {
                for (int i = 0; i < nn; ++i) {
                    const int idx[3] = {i, j, k};
                    double    x[3];
                    map(idx, n, x);
                    for (int d = 0; d < 3; ++d) m.x[d][id[(k * nn + j) * nn + i]] = x[d];
                }
            }
        }
        const int corner[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
        const int perms[6][3]  = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
        ptrdiff_t e            = 0;
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    if (!tets) {
                        for (int c = 0; c < 8; ++c) {
                            m.e[c][e] = (int)id[((k + corner[c][2]) * nn + j + corner[c][1]) * nn + i + corner[c][0]];
                        }
                        ++e;
                        continue;
                    }
                    for (int t = 0; t < 6; ++t) {
                        int pos[3] = {i, j, k};
                        m.e[0][e]  = (int)id[(pos[2] * nn + pos[1]) * nn + pos[0]];
                        for (int s = 0; s < 3; ++s) {
                            pos[perms[t][s]]++;
                            m.e[s + 1][e] = (int)id[(pos[2] * nn + pos[1]) * nn + pos[0]];
                        }
                        ++e;
                    }
                }
            }
        }
        free(id);
        return m;
    }

    // The fluid: [0, 1]^3, inner nodes perturbed; global ids another random numbering.
    Mesh fluid_mesh(const int n) {
        Mesh m = block(n, false, [](const int idx[3], const int n, double x[3]) {
            for (int d = 0; d < 3; ++d) {
                const bool inner = idx[d] > 0 && idx[d] < n;
                x[d]             = (idx[d] + (inner ? 0.3 * (uniform() - 0.5) : 0.0)) / n;
            }
        });
        m.ids = (long long *)malloc(sizeof(long long) * m.nn);
        permutation(m.nn, m.ids);
        return m;
    }

    Mesh solid_mesh(const int n, const bool tets, const bool warped) {
        return block(n, tets, [warped](const int idx[3], const int n, double x[3]) {
            const double p[3] = {(double)idx[0] / n, (double)idx[1] / n, (double)idx[2] / n};
            for (int d = 0; d < 3; ++d) x[d] = X0[d] + A[d][0] * p[0] + A[d][1] * p[1] + A[d][2] * p[2];
            if (warped) {
                x[0] += 0.03 * sin(3 * p[1] + 2 * p[2]);
                x[1] += 0.03 * p[0] * p[2];
                x[2] += 0.02 * cos(2 * p[0]);
            }
        });
    }

    // |split(solid)| and int_split(solid) x.
    void solid_moments(const Mesh &m, double *vol, double mom[3]) {
        *vol = 0;
        mom[0] = mom[1] = mom[2] = 0;
        for (ptrdiff_t e = 0; e < m.ne; ++e) {
            double    xs[8][3];
            long long ids[8];
            for (int a = 0; a < m.nxe; ++a) {
                ids[a] = m.e[a][e];
                for (int d = 0; d < 3; ++d) xs[a][d] = m.x[d][m.e[a][e]];
            }
            int t[6][4], n = 1;
            if (m.nxe == 8) {
                n = intersect::hex8_split(intersect::hex8_split_pattern(ids), t);
            } else {
                for (int v = 0; v < 4; ++v) t[0][v] = v;
            }
            for (int k = 0; k < n; ++k) {
                double ed[3][3];
                for (int j = 0; j < 3; ++j) {
                    for (int d = 0; d < 3; ++d) ed[j][d] = xs[t[k][j + 1]][d] - xs[t[k][0]][d];
                }
                const double V = fabs(intersect::tet4_det3(ed[0], ed[1], ed[2])) / 6;
                *vol += V;
                for (int d = 0; d < 3; ++d) {
                    double c = 0;
                    for (int v = 0; v < 4; ++v) c += xs[t[k][v]][d] / 4;
                    mom[d] += V * c;
                }
            }
        }
    }

    struct MeshOn {
        MeshView<double, int> view;
        int                  *e[8]{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        double               *x[3]{nullptr, nullptr, nullptr};
        long long            *ids{nullptr};
        int const           **ep{nullptr};
        double const        **xp{nullptr};
        bool                  device{false};
        int                   nxe{4};
        void                  free_all() {
#ifdef SRESAMPLE_ENABLE_CUDA
            if (device) {
                for (int v = 0; v < nxe; ++v) cudaFree(e[v]);
                for (int d = 0; d < 3; ++d) cudaFree(x[d]);
                cudaFree(ids);
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
            o.ids = m.ids;
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
            if (m.ids) {
                cudaMalloc(&o.ids, sizeof(long long) * m.nn);
                cudaMemcpy(o.ids, m.ids, sizeof(long long) * m.nn, cudaMemcpyHostToDevice);
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
        o.view.ids       = o.ids;
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

    int check(const bool device, const bool tets, const bool warped, const int basis, const int fluid_basis) {
        const Exec exec  = device ? Exec::device() : Exec::host();
        Mesh       fluid = fluid_mesh(6);
        Mesh       solid = solid_mesh(3, tets, warped);
        MeshOn     f = mesh_on(fluid, device), s = mesh_on(solid, device);

        // Candidate pairs: the cell list over the fluid's boxes, queried with the solid's.
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

        const int                      ns = solid.nxe;
        double                        *B  = (double *)alloc(device, sizeof(double) * 8 * ns * np);
        intersect::Hex8TransferOptions options;
        options.basis       = basis;
        options.fluid_basis = fluid_basis;
        intersect::Tet4PairCounts counts;
        int                       max_it = 0;
        SRESAMPLE_TEST_EQ(intersect::hex8_fluid_mixed_mass<double>(s.view, f.view, np, ps, pf, B, options, &counts, &max_it, exec), 0);
        printf("  %s, %s %s solid (%s), fluid %s: %td pairs; tet pairs: %td separated, %td fluid inside, %td solid inside, %td clipped, "
               "%td overflow; Newton <= %d iterations\n",
               device ? "device" : "host", warped ? "warped" : "affine", tets ? "TET4" : "HEX8", basis ? "split" : "trilinear",
               fluid_basis ? "split" : "trilinear", np, counts.separated, counts.fluid_inside, counts.solid_inside, counts.clipped,
               counts.overflow, max_it);
        SRESAMPLE_TEST_EQ(counts.overflow, 0);
        SRESAMPLE_TEST_ASSERT(max_it < intersect::hex8_newton_max_it);

        double vol = 0, mom[3];
        solid_moments(solid, &vol, mom);
        const double tol = 1e-13 * vol;

        // Uniform v, then v = x; u = x, then random u (with random v: the duality).
        const double vel[3] = {0.3, -1.2, 2.5};
        double      *v[3], *U[3], *u[3], *g[3];
        double      *W = (double *)alloc(device, sizeof(double) * fluid.nn);
        double      *c = (double *)alloc(device, sizeof(double) * solid.nn);
        double      *h = (double *)malloc(sizeof(double) * (fluid.nn > solid.nn ? fluid.nn : solid.nn));
        for (int d = 0; d < 3; ++d) {
            v[d] = (double *)alloc(device, sizeof(double) * solid.nn);
            for (ptrdiff_t a = 0; a < solid.nn; ++a) h[a] = vel[d];
            copy(device, v[d], h, sizeof(double) * solid.nn, false);
            U[d] = (double *)alloc(device, sizeof(double) * fluid.nn);
            u[d] = (double *)alloc(device, sizeof(double) * fluid.nn);
            copy(device, u[d], fluid.x[d], sizeof(double) * fluid.nn, false);
            g[d] = (double *)alloc(device, sizeof(double) * solid.nn);
        }
        SRESAMPLE_TEST_EQ(intersect::hex8_fluid_mixed_mass_apply<double>(s.view, f.view, np, ps, pf, B, v, W, U, exec), 0);
        SRESAMPLE_TEST_EQ(intersect::hex8_fluid_mixed_mass_apply_transpose<double>(s.view, f.view, np, ps, pf, B, u, c, g, exec), 0);
#ifdef SRESAMPLE_ENABLE_CUDA
        if (device) cudaDeviceSynchronize();
#endif
        double *hW = (double *)malloc(sizeof(double) * fluid.nn);
        double *hc = (double *)malloc(sizeof(double) * solid.nn);
        copy(device, hW, W, sizeof(double) * fluid.nn, true);
        copy(device, hc, c, sizeof(double) * solid.nn, true);
        double sumW = 0, sumc = 0;
        for (ptrdiff_t i = 0; i < fluid.nn; ++i) sumW += hW[i];
        for (ptrdiff_t a = 0; a < solid.nn; ++a) sumc += hc[a];
        SRESAMPLE_TEST_APPROXEQ(sumW, vol, tol);
        SRESAMPLE_TEST_APPROXEQ(sumc, vol, tol);
        for (int d = 0; d < 3; ++d) {
            copy(device, h, U[d], sizeof(double) * fluid.nn, true);
            for (ptrdiff_t i = 0; i < fluid.nn; ++i) SRESAMPLE_TEST_APPROXEQ(h[i], vel[d] * hW[i], tol);
            copy(device, h, g[d], sizeof(double) * solid.nn, true);
            double sum = 0;
            for (ptrdiff_t a = 0; a < solid.nn; ++a) sum += h[a];
            SRESAMPLE_TEST_APPROXEQ(sum, mom[d], tol);
        }

        // v = x at the solid nodes: sum_i U_i = int x.
        for (int d = 0; d < 3; ++d) {
            copy(device, v[d], solid.x[d], sizeof(double) * solid.nn, false);
            for (ptrdiff_t i = 0; i < fluid.nn; ++i) h[i] = 0;
            copy(device, U[d], h, sizeof(double) * fluid.nn, false);
        }
        for (ptrdiff_t i = 0; i < fluid.nn; ++i) h[i] = 0;
        copy(device, W, h, sizeof(double) * fluid.nn, false);
        SRESAMPLE_TEST_EQ(intersect::hex8_fluid_mixed_mass_apply<double>(s.view, f.view, np, ps, pf, B, v, W, U, exec), 0);
#ifdef SRESAMPLE_ENABLE_CUDA
        if (device) cudaDeviceSynchronize();
#endif
        for (int d = 0; d < 3; ++d) {
            copy(device, h, U[d], sizeof(double) * fluid.nn, true);
            double sum = 0;
            for (ptrdiff_t i = 0; i < fluid.nn; ++i) sum += h[i];
            SRESAMPLE_TEST_APPROXEQ(sum, mom[d], tol);
        }

        // The duality, component 0: random u (fluid), random v (solid).
        {
            double *ru = (double *)malloc(sizeof(double) * fluid.nn), *rv = (double *)malloc(sizeof(double) * solid.nn);
            for (ptrdiff_t i = 0; i < fluid.nn; ++i) ru[i] = uniform() - 0.5;
            for (ptrdiff_t a = 0; a < solid.nn; ++a) rv[a] = uniform() - 0.5;
            for (int d = 0; d < 3; ++d) {
                copy(device, u[d], ru, sizeof(double) * fluid.nn, false);
                copy(device, v[d], rv, sizeof(double) * solid.nn, false);
                for (ptrdiff_t i = 0; i < fluid.nn; ++i) h[i] = 0;
                copy(device, U[d], h, sizeof(double) * fluid.nn, false);
                copy(device, g[d], h, sizeof(double) * solid.nn, false);
            }
            copy(device, W, h, sizeof(double) * fluid.nn, false);
            SRESAMPLE_TEST_EQ(intersect::hex8_fluid_mixed_mass_apply<double>(s.view, f.view, np, ps, pf, B, v, W, U, exec), 0);
            SRESAMPLE_TEST_EQ(intersect::hex8_fluid_mixed_mass_apply_transpose<double>(s.view, f.view, np, ps, pf, B, u, nullptr, g, exec), 0);
#ifdef SRESAMPLE_ENABLE_CUDA
            if (device) cudaDeviceSynchronize();
#endif
            double lhs = 0, rhs = 0;
            copy(device, h, U[0], sizeof(double) * fluid.nn, true);
            for (ptrdiff_t i = 0; i < fluid.nn; ++i) lhs += ru[i] * h[i];
            copy(device, h, g[0], sizeof(double) * solid.nn, true);
            for (ptrdiff_t a = 0; a < solid.nn; ++a) rhs += rv[a] * h[a];
            SRESAMPLE_TEST_APPROXEQ(lhs, rhs, tol);
            free(ru);
            free(rv);
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
        }
        release(device, ptr);
        release(device, ps);
        release(device, pf);
        release(device, B);
        release(device, W);
        release(device, c);
        free(h);
        free(hW);
        free(hc);
        f.free_all();
        s.free_all();
        fluid.free_all();
        solid.free_all();
        return 0;
    }

    int run(const bool device) {
        int err = 0;
        for (int fluid_basis = 0; fluid_basis < 2; ++fluid_basis) {
            err += check(device, true, false, 0, fluid_basis);
            for (int basis = 0; basis < 2; ++basis) {
                err += check(device, false, false, basis, fluid_basis);
                err += check(device, false, true, basis, fluid_basis);
            }
        }
        return err;
    }

    int test_host() { return run(false); }
#ifdef SRESAMPLE_ENABLE_CUDA
    int test_device() { return run(true); }
#endif

}  // namespace

int main() {
    int err = 0;
    SRESAMPLE_RUN_TEST(test_pair);
    SRESAMPLE_RUN_TEST(test_host);
#ifdef SRESAMPLE_ENABLE_CUDA
    SRESAMPLE_RUN_TEST(test_device);
#endif
    return err ? EXIT_FAILURE : EXIT_SUCCESS;
}
