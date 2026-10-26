// The cubic B-spline sampling of an SDF grid (resampling/surface/sfem_sdf_sampling.hpp): exact where it
// must be, h^4 where the field is smooth, and continuous to second derivatives across cells.
// Moved from sfem's sfem_SDFSamplingTest: now sdf/sresample_bspline.hpp, and the nodal sampling
// sdf/sresample_node_sample.hpp.
#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <array>
#include <functional>
#include <random>
#include <vector>

#include "sresample_bspline.hpp"
#include "sresample_node_sample.hpp"
#include "sresample_test.hpp"

using namespace sresample;

namespace {
    using Field = std::function<double(const double, const double, const double)>;
    using Grid  = BSplineGrid<float, float>;

    // An SDF grid of `field` on [0, 1]^3 with n nodes per axis, and its coefficients.
    struct Sampled {
        ptrdiff_t          n;
        std::vector<float> samples;
        std::vector<float> coeffs;
        Grid               grid;

        Sampled(const Field &field, const ptrdiff_t n_) : n(n_) {
            const float h = 1.f / (n - 1);
            samples.resize(n * n * n);
            for (ptrdiff_t k = 0; k < n; ++k)
                for (ptrdiff_t j = 0; j < n; ++j)
                    for (ptrdiff_t i = 0; i < n; ++i)
                        samples[i + n * (j + n * k)] = (float)field((float)(i * h), (float)(j * h), (float)(k * h));

            const ptrdiff_t nn[3]            = {n, n, n};
            const ptrdiff_t sample_stride[3] = {1, n, n * n};
            ptrdiff_t       size, offset;
            sdf::bspline_layout(nn, grid.stride, &size, &offset);
            coeffs.resize(size);
            sdf::bspline_coefficients(nn, sample_stride, samples.data(), grid.stride, coeffs.data() + offset);
            for (int d = 0; d < 3; ++d) {
                grid.n[d]      = n;
                grid.origin[d] = 0;
                grid.delta[d]  = h;
            }
            grid.coeffs = coeffs.data() + offset;
        }

        template <int ORDER>
        bool sample(const double x, const double y, const double z, double *f, double *g, double *H) const {
            return sdf::sample<ORDER>(grid, x, y, z, f, g, H);
        }
    };

    const double sphere_radius = 0.3;

    double sphere(const double x, const double y, const double z) {
        return sphere_radius - std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + (z - 0.5) * (z - 0.5));
    }

    // The nodal sampling of node_interpolate.hpp at the points disp (xyz zero), all of them.
    void sample_points(const Sampled &s, const ptrdiff_t np, const double *const p, double *const gap, double *const *const normals, double *const *const curvature) {
        std::vector<float>       zero(np, 0.f);
        sdf::NodePoints<float>   points;
        for (int d = 0; d < 3; ++d) points.xyz[d] = zero.data();
        if (gap || normals) {
            sdf::NodeOutput<double> out;
            out.gap = gap;
            if (normals)
                for (int d = 0; d < 3; ++d) out.normals[d] = normals[d];
            sdf::sample_nodes(1, np, (const int32_t *)nullptr, points, p, s.grid, out, Exec::host());
        }
        if (curvature) {
            sdf::NodeOutput<double> out;
            for (int c = 0; c < 6; ++c) out.curvature[c] = curvature[c];
            sdf::sample_nodes(2, np, (const int32_t *)nullptr, points, p, s.grid, out, Exec::host());
        }
    }

    // Rows of a dense n x m array.
    struct Rows {
        std::vector<double>   data;
        std::vector<double *> rows;
        Rows(const int n, const ptrdiff_t m) : data(n * m), rows(n) {
            for (int i = 0; i < n; ++i) rows[i] = &data[i * m];
        }
        double *const *ptr() const { return rows.data(); }
        double        *operator[](const int i) const { return rows[i]; }
    };

    // Moves the points onto the spline's zero level set: p += gap n, where s(p + t n) = s + t |grad s|.
    void to_surface(const Sampled &s, const ptrdiff_t np, double *const p) {
        std::vector<double> gap(np);
        Rows                normals(3, np);
        for (int it = 0; it < 6; ++it) {
            sample_points(s, np, p, gap.data(), normals.ptr(), nullptr);
            for (ptrdiff_t i = 0; i < np; ++i) {
                for (int d = 0; d < 3; ++d) {
                    p[i * 3 + d] += gap[i] * normals[d][i];
                }
            }
        }
    }
}  // namespace

// Nodes are interpolated, and a linear field (a plane) is reproduced everywhere, gradient
// included; float storage of the coefficients sets the tolerance.
int test_reproduces_nodes_and_planes() {
    const Field   plane = [](const double x, const double y, const double z) { return 0.3 * x - 0.7 * y + 0.2 * z + 0.1; };
    const Sampled s(plane, 17);

    std::mt19937                           rng(7);
    std::uniform_real_distribution<double> u(0, 0.999);
    double                                 f, g[3], H[6];
    for (int k = 0; k < 1000; ++k) {
        const double x = u(rng), y = u(rng), z = u(rng);
        SRESAMPLE_TEST_ASSERT(s.sample<2>(x, y, z, &f, g, H));
        SRESAMPLE_TEST_APPROXEQ(f, plane(x, y, z), 1e-6);
        SRESAMPLE_TEST_APPROXEQ(g[0], 0.3, 1e-5);
        SRESAMPLE_TEST_APPROXEQ(g[1], -0.7, 1e-5);
        SRESAMPLE_TEST_APPROXEQ(g[2], 0.2, 1e-5);
        for (int i = 0; i < 6; ++i) {
            SRESAMPLE_TEST_APPROXEQ(H[i], 0, 1e-3);
        }
    }

    const Sampled r(sphere, 17);
    const double  h = 1. / 16;
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            for (int k = 0; k < 16; ++k) {
                SRESAMPLE_TEST_ASSERT(r.sample<0>(i * h, j * h, k * h, &f, nullptr, nullptr));
                SRESAMPLE_TEST_APPROXEQ(f, (double)r.samples[i + 17 * (j + 17 * k)], 1e-6);
            }
        }
    }
    return 0;
}

// Near a sphere's surface (away from its centre, where the distance is not smooth) the sampled
// distance converges as h^4; trilinear sampling converged as h^2.
int test_converges_as_h4_on_a_sphere() {
    std::mt19937                           rng(11);
    std::uniform_real_distribution<double> u(-1, 1);
    std::vector<std::array<double, 3>>     points;
    while (points.size() < 2000) {
        const double p[3] = {0.5 + u(rng) * 0.45, 0.5 + u(rng) * 0.45, 0.5 + u(rng) * 0.45};
        const double r    = std::sqrt((p[0] - 0.5) * (p[0] - 0.5) + (p[1] - 0.5) * (p[1] - 0.5) + (p[2] - 0.5) * (p[2] - 0.5));
        if (std::abs(r - sphere_radius) < 0.04) {
            points.push_back({p[0], p[1], p[2]});
        }
    }

    double previous = 0;
    for (const ptrdiff_t n : {13, 25, 49}) {
        const Sampled s(sphere, n);
        double        error = 0;
        for (const auto &p : points) {
            double f;
            SRESAMPLE_TEST_ASSERT(s.sample<0>(p[0], p[1], p[2], &f, nullptr, nullptr));
            error = std::max(error, std::abs(f - sphere(p[0], p[1], p[2])));
        }
        printf("n = %ld: max error %g", (long)n, error);
        if (previous > 0) {
            const double order = std::log2(previous / error);
            printf(", order %.2f", order);
            SRESAMPLE_TEST_ASSERT(order > 3.5);
        }
        printf("\n");
        previous = error;
    }
    return 0;
}

// The gradient and the Hessian are continuous across the faces between cells.
int test_continuous_across_cells() {
    const ptrdiff_t n = 21;
    const Sampled   s(sphere, n);
    const double    h   = 1. / (n - 1);
    const double    eps = 1e-9;

    std::mt19937                           rng(3);
    std::uniform_real_distribution<double> u(0.2, 0.8);
    double                                 jump_g = 0, jump_H = 0;
    for (int k = 0; k < 500; ++k) {
        double    p[3] = {u(rng), u(rng), u(rng)};
        const int axis = k % 3;
        p[axis]        = std::round(p[axis] / h) * h;  // on a face

        double a[3] = {p[0], p[1], p[2]}, b[3] = {p[0], p[1], p[2]};
        a[axis] -= eps;
        b[axis] += eps;
        double fa, ga[3], Ha[6], fb, gb[3], Hb[6];
        SRESAMPLE_TEST_ASSERT(s.sample<2>(a[0], a[1], a[2], &fa, ga, Ha));
        SRESAMPLE_TEST_ASSERT(s.sample<2>(b[0], b[1], b[2], &fb, gb, Hb));
        for (int d = 0; d < 3; ++d) jump_g = std::max(jump_g, std::abs(ga[d] - gb[d]));
        for (int d = 0; d < 6; ++d) jump_H = std::max(jump_H, std::abs(Ha[d] - Hb[d]));
    }
    printf("largest jump across a face: gradient %g, Hessian %g\n", jump_g, jump_H);
    SRESAMPLE_TEST_ASSERT(jump_g < 1e-6);
    SRESAMPLE_TEST_ASSERT(jump_H < 1e-4);
    return 0;
}

// Outside the grid nothing is sampled; a grid needs two nodes per axis.
int test_outside_is_reported() {
    const Sampled s(sphere, 9);
    double        f;
    SRESAMPLE_TEST_ASSERT(!s.sample<0>(-0.01, 0.5, 0.5, &f, nullptr, nullptr));
    SRESAMPLE_TEST_ASSERT(!s.sample<0>(0.5, 1.0, 0.5, &f, nullptr, nullptr));
    SRESAMPLE_TEST_ASSERT(s.sample<0>(0.5, 0.999, 0.5, &f, nullptr, nullptr));

    const ptrdiff_t n[3] = {1, 4, 4}, stride[3] = {1, 1, 4};
    float           samples[64] = {0}, coeffs[64];
    SRESAMPLE_TEST_EQ(sdf::bspline_coefficients(n, stride, samples, stride, coeffs), (int)SRESAMPLE_FAILURE);
    return 0;
}

// The nodal curvature (H - n m^T - m n^T) / |grad s|, m = H n, from the prefiltered coefficients:
// on a sphere it converges to -(I - n n^T) / R as h^2; on the surface of a field that is not a
// distance (|grad s| varies, m != 0) it is minus the Hessian of the gap -s / |grad s| exactly;
// quadratics are reproduced (Hessian 2A) away from the grid's ends and planes have none anywhere.
// Points are sampled as X[idx] + disp with any idx and disp, and outside the grid the gap is
// unreachable, the normal and curvature zero.
int test_nodal_curvature() {
    // Sphere: convergence. Small, so that it stays clear of the prefilter's end layer.
    {
        const double R    = 0.15;
        const Field  ball = [R](const double x, const double y, const double z) {
            return R - std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + (z - 0.5) * (z - 0.5));
        };
        const ptrdiff_t                  np = 200;
        std::vector<double>              p(3 * np), nx(3 * np);
        std::mt19937                     rng(5);
        std::normal_distribution<double> u(0, 1);
        for (ptrdiff_t i = 0; i < np; ++i) {
            double       d[3] = {u(rng), u(rng), u(rng)};
            const double len  = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            for (int k = 0; k < 3; ++k) {
                nx[i * 3 + k] = -d[k] / len;  // grad of R - |x - c|
                p[i * 3 + k]  = 0.5 + R * d[k] / len;
            }
        }

        double previous = 0;
        for (const ptrdiff_t n : {17, 33, 65}) {
            const Sampled s(ball, n);
            Rows          C(6, np);
            sample_points(s, np, p.data(), nullptr, nullptr, C.ptr());
            double error = 0;
            for (ptrdiff_t i = 0; i < np; ++i) {
                const double *const nn = &nx[i * 3];
                int                 c  = 0;
                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                        const double exact = -((a == b) - nn[a] * nn[b]) / R;
                        error              = std::max(error, std::abs(C[c][i] - exact));
                        c++;
                    }
                }
            }
            printf("sphere curvature, n = %ld: max error %g (1/R = %g)", (long)n, error, 1 / R);
            if (previous > 0) {
                const double order = std::log2(previous / error);
                printf(", order %.2f", order);
                SRESAMPLE_TEST_ASSERT(order > 1.7);
            }
            printf("\n");
            previous = error;
        }
    }

    // Ellipsoid level set (not a distance): the curvature is minus the Hessian of the sampled gap
    // -s / |grad s| on the surface, by central differences; the m terms matter here.
    {
        const Field ellipsoid = [](const double x, const double y, const double z) {
            return 0.04 - ((x - 0.5) * (x - 0.5) + 2 * (y - 0.5) * (y - 0.5) + 3 * (z - 0.5) * (z - 0.5));
        };
        const Sampled                    s(ellipsoid, 33);
        const ptrdiff_t                  np = 50;
        std::vector<double>              p(3 * np);
        std::mt19937                     rng(9);
        std::normal_distribution<double> u(0, 1);
        for (ptrdiff_t i = 0; i < np; ++i) {
            for (int k = 0; k < 3; ++k) {
                p[i * 3 + k] = 0.5 + 0.1 * u(rng);
            }
        }
        to_surface(s, np, p.data());

        Rows                C(6, np);
        std::vector<double> g(np);
        sample_points(s, np, p.data(), g.data(), nullptr, C.ptr());
        for (ptrdiff_t i = 0; i < np; ++i) {
            SRESAMPLE_TEST_ASSERT(std::abs(g[i]) < 1e-12);
        }

        const double        eps = 1e-4;
        std::vector<double> q(3 * np), gq(np), hess(np);
        double              error = 0, scale = 0, correction = 0;
        int                 c = 0;
        for (int a = 0; a < 3; ++a) {
            for (int b = a; b < 3; ++b) {
                // g at (+a, +b), (+a, -b), (-a, +b), (-a, -b)
                for (ptrdiff_t i = 0; i < np; ++i) hess[i] = 0;
                for (int corner = 0; corner < 4; ++corner) {
                    const double sa = corner < 2 ? 1 : -1, sb = corner % 2 == 0 ? 1 : -1;
                    for (ptrdiff_t i = 0; i < np; ++i) {
                        for (int k = 0; k < 3; ++k) q[i * 3 + k] = p[i * 3 + k];
                        q[i * 3 + a] += sa * eps;
                        q[i * 3 + b] += sb * eps;
                    }
                    sample_points(s, np, q.data(), gq.data(), nullptr, nullptr);
                    for (ptrdiff_t i = 0; i < np; ++i) {
                        hess[i] += sa * sb * gq[i] / (4 * eps * eps);
                    }
                }
                for (ptrdiff_t i = 0; i < np; ++i) {
                    error = std::max(error, std::abs(C[c][i] + hess[i]));
                    scale = std::max(scale, std::abs(C[c][i]));
                }
                c++;
            }
        }
        // How far the plain H / |grad s| is from it: what the m terms correct.
        {
            double f, gr[3], H[6];
            for (ptrdiff_t i = 0; i < np; ++i) {
                SRESAMPLE_TEST_ASSERT(s.sample<2>(p[i * 3], p[i * 3 + 1], p[i * 3 + 2], &f, gr, H));
                const double qn = std::sqrt(gr[0] * gr[0] + gr[1] * gr[1] + gr[2] * gr[2]);
                for (int k = 0; k < 6; ++k) correction = std::max(correction, std::abs(H[k] / qn - C[k][i]));
            }
        }
        printf("ellipsoid: curvature vs -hess(gap) by differences %g (of %g); H / |grad s| is off by %g\n", error, scale, correction);
        SRESAMPLE_TEST_ASSERT(error < 1e-5 * scale);
        SRESAMPLE_TEST_ASSERT(correction > 100 * error);

        // A quadratic is reproduced away from the grid's ends: Hessian 2A.
        const double                           exact[6] = {-2, 0, 0, -4, 0, -6};
        std::mt19937                           r2(13);
        std::uniform_real_distribution<double> w(0.3, 0.7);
        for (int k = 0; k < 200; ++k) {
            double f, gr[3], H[6];
            SRESAMPLE_TEST_ASSERT(s.sample<2>(w(r2), w(r2), w(r2), &f, gr, H));
            for (int e = 0; e < 6; ++e) SRESAMPLE_TEST_APPROXEQ(H[e], exact[e], 1e-3);
        }
    }

    // A plane has no curvature, up to the grid's ends.
    {
        const Field     plane = [](const double x, const double y, const double z) { return 0.3 * x - 0.7 * y + 0.2 * z + 0.1; };
        const Sampled   s(plane, 17);
        const ptrdiff_t np = 500;
        std::vector<double>                    p(3 * np);
        std::mt19937                           rng(17);
        std::uniform_real_distribution<double> u(0, 0.999);
        for (ptrdiff_t i = 0; i < 3 * np; ++i) p[i] = u(rng);
        Rows C(6, np);
        sample_points(s, np, p.data(), nullptr, nullptr, C.ptr());
        for (ptrdiff_t i = 0; i < np; ++i) {
            for (int c = 0; c < 6; ++c) SRESAMPLE_TEST_APPROXEQ(C[c][i], 0, 1e-3);
        }
    }

    // X[idx] + disp, against the same points given compactly; and outside the grid.
    {
        const Sampled                          s(sphere, 17);
        const ptrdiff_t                        nn = 64, np = 20;
        std::vector<float>                     X(3 * nn), Xc(3 * np);
        std::vector<double>                    disp(3 * nn), dc(3 * np);
        std::vector<int32_t>                   idx(np);
        std::mt19937                           rng(23);
        std::uniform_real_distribution<double> u(0.1, 0.6);
        for (ptrdiff_t i = 0; i < nn; ++i) {
            for (int d = 0; d < 3; ++d) {
                X[d * nn + i]    = (float)u(rng);
                disp[i * 3 + d] = u(rng) / 3;
            }
        }
        for (ptrdiff_t i = 0; i < np; ++i) {
            idx[i] = (int32_t)((7 * i + 3) % nn);
            for (int d = 0; d < 3; ++d) {
                Xc[d * np + i]  = X[d * nn + idx[i]];
                dc[i * 3 + d] = disp[idx[i] * 3 + d];
            }
        }
        dc[0] = 5;  // outside

        auto run = [&](const int32_t *const ix, const float *const xyz, const ptrdiff_t stride, const double *const dd, double *const gap, const Rows &nrm, const Rows &C) {
            sdf::NodePoints<float> points;
            for (int d = 0; d < 3; ++d) points.xyz[d] = xyz + d * stride;
            sdf::NodeOutput<double> out1, out2;
            out1.gap = gap;
            for (int d = 0; d < 3; ++d) out1.normals[d] = nrm[d];
            for (int c = 0; c < 6; ++c) out2.curvature[c] = C[c];
            sdf::sample_nodes(1, np, ix, points, dd, s.grid, out1, Exec::host());
            sdf::sample_nodes(2, np, ix, points, dd, s.grid, out2, Exec::host());
        };
        disp[idx[0] * 3] = 5;
        std::vector<double> ga(np), gb(np);
        Rows                na(3, np), nb(3, np), Ca(6, np), Cb(6, np);
        run(idx.data(), X.data(), nn, disp.data(), ga.data(), na, Ca);
        run(nullptr, Xc.data(), np, dc.data(), gb.data(), nb, Cb);
        for (ptrdiff_t i = 0; i < np; ++i) {
            SRESAMPLE_TEST_EQ(ga[i], gb[i]);
            for (int d = 0; d < 3; ++d) SRESAMPLE_TEST_EQ(na[d][i], nb[d][i]);
            for (int c = 0; c < 6; ++c) SRESAMPLE_TEST_EQ(Ca[c][i], Cb[c][i]);
        }

        // The sum is formed in real_t: the gap at point 1 is that of sdf_sample there.
        {
            const ptrdiff_t b = idx[1];
            double          p[3], f, gr[3];
            for (int d = 0; d < 3; ++d) p[d] = (double)X[d * nn + b] + disp[b * 3 + d];
            SRESAMPLE_TEST_ASSERT(s.sample<1>(p[0], p[1], p[2], &f, gr, nullptr));
            SRESAMPLE_TEST_APPROXEQ(ga[1], -f / std::sqrt(gr[0] * gr[0] + gr[1] * gr[1] + gr[2] * gr[2]), 1e-15);
        }

        SRESAMPLE_TEST_EQ(ga[0], double(sdf::unreachable));
        for (int d = 0; d < 3; ++d) SRESAMPLE_TEST_EQ(na[d][0], 0.);
        for (int c = 0; c < 6; ++c) SRESAMPLE_TEST_EQ(Ca[c][0], 0.);
    }
    return 0;
}

int main() {
    int err = 0;
    SRESAMPLE_RUN_TEST(test_reproduces_nodes_and_planes);
    SRESAMPLE_RUN_TEST(test_converges_as_h4_on_a_sphere);
    SRESAMPLE_RUN_TEST(test_continuous_across_cells);
    SRESAMPLE_RUN_TEST(test_outside_is_reported);
    SRESAMPLE_RUN_TEST(test_nodal_curvature);
    return err ? 1 : 0;
}
