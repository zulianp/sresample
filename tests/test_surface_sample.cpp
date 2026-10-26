// The surface sampling (sdf/sresample_surface_sample.hpp) on every element type it takes:
// TRISHELL3, QUADSHELL4 and BEAM2. On flat elements (triangles, parallelograms, straight segments)
// a linear field is reproduced at every node, the boundary ones included: the dual basis is
// biorthogonal, int psi_i phi_j = delta_ij w_j, so -int psi_i s / w_i = -s(x_i). The normal is the
// field's unit gradient there. The lumped weights sum to the measure; outside the grid the gap is
// sdf::unreachable and the normal the element's own. With CUDA the device agrees with the host.
//
// Mutation checks (each makes a test fail): a dual coefficient changed (3 -> 2 in Trishell3::qp,
// 4 -> 3 in Quadshell4::qp); the sign flip of the gap dropped; the outside gap taken as 0.
#include <math.h>
#include <stdio.h>

#include <functional>
#include <vector>

#include "sresample_bspline.hpp"
#include "sresample_surface_sample.hpp"
#include "sresample_test.hpp"

#ifdef SRESAMPLE_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

using namespace sresample;

namespace {
    // s(x) = a . x + b, its gradient not of unit length.
    const double a[3] = {0.3, -0.7, 0.2};
    const double b    = 0.1;

    double plane(const double x, const double y, const double z) { return a[0] * x + a[1] * y + a[2] * z + b; }

    // The field on [0, 1]^3, 17 nodes per axis, float coefficients.
    struct Spline {
        std::vector<float>        samples, coeffs;
        BSplineGrid<float, float> grid;

        explicit Spline(const std::function<double(double, double, double)> &f, const ptrdiff_t n = 17) {
            const float h = 1.f / (n - 1);
            samples.resize(n * n * n);
            for (ptrdiff_t k = 0; k < n; ++k)
                for (ptrdiff_t j = 0; j < n; ++j)
                    for (ptrdiff_t i = 0; i < n; ++i) samples[i + n * (j + n * k)] = (float)f(i * h, j * h, k * h);
            const ptrdiff_t nn[3] = {n, n, n}, ss[3] = {1, n, n * n};
            ptrdiff_t       size, offset;
            sdf::bspline_layout(nn, grid.stride, &size, &offset);
            coeffs.resize(size);
            sdf::bspline_coefficients(nn, ss, samples.data(), grid.stride, coeffs.data() + offset);
            for (int d = 0; d < 3; ++d) {
                grid.n[d]      = n;
                grid.origin[d] = 0;
                grid.delta[d]  = h;
            }
            grid.coeffs = coeffs.data() + offset;
        }
    };

    // A mesh held in vectors, viewed as a MeshView.
    struct Mesh {
        ElemType                          type;
        std::vector<std::vector<int32_t>> elems;
        std::vector<float>                xyz[3];
        std::vector<const int32_t *>      e_ptr;
        std::vector<const float *>        x_ptr;

        MeshView<float, int32_t> view() {
            e_ptr.clear();
            x_ptr.clear();
            for (auto &e : elems) e_ptr.push_back(e.data());
            for (int d = 0; d < 3; ++d) x_ptr.push_back(xyz[d].data());
            MeshView<float, int32_t> v;
            v.type      = type;
            v.nelements = elems[0].size();
            v.nnodes    = xyz[0].size();
            v.elements  = e_ptr.data();
            v.points    = x_ptr.data();
            return v;
        }
    };

    // An m x m patch of the parallelogram c + u e1 + v e2, u, v in [0, 1]: quadrilaterals, or each
    // split in two triangles. Every element is flat.
    Mesh patch(const ElemType type, const int m, const double *c, const double *e1, const double *e2) {
        Mesh mesh;
        mesh.type = type;
        mesh.elems.resize(nxe(type));
        for (int j = 0; j <= m; ++j) {
            for (int i = 0; i <= m; ++i) {
                for (int d = 0; d < 3; ++d) mesh.xyz[d].push_back((float)(c[d] + e1[d] * i / m + e2[d] * j / m));
            }
        }
        for (int j = 0; j < m; ++j) {
            for (int i = 0; i < m; ++i) {
                const int32_t v[4] = {j * (m + 1) + i, j * (m + 1) + i + 1, (j + 1) * (m + 1) + i + 1, (j + 1) * (m + 1) + i};
                if (type == ElemType::QUADSHELL4) {
                    for (int k = 0; k < 4; ++k) mesh.elems[k].push_back(v[k]);
                } else {
                    const int32_t t[2][3] = {{v[0], v[1], v[2]}, {v[0], v[2], v[3]}};
                    for (int s = 0; s < 2; ++s)
                        for (int k = 0; k < 3; ++k) mesh.elems[k].push_back(t[s][k]);
                }
            }
        }
        return mesh;
    }

    // m segments from p to q.
    Mesh polyline(const int m, const double *p, const double *q) {
        Mesh mesh;
        mesh.type = ElemType::BEAM2;
        mesh.elems.resize(2);
        for (int i = 0; i <= m; ++i) {
            for (int d = 0; d < 3; ++d) mesh.xyz[d].push_back((float)(p[d] + (q[d] - p[d]) * i / m));
        }
        for (int i = 0; i < m; ++i) {
            mesh.elems[0].push_back(i);
            mesh.elems[1].push_back(i + 1);
        }
        return mesh;
    }

    struct Result {
        std::vector<double> gap, nx, ny, nz;
        explicit Result(const ptrdiff_t n) : gap(n), nx(n), ny(n), nz(n) {}
        sdf::SurfaceOutput<double> out() {
            sdf::SurfaceOutput<double> o;
            o.gap        = gap.data();
            o.normals[0] = nx.data();
            o.normals[1] = ny.data();
            o.normals[2] = nz.data();
            return o;
        }
    };

    const double c0[3] = {0.2, 0.25, 0.3}, e1[3] = {0.5, 0.1, 0.05}, e2[3] = {0.1, 0.4, 0.3};

    std::vector<Mesh> meshes() {
        const double p[3] = {0.15, 0.2, 0.3}, q[3] = {0.8, 0.7, 0.6};
        std::vector<Mesh> m;
        m.push_back(patch(ElemType::TRISHELL3, 7, c0, e1, e2));
        m.push_back(patch(ElemType::QUADSHELL4, 7, c0, e1, e2));
        m.push_back(polyline(11, p, q));
        return m;
    }

    const char *name(const ElemType t) {
        return t == ElemType::TRISHELL3 ? "TRISHELL3" : t == ElemType::QUADSHELL4 ? "QUADSHELL4" : "BEAM2";
    }
}  // namespace

int test_linear_field_is_reproduced() {
    const Spline s(plane);
    const double na = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    for (auto &mesh : meshes()) {
        auto                v = mesh.view();
        Result              r(v.nnodes);
        std::vector<double> w(v.nnodes);
        SRESAMPLE_TEST_EQ(sdf::surface_sample(v, s.grid, r.out(), w.data(), Exec::host()), (int)SRESAMPLE_SUCCESS);
        double err_gap = 0, err_n = 0;
        for (ptrdiff_t i = 0; i < v.nnodes; ++i) {
            const double x = mesh.xyz[0][i], y = mesh.xyz[1][i], z = mesh.xyz[2][i];
            err_gap        = std::max(err_gap, std::abs(r.gap[i] + plane(x, y, z)));
            err_n          = std::max(err_n, std::abs(r.nx[i] - a[0] / na) + std::abs(r.ny[i] - a[1] / na) + std::abs(r.nz[i] - a[2] / na));
        }
        printf("%s: %ld nodes, gap error %g, normal error %g\n", name(mesh.type), (long)v.nnodes, err_gap, err_n);
        SRESAMPLE_TEST_ASSERT(err_gap < 1e-6);
        SRESAMPLE_TEST_ASSERT(err_n < 1e-5);
    }
    return 0;
}

int test_weights_sum_to_the_measure() {
    for (auto &mesh : meshes()) {
        auto                v = mesh.view();
        std::vector<double> w(v.nnodes, 0);
        SRESAMPLE_TEST_EQ(sdf::surface_weights_local(v, w.data(), Exec::host()), (int)SRESAMPLE_SUCCESS);
        double sum = 0;
        for (auto x : w) sum += x;
        double measure;
        if (mesh.type == ElemType::BEAM2) {
            const double d[3] = {0.8 - 0.15, 0.7 - 0.2, 0.6 - 0.3};
            measure           = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        } else {
            const double cx = e1[1] * e2[2] - e1[2] * e2[1], cy = e1[2] * e2[0] - e1[0] * e2[2], cz = e1[0] * e2[1] - e1[1] * e2[0];
            measure = std::sqrt(cx * cx + cy * cy + cz * cz);
        }
        printf("%s: weights sum %.12g, measure %.12g\n", name(mesh.type), sum, measure);
        SRESAMPLE_TEST_APPROXEQ(sum, measure, 1e-6 * measure);
    }
    return 0;
}

// A patch beyond the grid's end in x: unreachable gap, the element's normal. u x v = (0.09, 0, 0),
// so the triangles' winding normal is +x. The quadrilaterals' is -x: Quadshell4::measure_and_normal
// crosses dX/deta with dX/dxi (as sfem's quadshell4 kernels did), the reverse of the triangles.
int test_outside_the_grid() {
    const Spline s(plane);
    const double c[3] = {1.2, 0.3, 0.3}, u[3] = {0.0, 0.3, 0.0}, v3[3] = {0.0, 0.0, 0.3};
    for (const ElemType t : {ElemType::TRISHELL3, ElemType::QUADSHELL4}) {
        auto                mesh = patch(t, 2, c, u, v3);
        auto                v    = mesh.view();
        Result              r(v.nnodes);
        std::vector<double> w(v.nnodes);
        SRESAMPLE_TEST_EQ(sdf::surface_sample(v, s.grid, r.out(), w.data(), Exec::host()), (int)SRESAMPLE_SUCCESS);
        const double sign = t == ElemType::QUADSHELL4 ? -1 : 1;
        for (ptrdiff_t i = 0; i < v.nnodes; ++i) {
            SRESAMPLE_TEST_APPROXEQ(r.gap[i], (double)sdf::unreachable, 1e-6 * sdf::unreachable);
            SRESAMPLE_TEST_APPROXEQ(r.nx[i], sign, 1e-12);
            SRESAMPLE_TEST_APPROXEQ(r.ny[i], 0, 1e-12);
            SRESAMPLE_TEST_APPROXEQ(r.nz[i], 0, 1e-12);
        }
    }
    return 0;
}

// The parts on their own add up to the whole, and the value-only and normals-only samplings agree
// with it bitwise.
int test_parts_agree_with_the_whole() {
    const Spline s([](double x, double y, double z) {
        return std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + (z - 0.5) * (z - 0.5)) - 0.3;
    });
    for (auto &mesh : meshes()) {
        auto                v = mesh.view();
        Result              whole(v.nnodes), parts(v.nnodes);
        std::vector<double> w(v.nnodes);
        SRESAMPLE_TEST_EQ(sdf::surface_sample(v, s.grid, whole.out(), w.data(), Exec::host()), (int)SRESAMPLE_SUCCESS);

        sdf::SurfaceOutput<double> gap_only, normals_only;
        gap_only.gap = parts.gap.data();
        normals_only.normals[0] = parts.nx.data();
        normals_only.normals[1] = parts.ny.data();
        normals_only.normals[2] = parts.nz.data();
        SRESAMPLE_TEST_EQ(sdf::surface_sample(v, s.grid, gap_only, w.data(), Exec::host()), (int)SRESAMPLE_SUCCESS);
        SRESAMPLE_TEST_EQ(sdf::surface_sample(v, s.grid, normals_only, (double *)nullptr, Exec::host()), (int)SRESAMPLE_SUCCESS);

        // One thread at a time is not needed: each node's sums are formed in element order only
        // up to rounding, so compare with a tolerance.
        for (ptrdiff_t i = 0; i < v.nnodes; ++i) {
            SRESAMPLE_TEST_APPROXEQ(parts.gap[i], whole.gap[i], 1e-12);
            SRESAMPLE_TEST_APPROXEQ(parts.nx[i], whole.nx[i], 1e-12);
            SRESAMPLE_TEST_APPROXEQ(parts.ny[i], whole.ny[i], 1e-12);
            SRESAMPLE_TEST_APPROXEQ(parts.nz[i], whole.nz[i], 1e-12);
        }

        // A gap without a workspace is refused.
        SRESAMPLE_TEST_EQ(sdf::surface_sample(v, s.grid, gap_only, (double *)nullptr, Exec::host()), (int)SRESAMPLE_FAILURE);
    }
    return 0;
}

#ifdef SRESAMPLE_ENABLE_CUDA
namespace {
    template <typename T>
    T *to_device(const std::vector<T> &v) {
        T *d = nullptr;
        cudaMalloc(&d, v.size() * sizeof(T));
        cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
        return d;
    }

    template <typename T>
    std::vector<T> to_host(const T *const d, const size_t n) {
        std::vector<T> v(n);
        cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
        return v;
    }
}  // namespace

// The device kernels are the host's: the same results up to the order of the atomic sums.
int test_device_agrees_with_host() {
    const Spline s([](double x, double y, double z) {
        return std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + (z - 0.5) * (z - 0.5)) - 0.3;
    });
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    for (auto &mesh : meshes()) {
        auto                v = mesh.view();
        Result              host(v.nnodes);
        std::vector<double> w(v.nnodes);
        SRESAMPLE_TEST_EQ(sdf::surface_sample(v, s.grid, host.out(), w.data(), Exec::host()), (int)SRESAMPLE_SUCCESS);

        std::vector<int32_t *> d_e;
        for (auto &e : mesh.elems) d_e.push_back(to_device(e));
        std::vector<float *> d_x;
        for (int d = 0; d < 3; ++d) d_x.push_back(to_device(mesh.xyz[d]));
        std::vector<const int32_t *> e_ptrs(d_e.begin(), d_e.end());
        std::vector<const float *>   x_ptrs(d_x.begin(), d_x.end());
        const int32_t *const        *d_elements = to_device(e_ptrs);
        const float *const          *d_points   = to_device(x_ptrs);

        MeshView<float, int32_t> dv = v;
        dv.elements                 = d_elements;
        dv.points                   = d_points;

        BSplineGrid<float, float> dg = s.grid;
        float                    *d_coeffs = to_device(s.coeffs);
        dg.coeffs                          = d_coeffs + (s.grid.coeffs - s.coeffs.data());

        double *d_out = nullptr;
        cudaMalloc(&d_out, 5 * v.nnodes * sizeof(double));
        sdf::SurfaceOutput<double> out;
        out.gap = d_out;
        for (int d = 0; d < 3; ++d) out.normals[d] = d_out + (1 + d) * v.nnodes;
        double *const workspace = d_out + 4 * v.nnodes;

        SRESAMPLE_TEST_EQ(sdf::surface_sample(dv, dg, out, workspace, Exec::device(stream)), (int)SRESAMPLE_SUCCESS);
        cudaStreamSynchronize(stream);
        const auto dev = to_host(d_out, 4 * v.nnodes);

        double err = 0;
        for (ptrdiff_t i = 0; i < v.nnodes; ++i) {
            err = std::max(err, std::abs(dev[i] - host.gap[i]) / (1 + std::abs(host.gap[i])));
            err = std::max(err, std::abs(dev[v.nnodes + i] - host.nx[i]));
            err = std::max(err, std::abs(dev[2 * v.nnodes + i] - host.ny[i]));
            err = std::max(err, std::abs(dev[3 * v.nnodes + i] - host.nz[i]));
        }
        printf("%s: device vs host %g\n", name(mesh.type), err);
        SRESAMPLE_TEST_ASSERT(err < 1e-12);

        cudaFree(d_out);
        cudaFree(d_coeffs);
        cudaFree((void *)d_elements);
        cudaFree((void *)d_points);
        for (auto p : d_e) cudaFree(p);
        for (auto p : d_x) cudaFree(p);
    }
    cudaStreamDestroy(stream);
    return 0;
}
#endif

int main() {
    int err = 0;
    SRESAMPLE_RUN_TEST(test_linear_field_is_reproduced);
    SRESAMPLE_RUN_TEST(test_weights_sum_to_the_measure);
    SRESAMPLE_RUN_TEST(test_outside_the_grid);
    SRESAMPLE_RUN_TEST(test_parts_agree_with_the_whole);
#ifdef SRESAMPLE_ENABLE_CUDA
    SRESAMPLE_RUN_TEST(test_device_agrees_with_host);
#endif
    return err ? 1 : 0;
}
