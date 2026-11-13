// The cell list (src/search): box-against-box queries against brute force, on boxes whose sizes
// span six decades (several levels), with duplicates and degenerate boxes; every overlapping pair
// exactly once. On the device too when sresample has CUDA.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sresample_base.hpp"  // sresample_config.hpp: SRESAMPLE_ENABLE_CUDA

#ifdef SRESAMPLE_ENABLE_CUDA
#include "sresample_gpu_runtime.cuh"
#endif

#include "sresample_cell_list.hpp"
#include "sresample_cell_list_kernel.hpp"
#include "sresample_test.hpp"

using namespace sresample;
using namespace sresample::search;

namespace {

    unsigned long long rng_state = 0x2545F4914F6CDD1DULL;
    double             uniform() {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        return (double)(rng_state >> 11) * (1.0 / 9007199254740992.0);
    }

    struct BoxSet {
        ptrdiff_t n{0};
        float    *lo[3]{nullptr, nullptr, nullptr};
        float    *hi[3]{nullptr, nullptr, nullptr};
        Boxes     view() const {
            Boxes b;
            b.n = n;
            for (int d = 0; d < 3; ++d) {
                b.lo[d] = lo[d];
                b.hi[d] = hi[d];
            }
            return b;
        }
        void free_all() {
            for (int d = 0; d < 3; ++d) {
                free(lo[d]);
                free(hi[d]);
            }
        }
    };

    // n boxes in [0, 1]^3, sizes log-uniform in [smin, smax]; some duplicates and flat boxes.
    BoxSet random_boxes(const ptrdiff_t n, const double smin, const double smax) {
        BoxSet s;
        s.n = n;
        for (int d = 0; d < 3; ++d) {
            s.lo[d] = (float *)malloc(sizeof(float) * n);
            s.hi[d] = (float *)malloc(sizeof(float) * n);
        }
        for (ptrdiff_t b = 0; b < n; ++b) {
            if (b > 0 && uniform() < 0.02) {  // a duplicate
                for (int d = 0; d < 3; ++d) {
                    s.lo[d][b] = s.lo[d][b - 1];
                    s.hi[d][b] = s.hi[d][b - 1];
                }
                continue;
            }
            const double size = smin * pow(smax / smin, uniform());
            for (int d = 0; d < 3; ++d) {
                const double c = uniform();
                const double e = uniform() < 0.02 ? 0.0 : size * (0.3 + 0.7 * uniform());
                s.lo[d][b]     = (float)(c - e / 2);
                s.hi[d][b]     = (float)(c + e / 2);
            }
        }
        return s;
    }

    int compare_pairs(const void *a, const void *b) {
        const long long x = *(const long long *)a, y = *(const long long *)b;
        return x < y ? -1 : (x > y ? 1 : 0);
    }

    // The overlapping pairs by brute force, as sorted keys q * nb + b.
    long long *brute_force(const BoxSet &tree, const BoxSet &queries, ptrdiff_t *np) {
        ptrdiff_t  cap = 1024, n = 0;
        long long *p   = (long long *)malloc(sizeof(long long) * cap);
        for (ptrdiff_t q = 0; q < queries.n; ++q) {
            const float qlo[3] = {queries.lo[0][q], queries.lo[1][q], queries.lo[2][q]};
            const float qhi[3] = {queries.hi[0][q], queries.hi[1][q], queries.hi[2][q]};
            for (ptrdiff_t b = 0; b < tree.n; ++b) {
                const float blo[3] = {tree.lo[0][b], tree.lo[1][b], tree.lo[2][b]};
                const float bhi[3] = {tree.hi[0][b], tree.hi[1][b], tree.hi[2][b]};
                if (!boxes_overlap(qlo, qhi, blo, bhi)) continue;
                if (n == cap) {
                    cap *= 2;
                    p = (long long *)realloc(p, sizeof(long long) * cap);
                }
                p[n++] = (long long)q * tree.n + b;
            }
        }
        qsort(p, n, sizeof(long long), compare_pairs);
        *np = n;
        return p;
    }

    // The cell list's pairs (host or device), as sorted keys; fails on a duplicate.
    int cell_list_pairs(const BoxSet &tree, const BoxSet &queries, const bool device, long long **out, ptrdiff_t *np) {
        CellListOptions options;
        options.max_columns = 1 << 16;
        CellList  list;
        ptrdiff_t total = 0;
        int      *pq = nullptr, *pb = nullptr;
        if (!device) {
            if (cell_list_build(tree.view(), options, &list, Exec::host())) return 1;
            ptrdiff_t *ptr = (ptrdiff_t *)malloc(sizeof(ptrdiff_t) * (queries.n + 1));
            if (cell_list_count(list, queries.view(), ptr, Exec::host())) return 1;
            if (exclusive_scan(ptr, queries.n, &total, Exec::host())) return 1;
            pq = (int *)malloc(sizeof(int) * (total ? total : 1));
            pb = (int *)malloc(sizeof(int) * (total ? total : 1));
            if (cell_list_fill(list, queries.view(), ptr, pq, pb, Exec::host())) return 1;
            free(ptr);
            cell_list_destroy(&list);
        } else {
#ifdef SRESAMPLE_ENABLE_CUDA
            // Copies of both box sets on the device.
            BoxSet dt = tree, dq = queries;
            for (int d = 0; d < 3; ++d) {
                cudaMalloc(&dt.lo[d], sizeof(float) * tree.n);
                cudaMalloc(&dt.hi[d], sizeof(float) * tree.n);
                cudaMalloc(&dq.lo[d], sizeof(float) * queries.n);
                cudaMalloc(&dq.hi[d], sizeof(float) * queries.n);
                cudaMemcpy(dt.lo[d], tree.lo[d], sizeof(float) * tree.n, cudaMemcpyHostToDevice);
                cudaMemcpy(dt.hi[d], tree.hi[d], sizeof(float) * tree.n, cudaMemcpyHostToDevice);
                cudaMemcpy(dq.lo[d], queries.lo[d], sizeof(float) * queries.n, cudaMemcpyHostToDevice);
                cudaMemcpy(dq.hi[d], queries.hi[d], sizeof(float) * queries.n, cudaMemcpyHostToDevice);
            }
            if (cell_list_build(dt.view(), options, &list, Exec::device())) return 1;
            ptrdiff_t *ptr = nullptr;
            cudaMalloc(&ptr, sizeof(ptrdiff_t) * (queries.n + 1));
            if (cell_list_count(list, dq.view(), ptr, Exec::device())) return 1;
            if (exclusive_scan(ptr, queries.n, &total, Exec::device())) return 1;
            int *dpq = nullptr, *dpb = nullptr;
            cudaMalloc(&dpq, sizeof(int) * (total ? total : 1));
            cudaMalloc(&dpb, sizeof(int) * (total ? total : 1));
            if (cell_list_fill(list, dq.view(), ptr, dpq, dpb, Exec::device())) return 1;
            pq = (int *)malloc(sizeof(int) * (total ? total : 1));
            pb = (int *)malloc(sizeof(int) * (total ? total : 1));
            cudaMemcpy(pq, dpq, sizeof(int) * total, cudaMemcpyDeviceToHost);
            cudaMemcpy(pb, dpb, sizeof(int) * total, cudaMemcpyDeviceToHost);
            cudaFree(dpq);
            cudaFree(dpb);
            cudaFree(ptr);
            cell_list_destroy(&list);
            for (int d = 0; d < 3; ++d) {
                cudaFree(dt.lo[d]);
                cudaFree(dt.hi[d]);
                cudaFree(dq.lo[d]);
                cudaFree(dq.hi[d]);
            }
#else
            return 1;
#endif
        }
        long long *p = (long long *)malloc(sizeof(long long) * (total ? total : 1));
        for (ptrdiff_t k = 0; k < total; ++k) p[k] = (long long)pq[k] * tree.n + pb[k];
        qsort(p, total, sizeof(long long), compare_pairs);
        free(pq);
        free(pb);
        for (ptrdiff_t k = 1; k < total; ++k) {
            if (p[k] == p[k - 1]) {
                fprintf(stderr, "duplicate pair %lld\n", p[k]);
                return 1;
            }
        }
        *out = p;
        *np  = total;
        return 0;
    }

    int check(const bool device, const double smin, const double smax) {
        BoxSet     tree = random_boxes(4000, smin, smax), queries = random_boxes(1500, smin, smax);
        ptrdiff_t  nb = 0, nc = 0;
        long long *pb = brute_force(tree, queries, &nb);
        long long *pc = nullptr;
        SRESAMPLE_TEST_ASSERT(cell_list_pairs(tree, queries, device, &pc, &nc) == 0);
        printf("  %s, sizes %g..%g: %td pairs\n", device ? "device" : "host", smin, smax, nb);
        SRESAMPLE_TEST_EQ(nb, nc);
        SRESAMPLE_TEST_ASSERT(memcmp(pb, pc, sizeof(long long) * nb) == 0);
        free(pb);
        free(pc);
        tree.free_all();
        queries.free_all();
        return 0;
    }

    int test_host_uniform() { return check(false, 0.01, 0.02); }
    int test_host_six_decades() { return check(false, 1e-6, 1.0); }
#ifdef SRESAMPLE_ENABLE_CUDA
    int test_device_uniform() { return check(true, 0.01, 0.02); }
    int test_device_six_decades() { return check(true, 1e-6, 1.0); }
#endif

}  // namespace

int main() {
    int err = 0;
    SRESAMPLE_RUN_TEST(test_host_uniform);
    SRESAMPLE_RUN_TEST(test_host_six_decades);
#ifdef SRESAMPLE_ENABLE_CUDA
    SRESAMPLE_RUN_TEST(test_device_uniform);
    SRESAMPLE_RUN_TEST(test_device_six_decades);
#endif
    return err ? EXIT_FAILURE : EXIT_SUCCESS;
}
