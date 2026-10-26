// The z-slab gather (mpi/sresample_zslab.hpp): a grid split into uneven z-slabs across the ranks,
// every range of layers gathered on every rank equals the whole grid's, and the view of a rank's
// points covers them with the margin asked for. Run on several ranks.
//
// Mutation checks: the receive displacements left per rank (not summed) make the gather fail; the
// view's margin dropped makes the coverage check fail.
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

#include "sresample_test.hpp"
#include "sresample_zslab.hpp"

using namespace sresample;

namespace {
    const ptrdiff_t layer = 6;  // values per z-layer

    // The value of entry k of global layer z.
    float value(const ptrdiff_t z, const ptrdiff_t k) { return (float)(z * 100 + k); }

    // Rank r holds 3 (r + 2) layers.
    struct Slabs {
        int                rank, size;
        ptrdiff_t          nlocal, offset, nglobal;
        std::vector<float> data;

        explicit Slabs(MPI_Comm comm) {
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &size);
            nlocal  = 3 * (rank + 2);
            offset  = 0;
            nglobal = 0;
            for (int r = 0; r < size; ++r) {
                if (r < rank) offset += 3 * (r + 2);
                nglobal += 3 * (r + 2);
            }
            data.resize(nlocal * layer);
            for (ptrdiff_t z = 0; z < nlocal; ++z)
                for (ptrdiff_t k = 0; k < layer; ++k) data[z * layer + k] = value(offset + z, k);
        }
    };
}  // namespace

int test_gather_every_range(MPI_Comm comm) {
    const Slabs s(comm);
    for (ptrdiff_t b = 0; b < s.nglobal; ++b) {
        for (ptrdiff_t e = b + 1; e <= s.nglobal; ++e) {
            std::vector<float> out((e - b) * layer, -1);
            SRESAMPLE_TEST_EQ(mpi::zslab_gather(comm, s.nlocal, layer, s.data.data(), b, e, out.data()), (int)SRESAMPLE_SUCCESS);
            for (ptrdiff_t z = b; z < e; ++z)
                for (ptrdiff_t k = 0; k < layer; ++k) SRESAMPLE_TEST_EQ(out[(z - b) * layer + k], value(z, k));
        }
    }
    return 0;
}

int test_view_covers_the_points(MPI_Comm comm) {
    const Slabs     s(comm);
    const ptrdiff_t nlocal[3]  = {2, 3, s.nlocal};
    const ptrdiff_t nglobal[3] = {2, 3, s.nglobal};
    const ptrdiff_t stride[3]  = {1, 2, layer};
    const float     origin[3] = {0, 0, 0}, delta[3] = {1, 1, 0.5f};

    // Points in z around the middle of the grid, different per rank.
    const float z[3] = {0.5f * (float)(s.nglobal / 2) + 0.1f * s.rank, 0.5f * (float)(s.nglobal / 2) + 0.3f, 0.5f * (float)(s.nglobal / 2)};
    const ptrdiff_t margin = 2;

    float    *view = nullptr;
    ptrdiff_t nz;
    float     z0;
    SRESAMPLE_TEST_EQ(mpi::zslab_view(comm, 3, z, nlocal, nglobal, stride, origin, delta, s.data.data(), margin, &view, &nz, &z0),
                      (int)SRESAMPLE_SUCCESS);
    const ptrdiff_t first = (ptrdiff_t)(z0 / delta[2] + 0.5f);
    for (int i = 0; i < 3; ++i) {
        const ptrdiff_t cell = (ptrdiff_t)(z[i] / delta[2]);
        SRESAMPLE_TEST_ASSERT(first > 0 && cell - margin >= first);
        SRESAMPLE_TEST_ASSERT(first + nz < s.nglobal && cell + 1 + margin < first + nz);
    }
    for (ptrdiff_t l = 0; l < nz; ++l)
        for (ptrdiff_t k = 0; k < layer; ++k) SRESAMPLE_TEST_EQ(view[l * layer + k], value(first + l, k));
    free(view);
    return 0;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int err = 0;
    {
        MPI_Comm comm = MPI_COMM_WORLD;
        auto     run  = [&](int (*test)(MPI_Comm), const char *name) {
            int e = test(comm), any = 0;
            MPI_Allreduce(&e, &any, 1, MPI_INT, MPI_MAX, comm);
            int rank;
            MPI_Comm_rank(comm, &rank);
            if (!rank) printf("%s: %s\n", name, any ? "FAILED" : "ok");
            err += any;
        };
        run(test_gather_every_range, "test_gather_every_range");
        run(test_view_covers_the_points, "test_view_covers_the_points");
    }
    MPI_Finalize();
    return err ? 1 : 0;
}
