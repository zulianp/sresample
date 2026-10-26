#include "sresample_zslab.hpp"

#include <stdlib.h>

#include <algorithm>
#include <vector>

namespace sresample {
    namespace mpi {

        namespace {
            template <typename T>
            MPI_Datatype mpi_type();
            template <>
            MPI_Datatype mpi_type<float>() {
                return MPI_FLOAT;
            }
            template <>
            MPI_Datatype mpi_type<double>() {
                return MPI_DOUBLE;
            }

            int check(const int err) { return err == MPI_SUCCESS ? SRESAMPLE_SUCCESS : SRESAMPLE_FAILURE; }
        }  // namespace

        template <typename TGrid>
        int zslab_gather(MPI_Comm           comm,
                         const ptrdiff_t    nlocal_z,
                         const ptrdiff_t    layer_size,
                         const TGrid *const in,
                         const ptrdiff_t    z_begin,
                         const ptrdiff_t    z_end,
                         TGrid *const       out) {
            int rank, size;
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &size);

            // The layers each rank holds, in rank order.
            std::vector<long> parts(size + 1, 0);
            parts[rank + 1] = (long)nlocal_z;
            if (check(MPI_Allreduce(MPI_IN_PLACE, &parts[1], size, MPI_LONG, MPI_SUM, comm))) return SRESAMPLE_FAILURE;
            for (int r = 1; r < size; r++) {
                parts[r + 1] += parts[r];
            }

            // Counted in layers, so a slab of any size fits the int counts.
            std::vector<int> recv_count(size), recv_displs(size), send_count(size), send_displs(size);
            for (int r = 0; r < size; r++) {
                const long bmin = std::max(parts[r], (long)z_begin);
                const long bmax = std::min(parts[r + 1], (long)z_end);
                recv_count[r]   = (int)std::max(0l, bmax - bmin);
                // From which offset process r needs to send data to this rank
                recv_displs[r] = (int)(bmin - parts[r]);
            }

            if (check(MPI_Alltoall(recv_count.data(), 1, MPI_INT, send_count.data(), 1, MPI_INT, comm))) return SRESAMPLE_FAILURE;
            if (check(MPI_Alltoall(recv_displs.data(), 1, MPI_INT, send_displs.data(), 1, MPI_INT, comm))) return SRESAMPLE_FAILURE;

            recv_displs[0] = 0;
            for (int r = 1; r < size; r++) {
                recv_displs[r] = recv_displs[r - 1] + recv_count[r - 1];
            }

            MPI_Datatype layer;
            MPI_Type_contiguous((int)layer_size, mpi_type<TGrid>(), &layer);
            MPI_Type_commit(&layer);
            const int err = MPI_Alltoallv(
                    in, send_count.data(), send_displs.data(), layer, out, recv_count.data(), recv_displs.data(), layer, comm);
            MPI_Type_free(&layer);
            return check(err);
        }

        template <typename TGeom, typename TGrid>
        int zslab_view(MPI_Comm                                  comm,
                       const ptrdiff_t                           nnodes,
                       const TGeom *const SRESAMPLE_RESTRICT     z_coordinate,
                       const ptrdiff_t *const                    nlocal,
                       const ptrdiff_t *const SRESAMPLE_RESTRICT nglobal,
                       const ptrdiff_t *const SRESAMPLE_RESTRICT stride,
                       const TGeom *const                        origin,
                       const TGeom *const SRESAMPLE_RESTRICT     delta,
                       const TGrid *const                        grid,
                       const ptrdiff_t                           z_margin,
                       TGrid                                   **view,
                       ptrdiff_t                                *z_nlocal_out,
                       TGeom *const SRESAMPLE_RESTRICT           z_origin_out) {
            int rank, size;
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &size);

            if (size == 1) {
                SRESAMPLE_LOG(0, "[%d] zslab_view cannot be used in serial runs!\n", rank);
                return SRESAMPLE_FAILURE;
            }

            TGeom zmin = origin[2], zmax = origin[2];
            if (nnodes) {
                zmin = z_coordinate[0];
                zmax = z_coordinate[0];
                for (ptrdiff_t i = 1; i < nnodes; i++) {
                    zmin = std::min(zmin, z_coordinate[i]);
                    zmax = std::max(zmax, z_coordinate[i]);
                }
            }

            // // Compute Local z-tile
            ptrdiff_t start = (zmin - origin[2]) / delta[2];
            ptrdiff_t end   = (zmax - origin[2]) / delta[2];

            // Make sure we are inside the grid and get also the required margin for resampling
            start = std::max(ptrdiff_t(0), start - 1 - z_margin);
            end   = std::min(nglobal[2], end + 2 + z_margin);  // 1 for the rightside of the cell 1 for the exclusive range

            // Points beyond either end of the grid still get two layers, the least a grid can have
            start = std::min(start, nglobal[2] - 2);
            end   = std::max(end, start + 2);

            const ptrdiff_t nz  = end - start;
            TGrid *const    out = (TGrid *)malloc(nz * stride[2] * sizeof(TGrid));

            if (zslab_gather(comm, nlocal[2], stride[2], grid, start, end, out) != SRESAMPLE_SUCCESS) {
                free(out);
                return SRESAMPLE_FAILURE;
            }

            *view         = out;
            *z_nlocal_out = nz;
            *z_origin_out = origin[2] + start * delta[2];
            return SRESAMPLE_SUCCESS;
        }

        template int zslab_gather<float>(MPI_Comm, const ptrdiff_t, const ptrdiff_t, const float *const, const ptrdiff_t, const ptrdiff_t, float *const);
        template int zslab_gather<double>(MPI_Comm, const ptrdiff_t, const ptrdiff_t, const double *const, const ptrdiff_t, const ptrdiff_t, double *const);

#define SRESAMPLE_INSTANTIATE_ZSLAB_VIEW(TGeom, TGrid)                                       \
    template int zslab_view<TGeom, TGrid>(MPI_Comm,                                          \
                                          const ptrdiff_t,                                   \
                                          const TGeom *const,                                \
                                          const ptrdiff_t *const,                            \
                                          const ptrdiff_t *const,                            \
                                          const ptrdiff_t *const,                            \
                                          const TGeom *const,                                \
                                          const TGeom *const,                                \
                                          const TGrid *const,                                \
                                          const ptrdiff_t,                                   \
                                          TGrid **,                                          \
                                          ptrdiff_t *,                                       \
                                          TGeom *const);

        SRESAMPLE_INSTANTIATE_ZSLAB_VIEW(float, float)
        SRESAMPLE_INSTANTIATE_ZSLAB_VIEW(float, double)
        SRESAMPLE_INSTANTIATE_ZSLAB_VIEW(double, double)

    }  // namespace mpi
}  // namespace sresample
