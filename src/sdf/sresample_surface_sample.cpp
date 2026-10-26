#include "sresample_surface_sample.hpp"

#include "sresample_instantiations.hpp"
#include "sresample_surface_sample_kernel.hpp"

#include <math.h>
#include <string.h>

#ifdef SRESAMPLE_ENABLE_CUDA
#include "sresample_surface_sample_cuda.hpp"
#endif

namespace sresample {
    namespace sdf {

        namespace {
            template <typename TReal, typename TGeom, typename TIdx, typename TGrid>
            struct SampleLoop {
                const MeshView<TGeom, TIdx>     &mesh;
                const BSplineGrid<TGeom, TGrid> &grid;
                const SurfaceOutput<TReal>      &out;

                template <class Elem>
                void operator()() const {
#pragma omp parallel for
                    for (ptrdiff_t e = 0; e < mesh.nelements; ++e) {
                        surface_sample_element<Elem>(e, mesh, grid, out);
                    }
                }
            };

            template <typename TReal, typename TGeom, typename TIdx>
            struct WeightsLoop {
                const MeshView<TGeom, TIdx> &mesh;
                TReal *const                 w;

                template <class Elem>
                void operator()() const {
#pragma omp parallel for
                    for (ptrdiff_t e = 0; e < mesh.nelements; ++e) {
                        surface_weights_element<Elem>(e, mesh, w);
                    }
                }
            };

            int not_built_for(const Exec &exec) {
                if (exec.space == MemorySpace::DEVICE) {
                    SRESAMPLE_LOG(0, "device execution requested, but sresample was built without CUDA\n");
                    return 1;
                }
                return 0;
            }
        }  // namespace

        template <typename TReal, typename TGeom, typename TIdx, typename TGrid>
        int surface_sample_local(const MeshView<TGeom, TIdx>     &mesh,
                                 const BSplineGrid<TGeom, TGrid> &grid,
                                 const SurfaceOutput<TReal>      &out,
                                 const Exec                      &exec) {
            if (!mesh.nelements) return SRESAMPLE_SUCCESS;
#ifdef SRESAMPLE_ENABLE_CUDA
            if (exec.space == MemorySpace::DEVICE) return cuda::surface_sample_local(mesh, grid, out, exec.stream);
#endif
            if (not_built_for(exec)) return SRESAMPLE_FAILURE;
            return dispatch_shell<TReal, TGeom>(mesh.type, SampleLoop<TReal, TGeom, TIdx, TGrid>{mesh, grid, out});
        }

        template <typename TReal, typename TGeom, typename TIdx>
        int surface_weights_local(const MeshView<TGeom, TIdx> &mesh, TReal *const w, const Exec &exec) {
            if (!mesh.nelements) return SRESAMPLE_SUCCESS;
#ifdef SRESAMPLE_ENABLE_CUDA
            if (exec.space == MemorySpace::DEVICE) return cuda::surface_weights_local(mesh, w, exec.stream);
#endif
            if (not_built_for(exec)) return SRESAMPLE_FAILURE;
            return dispatch_shell<TReal, TGeom>(mesh.type, WeightsLoop<TReal, TGeom, TIdx>{mesh, w});
        }

        template <typename TReal>
        int finish_surface_sample(const ptrdiff_t n, const TReal *const w, const SurfaceOutput<TReal> &out, const Exec &exec) {
#ifdef SRESAMPLE_ENABLE_CUDA
            if (exec.space == MemorySpace::DEVICE) return cuda::finish_surface_sample(n, w, out, exec.stream);
#endif
            if (not_built_for(exec)) return SRESAMPLE_FAILURE;

            if (out.gap) {
                // Lumped-mass pseudo-inverse (same pattern as blas.reciprocal): skip w_i == 0.
                TReal *const g = out.gap;
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    if (w[i]) g[i] /= w[i];
                }
            }

            if (out.normals[0]) {
                TReal *const nx = out.normals[0];
                TReal *const ny = out.normals[1];
                TReal *const nz = out.normals[2];
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    const TReal denom = sqrt(nx[i] * nx[i] + ny[i] * ny[i] + nz[i] * nz[i]);
                    if (denom > 0) {
                        nx[i] /= denom;
                        ny[i] /= denom;
                        nz[i] /= denom;
                    }
                }
            }

            return SRESAMPLE_SUCCESS;
        }

        template <typename TReal, typename TGeom, typename TIdx, typename TGrid>
        int surface_sample(const MeshView<TGeom, TIdx>     &mesh,
                           const BSplineGrid<TGeom, TGrid> &grid,
                           const SurfaceOutput<TReal>      &out,
                           TReal *const                     workspace,
                           const Exec                      &exec) {
            if (!mesh.nelements) return SRESAMPLE_SUCCESS;
            if (out.gap && !workspace) {
                SRESAMPLE_LOG(0, "surface_sample: the gap needs a workspace of nnodes entries\n");
                return SRESAMPLE_FAILURE;
            }

            const ptrdiff_t n = mesh.nnodes;
#ifdef SRESAMPLE_ENABLE_CUDA
            if (exec.space == MemorySpace::DEVICE) {
                cuda::zero(n, out.gap, exec.stream);
                for (int d = 0; d < 3; ++d) cuda::zero(n, out.normals[d], exec.stream);
                cuda::zero(n, out.gap ? workspace : nullptr, exec.stream);
            } else
#endif
            {
                if (not_built_for(exec)) return SRESAMPLE_FAILURE;
                if (out.gap) memset(out.gap, 0, n * sizeof(TReal));
                // The local kernels accumulate the normals too
                for (int d = 0; d < 3; ++d)
                    if (out.normals[d]) memset(out.normals[d], 0, n * sizeof(TReal));
                if (out.gap) memset(workspace, 0, n * sizeof(TReal));
            }

            if (surface_sample_local(mesh, grid, out, exec) != SRESAMPLE_SUCCESS) return SRESAMPLE_FAILURE;
            if (out.gap && surface_weights_local(mesh, workspace, exec) != SRESAMPLE_SUCCESS) return SRESAMPLE_FAILURE;
            return finish_surface_sample(n, (const TReal *)workspace, out, exec);
        }

#define SRESAMPLE_INSTANTIATE_SURFACE_SAMPLE(TReal, TGeom, TIdx, TGrid)                                                            \
    template int surface_sample_local<TReal, TGeom, TIdx, TGrid>(                                                                  \
            const MeshView<TGeom, TIdx> &, const BSplineGrid<TGeom, TGrid> &, const SurfaceOutput<TReal> &, const Exec &);         \
    template int surface_sample<TReal, TGeom, TIdx, TGrid>(                                                                        \
            const MeshView<TGeom, TIdx> &, const BSplineGrid<TGeom, TGrid> &, const SurfaceOutput<TReal> &, TReal *const, const Exec &);

#define SRESAMPLE_INSTANTIATE_SURFACE_WEIGHTS(TReal, TGeom, TIdx) \
    template int surface_weights_local<TReal, TGeom, TIdx>(const MeshView<TGeom, TIdx> &, TReal *const, const Exec &);

#define SRESAMPLE_INSTANTIATE_FINISH(TReal) \
    template int finish_surface_sample<TReal>(const ptrdiff_t, const TReal *const, const SurfaceOutput<TReal> &, const Exec &);

        SRESAMPLE_FOREACH_TYPES(SRESAMPLE_INSTANTIATE_SURFACE_SAMPLE)
        SRESAMPLE_FOREACH_MESH_TYPES(SRESAMPLE_INSTANTIATE_SURFACE_WEIGHTS)
        SRESAMPLE_FOREACH_REAL_TYPES(SRESAMPLE_INSTANTIATE_FINISH)

    }  // namespace sdf
}  // namespace sresample
