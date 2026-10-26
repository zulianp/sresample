#include "sresample_surface_sample_cuda.hpp"

#include "sresample_cuda_base.cuh"
#include "sresample_instantiations.hpp"
#include "sresample_surface_sample_kernel.hpp"

namespace sresample {
    namespace sdf {
        namespace cuda {

            namespace {
                template <class Elem, typename TReal, typename TGeom, typename TIdx, typename TGrid>
                __global__ void surface_sample_kernel(const MeshView<TGeom, TIdx>     mesh,
                                                      const BSplineGrid<TGeom, TGrid> grid,
                                                      const SurfaceOutput<TReal>      out) {
                    for (ptrdiff_t e = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < mesh.nelements;
                         e += (ptrdiff_t)blockDim.x * gridDim.x) {
                        surface_sample_element<Elem>(e, mesh, grid, out);
                    }
                }

                template <class Elem, typename TReal, typename TGeom, typename TIdx>
                __global__ void surface_weights_kernel(const MeshView<TGeom, TIdx> mesh, TReal *const w) {
                    for (ptrdiff_t e = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < mesh.nelements;
                         e += (ptrdiff_t)blockDim.x * gridDim.x) {
                        surface_weights_element<Elem>(e, mesh, w);
                    }
                }

                template <typename TReal>
                __global__ void finish_kernel(const ptrdiff_t n, const TReal *const w, const SurfaceOutput<TReal> out) {
                    for (ptrdiff_t i = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; i < n; i += (ptrdiff_t)blockDim.x * gridDim.x) {
                        if (out.gap) {
                            // Lumped-mass pseudo-inverse (same pattern as blas.reciprocal): skip w_i == 0.
                            if (w[i]) out.gap[i] /= w[i];
                        }
                        if (out.normals[0]) {
                            const TReal nx    = out.normals[0][i];
                            const TReal ny    = out.normals[1][i];
                            const TReal nz    = out.normals[2][i];
                            const TReal denom = sqrt(nx * nx + ny * ny + nz * nz);
                            if (denom > 0) {
                                out.normals[0][i] = nx / denom;
                                out.normals[1][i] = ny / denom;
                                out.normals[2][i] = nz / denom;
                            }
                        }
                    }
                }

                template <typename TReal, typename TGeom, typename TIdx, typename TGrid>
                struct SampleLaunch {
                    const MeshView<TGeom, TIdx>     &mesh;
                    const BSplineGrid<TGeom, TGrid> &grid;
                    const SurfaceOutput<TReal>      &out;
                    void *const                      stream;

                    template <class Elem>
                    void operator()() const {
                        surface_sample_kernel<Elem, TReal, TGeom, TIdx, TGrid>
                                <<<n_blocks(mesh.nelements), block_size, 0, (cudaStream_t)stream>>>(mesh, grid, out);
                    }
                };

                template <typename TReal, typename TGeom, typename TIdx>
                struct WeightsLaunch {
                    const MeshView<TGeom, TIdx> &mesh;
                    TReal *const                 w;
                    void *const                  stream;

                    template <class Elem>
                    void operator()() const {
                        surface_weights_kernel<Elem, TReal, TGeom, TIdx>
                                <<<n_blocks(mesh.nelements), block_size, 0, (cudaStream_t)stream>>>(mesh, w);
                    }
                };
            }  // namespace

            template <typename TReal, typename TGeom, typename TIdx, typename TGrid>
            int surface_sample_local(const MeshView<TGeom, TIdx>     &mesh,
                                     const BSplineGrid<TGeom, TGrid> &grid,
                                     const SurfaceOutput<TReal>      &out,
                                     void *const                      stream) {
                if (!mesh.nelements) return SRESAMPLE_SUCCESS;
                const int err =
                        dispatch_shell<TReal, TGeom>(mesh.type, SampleLaunch<TReal, TGeom, TIdx, TGrid>{mesh, grid, out, stream});
                return err ? err : check_launch();
            }

            template <typename TReal, typename TGeom, typename TIdx>
            int surface_weights_local(const MeshView<TGeom, TIdx> &mesh, TReal *const w, void *const stream) {
                if (!mesh.nelements) return SRESAMPLE_SUCCESS;
                const int err = dispatch_shell<TReal, TGeom>(mesh.type, WeightsLaunch<TReal, TGeom, TIdx>{mesh, w, stream});
                return err ? err : check_launch();
            }

            template <typename TReal>
            int finish_surface_sample(const ptrdiff_t n, const TReal *const w, const SurfaceOutput<TReal> &out, void *const stream) {
                if (!n) return SRESAMPLE_SUCCESS;
                finish_kernel<TReal><<<n_blocks(n), block_size, 0, (cudaStream_t)stream>>>(n, w, out);
                return check_launch();
            }

            template <typename TReal>
            int zero(const ptrdiff_t n, TReal *const x, void *const stream) {
                if (!n || !x) return SRESAMPLE_SUCCESS;
                return cudaMemsetAsync(x, 0, n * sizeof(TReal), (cudaStream_t)stream) == cudaSuccess ? SRESAMPLE_SUCCESS
                                                                                                      : SRESAMPLE_FAILURE;
            }

#define SRESAMPLE_INSTANTIATE_SURFACE_SAMPLE_CUDA(TReal, TGeom, TIdx, TGrid) \
    template int surface_sample_local<TReal, TGeom, TIdx, TGrid>(           \
            const MeshView<TGeom, TIdx> &, const BSplineGrid<TGeom, TGrid> &, const SurfaceOutput<TReal> &, void *const);

#define SRESAMPLE_INSTANTIATE_SURFACE_WEIGHTS_CUDA(TReal, TGeom, TIdx) \
    template int surface_weights_local<TReal, TGeom, TIdx>(const MeshView<TGeom, TIdx> &, TReal *const, void *const);

#define SRESAMPLE_INSTANTIATE_FINISH_CUDA(TReal)                                                                           \
    template int finish_surface_sample<TReal>(const ptrdiff_t, const TReal *const, const SurfaceOutput<TReal> &, void *const); \
    template int zero<TReal>(const ptrdiff_t, TReal *const, void *const);

            SRESAMPLE_FOREACH_TYPES(SRESAMPLE_INSTANTIATE_SURFACE_SAMPLE_CUDA)
            SRESAMPLE_FOREACH_MESH_TYPES(SRESAMPLE_INSTANTIATE_SURFACE_WEIGHTS_CUDA)
            SRESAMPLE_FOREACH_REAL_TYPES(SRESAMPLE_INSTANTIATE_FINISH_CUDA)

        }  // namespace cuda
    }  // namespace sdf
}  // namespace sresample
