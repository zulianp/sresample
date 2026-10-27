# sresample

sresample transfers fields between regular grids and finite element meshes, on CPUs and GPUs.
It is a C++17 library of templated kernels, written once for the host and the device. It has no
required dependencies: OpenMP, CUDA and MPI are opt-in.

Its main user is [SFEM](https://github.com/zulianp/sfem), where it samples signed distance fields
(SDFs) of rigid obstacles for contact.

- **SDF sampling**: an interpolating cubic B-spline of a grid of SDF samples. The field, its
  gradient and its Hessian are continuous across cells, so contact normals and curvatures do not
  jump.
- **Surface sampling**: gap and unit normal at the nodes of a surface mesh of shells or beams
  (TRISHELL3, QUADSHELL4, BEAM2), weighted with the elements' dual basis.
- **Node sampling**: gap, unit normal and curvature of the obstacle at any set of points, for
  example a displaced mesh's nodes.
- **Distributed grids**: grids split into z-slabs across MPI ranks; each rank gathers the slab
  that covers its points.
- **Execution**: the same element kernels on the host (serial or OpenMP) and on the device
  (CUDA). Every device call is ordered on the stream it is given.
- **Types**: kernels are templated on the scalar, geometry, index and grid types; the build
  chooses which combinations are compiled.

## Quick start

Requirements: a C++17 compiler and CMake 3.20 or newer.

```sh
git clone https://github.com/zulianp/sresample.git
cd sresample
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
(cd build && ctest)
cmake --install build --prefix /path/to/install
```

Add `-DSRESAMPLE_ENABLE_OPENMP=ON`, `-DSRESAMPLE_ENABLE_CUDA=ON` or `-DSRESAMPLE_ENABLE_MPI=ON`
for threads, GPUs or distributed grids. On macOS with Apple Clang, OpenMP needs
`-DOpenMP_ROOT=$(brew --prefix libomp)`.

## Build options

| Option | Default | Effect |
|---|---|---|
| `SRESAMPLE_ENABLE_OPENMP` | `OFF` | Thread the host kernels with OpenMP. |
| `SRESAMPLE_ENABLE_CUDA` | `OFF` | Build the device kernels. `CMAKE_CUDA_ARCHITECTURES` defaults to 90; `SRESAMPLE_CUDA_ARCHITECTURES` overrides it for the library alone. |
| `SRESAMPLE_ENABLE_MPI` | `OFF` | Build the z-slab utilities (`src/mpi`) and their tests. |
| `SRESAMPLE_ENABLE_SMESH` | `OFF` | Find smesh, for demos that read meshes (there are none yet). |
| `SRESAMPLE_ENABLE_TESTING` | `ON` standalone, `OFF` as a subproject | Build the tests. |
| `SRESAMPLE_ENABLE_DEV_MODE` | `OFF` | Compile sresample's own targets with warnings as errors. |
| `SRESAMPLE_BUILD_SHARED_LIBS` | `OFF` | Build a shared library instead of a static one. |
| `SRESAMPLE_LOG_LEVEL` | `1` | Messages printed: 0 errors, 1 warnings, 2 info, 3 debug. |
| `SRESAMPLE_INSTANTIATIONS` | see below | Type tuples the templates are compiled for. |

### Type instantiations

The kernels are templates on four types: the real type of the results (`TReal`), the
coordinates (`TGeom`), the mesh indices (`TIdx`) and the grid values (`TGrid`).
`SRESAMPLE_INSTANTIATIONS` lists the tuples compiled into the library, as `real,geom,idx,grid`
separated by `;`, each type one of `float32`, `float64`, `int32` or `int64`. The default is:

```
float64,float32,int32,float32;float64,float32,int32,float64;float32,float32,int32,float32
```

Calling a kernel with a tuple that is not in the list fails at link time. Add the tuple to the
list to fix it.

## Using sresample in a CMake project

After `cmake --install`, find the package and link the target:

```cmake
find_package(SRESAMPLE REQUIRED)
target_link_libraries(my_app PRIVATE SRESAMPLE::sresample)
```

Point `CMAKE_PREFIX_PATH` at the install prefix if it is not a system location. The headers are
installed flat under `include/sresample`, so they are included by file name.

Alternatively, add the source tree to your build with `add_subdirectory(path/to/sresample)` and
link the same target. When the parent project already defines `OpenMP::OpenMP_CXX` or
`MPI::MPI_CXX`, sresample uses those targets. This is how SFEM builds it, with
`SFEM_ENABLE_SRESAMPLE` (on by default).

## Example: gap and normal on a surface mesh

The program below samples the signed distance to the plane z = 0.5 on a triangle at height
z = 0.75. Each node gets the gap −sdf = −0.25 and the normal (0, 0, 1).

```cpp
#include <cstdio>
#include <vector>

#include "sresample_bspline.hpp"
#include "sresample_surface_sample.hpp"

using namespace sresample;

int main() {
    // The signed distance to the plane z = 0.5, sampled on 33^3 nodes over [0, 1]^3 (x fastest).
    const ptrdiff_t n[3]             = {33, 33, 33};
    const ptrdiff_t sample_stride[3] = {1, n[0], n[0] * n[1]};
    const float     h                = 1.f / (n[0] - 1);

    std::vector<float> samples(n[0] * n[1] * n[2]);
    for (ptrdiff_t k = 0; k < n[2]; ++k)
        for (ptrdiff_t j = 0; j < n[1]; ++j)
            for (ptrdiff_t i = 0; i < n[0]; ++i) samples[i + j * sample_stride[1] + k * sample_stride[2]] = k * h - 0.5f;

    // Its cubic B-spline: coefficients padded by one node on every side.
    BSplineGrid<float, float> grid;
    ptrdiff_t                 size, offset;
    sdf::bspline_layout(n, grid.stride, &size, &offset);
    std::vector<float> coeffs(size);
    sdf::bspline_coefficients(n, sample_stride, samples.data(), grid.stride, coeffs.data() + offset);
    for (int d = 0; d < 3; ++d) {
        grid.n[d]      = n[d];
        grid.origin[d] = 0;
        grid.delta[d]  = h;
    }
    grid.coeffs = coeffs.data() + offset;

    // One triangle at height z = 0.75, as a structure of arrays.
    const int32_t  e0[1] = {0}, e1[1] = {1}, e2[1] = {2};
    const float    x[3] = {0.2f, 0.8f, 0.2f}, y[3] = {0.2f, 0.2f, 0.8f}, z[3] = {0.75f, 0.75f, 0.75f};
    const int32_t *elements[3] = {e0, e1, e2};
    const float   *points[3]   = {x, y, z};

    MeshView<float, int32_t> mesh;
    mesh.type      = ElemType::TRISHELL3;
    mesh.nelements = 1;
    mesh.nnodes    = 3;
    mesh.elements  = elements;
    mesh.points    = points;

    // Gap and unit normal at the nodes; the workspace holds the lumped mass.
    std::vector<double>        gap(3), nx(3), ny(3), nz(3), workspace(3);
    sdf::SurfaceOutput<double> out;
    out.gap        = gap.data();
    out.normals[0] = nx.data();
    out.normals[1] = ny.data();
    out.normals[2] = nz.data();

    if (sdf::surface_sample(mesh, grid, out, workspace.data(), Exec::host()) != SRESAMPLE_SUCCESS) return 1;

    for (int i = 0; i < 3; ++i) printf("node %d: gap %g, normal (%g, %g, %g)\n", i, gap[i], nx[i], ny[i], nz[i]);
    return 0;
}
```

To run the same call on the GPU, pass `Exec::device(stream)` and put the mesh arrays, the grid's
coefficients and the outputs in device memory.

## Concepts

- **`Exec`** (`sresample_base.hpp`): where a call's arrays live (`MemorySpace::HOST` or
  `DEVICE`) and, on the device, the `cudaStream_t` its work is ordered on. Device calls return
  without waiting for the stream; define `SRESAMPLE_DEBUG_SYNC` to wait after every launch.
- **`MeshView`** (`sresample_views.hpp`): a non-owning, structure-of-arrays view of a mesh.
  Node `v` of element `e` is `elements[v][e]`, and coordinate `d` of a node is
  `points[d][node]`.
- **`BSplineGrid`** (`sresample_views.hpp`): a regular grid's B-spline coefficients, with the
  number of nodes, origin, spacing and strides per axis. It is passed to kernels by value, so
  only the coefficient array must be in device memory.
- **Outside the grid**: samplers report the gap `sdf::unreachable` (10000), so a contact
  constraint there stays inactive.

The functions are documented where they are declared:

| Header | Contents |
|---|---|
| `sresample_bspline.hpp` | B-spline coefficients and point sampling with gradient and Hessian |
| `sresample_surface_sample.hpp` | Dual-basis gap and normals on shell and beam meshes |
| `sresample_node_sample.hpp` | Gap, normal and curvature at points |
| `sresample_zslab.hpp` | Z-slab gather and per-rank grid views (MPI) |
| `sresample_trishell3.hpp`, `sresample_quadshell4.hpp`, `sresample_beam2.hpp` | Element geometry, bases and quadrature |

## Repository layout

| Path | Contents |
|---|---|
| `src/base` | Execution and memory spaces, mesh and grid views, atomics, error and log macros |
| `src/quadrature` | Quadrature rules |
| `src/mesh` | Shell and beam elements |
| `src/sdf` | B-spline grids, surface and node sampling; device versions in `src/sdf/cuda` |
| `src/mpi` | Z-slab distribution of grids |
| `tests/` | One ctest per `test_*.cpp`; `tests/mpi` runs on 3 ranks |
| `scripts/check_streams.py` | The stream check (ctest `sresample_StreamGate`): every device launch names its stream |
| `legacy/` | SFEM's resampling code as it was imported, with its history. It is not built; code moves into `src/` as it is ported. |
| `cmake/` | Options, dependencies, tests and the package configuration |

## Development

Build with `-DSRESAMPLE_ENABLE_DEV_MODE=ON` to make warnings errors, and run `ctest` before you
commit. Each test's header comment says what it checks and, where recorded, the mutations it
catches. Device kernels must take their stream from `Exec`; `sresample_StreamGate` fails on any
launch without one.

## Authors

Patrick Zulian and Simone Riva.

## License

BSD 3-Clause; see [LICENSE](LICENSE).
