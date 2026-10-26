# sresample's options. Standalone, the library has no dependency at all; each one is opt-in.

option(SRESAMPLE_ENABLE_OPENMP "Thread the host kernels with OpenMP" OFF)
option(SRESAMPLE_ENABLE_CUDA "Build the device kernels" OFF)
option(SRESAMPLE_ENABLE_MPI "Build the distributed (z-slab) utilities" OFF)
option(SRESAMPLE_ENABLE_SMESH "Build the demos, which read meshes with smesh" OFF)
option(SRESAMPLE_ENABLE_DEV_MODE "Compile sresample's own targets with warnings as errors" OFF)
option(SRESAMPLE_BUILD_SHARED_LIBS "Build sresample as a shared library" OFF)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    option(SRESAMPLE_ENABLE_TESTING "Build the tests" ON)
else()
    option(SRESAMPLE_ENABLE_TESTING "Build the tests" OFF)
endif()

set(SRESAMPLE_LOG_LEVEL "1" CACHE STRING "Messages printed: 0 errors, 1 warnings, 2 info, 3 debug")

# The type tuples the templates are instantiated for, each "real,geometry,index,grid" with
# float32, float64, int32 or int64. The all-float32 tuple is there to find type cascades.
set(SRESAMPLE_INSTANTIATIONS
    "float64,float32,int32,float32;float64,float32,int32,float64;float32,float32,int32,float32"
    CACHE STRING "Type tuples real,geom,idx,grid, separated by ;")

# The flags of SRESAMPLE_ENABLE_DEV_MODE, the same as sfem's SFEM_ENABLE_DEV_MODE.
set(SRESAMPLE_DEV_CXX_FLAGS
    -Wall -Wextra -pedantic -Werror -Werror=enum-compare -Werror=delete-non-virtual-dtor -Werror=reorder
    -Werror=return-type -Wno-error=unknown-pragmas -Wno-missing-field-initializers)
set(SRESAMPLE_DEV_CUDA_FLAGS
    --Werror=all-warnings "-Xcompiler=-Wall,-Wextra,-Werror,-Wno-error=unknown-pragmas,-Wno-missing-field-initializers")

function(sresample_target_dev_flags target)
    if(SRESAMPLE_ENABLE_DEV_MODE)
        target_compile_options(${target} PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:${SRESAMPLE_DEV_CXX_FLAGS}>"
            "$<$<COMPILE_LANGUAGE:CUDA>:${SRESAMPLE_DEV_CUDA_FLAGS}>")
    endif()
endfunction()
