cmake_minimum_required(VERSION 3.20)

project(sresample VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(cmake/SRESAMPLEOptions.cmake)
include(cmake/SRESAMPLEDependencies.cmake)

# Sources: one folder per concern, each with its device sources in a cuda/ subfolder. legacy/
# holds sfem's resampling code as it was imported, not built.
set(SRESAMPLE_MODULES base quadrature mesh sdf)
if(SRESAMPLE_ENABLE_MPI)
    list(APPEND SRESAMPLE_MODULES mpi)
endif()

set(SRESAMPLE_SOURCES "")
set(SRESAMPLE_HEADERS "")
set(SRESAMPLE_INCLUDE_DIRS "")
foreach(module ${SRESAMPLE_MODULES})
    file(GLOB module_sources CONFIGURE_DEPENDS "src/${module}/*.cpp")
    file(GLOB module_headers CONFIGURE_DEPENDS "src/${module}/*.hpp")
    list(APPEND SRESAMPLE_SOURCES ${module_sources})
    list(APPEND SRESAMPLE_HEADERS ${module_headers})
    list(APPEND SRESAMPLE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src/${module}")
    if(SRESAMPLE_ENABLE_CUDA)
        file(GLOB module_cuda_sources CONFIGURE_DEPENDS "src/${module}/cuda/*.cu")
        file(GLOB module_cuda_headers CONFIGURE_DEPENDS "src/${module}/cuda/*.hpp" "src/${module}/cuda/*.cuh")
        list(APPEND SRESAMPLE_SOURCES ${module_cuda_sources})
        list(APPEND SRESAMPLE_HEADERS ${module_cuda_headers})
        list(APPEND SRESAMPLE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src/${module}/cuda")
    endif()
endforeach()

configure_file(cmake/sresample_config.hpp.in "${CMAKE_CURRENT_BINARY_DIR}/sresample_config.hpp")
configure_file(cmake/sresample_instantiations.hpp.in "${CMAKE_CURRENT_BINARY_DIR}/sresample_instantiations.hpp")
list(APPEND SRESAMPLE_HEADERS
    "${CMAKE_CURRENT_BINARY_DIR}/sresample_config.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/sresample_instantiations.hpp")

if(SRESAMPLE_BUILD_SHARED_LIBS)
    add_library(sresample SHARED ${SRESAMPLE_SOURCES})
else()
    add_library(sresample STATIC ${SRESAMPLE_SOURCES})
endif()
add_library(SRESAMPLE::sresample ALIAS sresample)

foreach(dir ${SRESAMPLE_INCLUDE_DIRS})
    target_include_directories(sresample PUBLIC $<BUILD_INTERFACE:${dir}>)
endforeach()
target_include_directories(sresample PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:include/sresample>)

set_target_properties(sresample PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(SRESAMPLE_ENABLE_CUDA)
    set_target_properties(sresample PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    if(DEFINED SRESAMPLE_CUDA_ARCHITECTURES)
        set_target_properties(sresample PROPERTIES CUDA_ARCHITECTURES "${SRESAMPLE_CUDA_ARCHITECTURES}")
    endif()
    target_link_libraries(sresample PUBLIC CUDA::cudart)
endif()
if(SRESAMPLE_ENABLE_OPENMP)
    target_link_libraries(sresample PUBLIC OpenMP::OpenMP_CXX)
    if(SRESAMPLE_ENABLE_CUDA)
        target_compile_options(sresample PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>")
    endif()
endif()
if(SRESAMPLE_ENABLE_MPI)
    target_link_libraries(sresample PUBLIC MPI::MPI_CXX)
endif()
sresample_target_dev_flags(sresample)

# Install
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS sresample EXPORT SRESAMPLETargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES ${SRESAMPLE_HEADERS} DESTINATION include/sresample)
install(EXPORT SRESAMPLETargets NAMESPACE SRESAMPLE:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sresample)
configure_package_config_file(cmake/SRESAMPLEConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/SRESAMPLEConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sresample)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/SRESAMPLEConfigVersion.cmake"
    COMPATIBILITY SameMinorVersion)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SRESAMPLEConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/SRESAMPLEConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sresample)

if(SRESAMPLE_ENABLE_TESTING)
    enable_testing()
    include(cmake/SRESAMPLETests.cmake)
endif()
