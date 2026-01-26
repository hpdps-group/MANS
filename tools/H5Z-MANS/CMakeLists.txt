cmake_minimum_required(VERSION 3.15)
if(DEFINED HDF5_PREFER_PARALLEL)
    set(HDF5_PREFER_PARALLEL ${HDF5_PREFER_PARALLEL})
endif()
find_package(HDF5 REQUIRED COMPONENTS C HL)
if(HDF5_PREFER_PARALLEL)
    find_package(MPI REQUIRED)
    if(NOT HDF5_IS_PARALLEL)
        message(WARNING "HDF5_PREFER_PARALLEL=ON but the found HDF5 is serial. Set HDF5_ROOT/CMAKE_PREFIX_PATH to a parallel HDF5 install if needed.")
    endif()
endif()
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
    ${HDF5_INCLUDE_DIRS}
)

# =========================================================
# Target 1: H5Z-MANS (Filter Plugin shared library)
# =========================================================
# Build a shared library named libH5Z-MANS.so; HDF5 will automatically load it at runtime.
add_library(H5Z-MANS SHARED src/H5Z-MANS.cpp)
target_link_libraries(H5Z-MANS PRIVATE mans_common ${HDF5_LIBRARIES} $<$<BOOL:${HDF5_PREFER_PARALLEL}>:MPI::MPI_C>)
set_target_properties(H5Z-MANS PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/plugins
    OUTPUT_NAME "H5Z-MANS"
    PREFIX "lib"
)


# =========================================================
# Target 2: H5Z-MANS_test
# =========================================================
add_executable(H5Z-MANS_test test/H5Z-MANS_test.cpp)
target_link_libraries(H5Z-MANS_test PRIVATE ${HDF5_LIBRARIES} $<$<BOOL:${HDF5_PREFER_PARALLEL}>:MPI::MPI_C>)
set_target_properties(H5Z-MANS_test PROPERTIES
    # output to build/bin/tools
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/h5z-mans
)
