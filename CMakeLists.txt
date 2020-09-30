list(APPEND CMAKE_MODULE_PATH "C:/vcpkg/package/openvdb_x64-windows/share/openvdb")
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(CMAKE_GENERATOR_PLATFORM "x64")
cmake_minimum_required(VERSION 3.9)
project(LBVHPaper LANGUAGES CUDA CXX)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT LBVHPaper)
# Using cpp17
set(CMAKE_CXX_STANDARD 17)

#CUDA
find_package(CUDA)
enable_language(CUDA) 
include_directories(${CMAKE_BINARY_DIR}/src)

#OPENEXR
find_package(OPENEXR)
find_path(OPENEXR_INCLUDE_DIRS NAMES openexr/openexr.h)
find_library(OPENEXR_LIBRARIES NAMES openexr)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenVDB REQUIRED_VARS OPENEXR_INCLUDE_DIRS OPENEXR_LIBRARIES)

#TBB
find_path(TBB_INCLUDE_DIRS NAMES tbb/tbb.h)
find_library(TBB_LIBRARIES NAMES tbb)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TBB REQUIRED_VARS TBB_INCLUDE_DIRS TBB_LIBRARIES)

#OPENVDB
find_path(OPENVDB_INCLUDE_DIRS NAMES openvdb/openvdb.h)
find_library(OPENVDB_LIBRARIES NAMES openvdb)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenVDB REQUIRED_VARS OPENVDB_INCLUDE_DIRS OPENVDB_LIBRARIES)

IF( OPENVDB_FOUND)
	message(STATUS "OK openvdb_found with success!")
ENDIF()

# GLM library
include_directories(includes/glm)

#Workaround with Half library not working because of dll missing from correct build folder.
file(COPY "C:/vcpkg/installed/x86-windows/bin/Half-2_3.dll" DESTINATION ${CMAKE_BINARY_DIR}/Debug)
file(COPY "C:/vcpkg/installed/x86-windows/bin/tbb.dll" DESTINATION ${CMAKE_BINARY_DIR}/Debug)
file(COPY "C:/vcpkg/installed/x64-windows/bin/Half-2_3.dll" DESTINATION ${CMAKE_BINARY_DIR}/Debug)
file(COPY "C:/vcpkg/installed/x64-windows/bin/tbb.dll" DESTINATION ${CMAKE_BINARY_DIR}/Debug)

# Sources files
file(GLOB SOURCE_FILES
    "src/*.h"
    "src/*.cpp"
    "src/*.cu"
    "src/*.cuh"
)

include_directories(${CUDA_INCLUDE_DIRS} ${OPENVDB_INCLUDE_DIRS} ${TBB_INCLUDE_DIRS} ${OPENEXR_INCLUDE_DIRS})
add_executable(LBVHPaper ${SOURCE_FILES} )
target_link_libraries(LBVHPaper ${CUDA_LIBRARIES} ${OPENVDB_LIBRARIES} ${TBB_LIBRARIES} ${OPENEXR_LIBRARIES})
