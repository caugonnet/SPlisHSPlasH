################################################################################
# general CMake and project setup
################################################################################
cmake_minimum_required(VERSION 3.10)

project(SPlisHSPlasH)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(Common)
if (NOT WIN32)
	message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

################################################################################
# Optional CUDA DFSPH backend (CCCL + CUDASTF). OFF by default so the stock
# CPU-only build and every existing solver are completely unaffected.
################################################################################
option(USE_CUDA_DFSPH "Build the optional device-resident DFSPH CUDA backend (CCCL + CUDASTF)" OFF)
if (USE_CUDA_DFSPH)
	# Local CCCL checkout that carries the deferred-args / CUDASTF features.
	set(CCCL_DIR "$ENV{HOME}/git/caugonnet_cccl" CACHE PATH "Path to the local CCCL checkout")
	set(CUDA_DFSPH_ARCHITECTURES "86" CACHE STRING "CUDA architectures for the DFSPH backend")

	enable_language(CUDA)
	find_package(CUDAToolkit REQUIRED)

	if (NOT EXISTS "${CCCL_DIR}/libcudacxx/include")
		message(FATAL_ERROR "USE_CUDA_DFSPH=ON but CCCL not found at CCCL_DIR='${CCCL_DIR}'.")
	endif()

	# Make Simulation.cpp / the facade compile the DFSPH_CUDA registration.
	add_definitions(-DUSE_CUDA_DFSPH)
	message(STATUS "CUDA DFSPH backend enabled (CCCL_DIR=${CCCL_DIR}, arch=${CUDA_DFSPH_ARCHITECTURES})")
endif()

set(ExternalInstallDir "${CMAKE_BINARY_DIR}/extern/install" CACHE INTERNAL "")
set(EXT_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE INTERNAL "")
if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
	set(EXT_CMAKE_BUILD_TYPE "Release" CACHE INTERNAL "")
endif()

set(TOPLEVEL_INCLUDE_DIR ${PROJECT_SOURCE_DIR})

################################################################################
# foreign external libraries
################################################################################
add_subdirectory(extern/zlib)
add_subdirectory(extern/partio)
add_subdirectory(extern/md5)
add_subdirectory(extern/tinyexpr)
if (NOT SPH_LIBS_ONLY)
	add_subdirectory(extern/glfw)
	add_subdirectory(extern/imgui)
	add_subdirectory(extern/nfd)
	if (USE_PYTHON_BINDINGS)
		add_subdirectory(extern/pybind)
	endif ()
endif()

## Eigen3 is used by most of the libraries that follow
find_package(Eigen3 REQUIRED)
add_definitions(-DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -idirafter${EIGEN3_INCLUDE_DIR}")
endif()

################################################################################
# own external libraries
################################################################################
include(SetUpExternalProjects)

################################################################################
# internal libraries
################################################################################
add_subdirectory(SPlisHSPlasH)
add_subdirectory(Utilities)

################################################################################
# executables
################################################################################
if (NOT SPH_LIBS_ONLY)
	include(DataCopyTargets)
	add_subdirectory(Tools)
	add_subdirectory(Simulator)
	add_subdirectory(Tests)
	if (USE_PYTHON_BINDINGS)
		add_subdirectory(pySPlisHSPlasH)
	endif ()
endif()
