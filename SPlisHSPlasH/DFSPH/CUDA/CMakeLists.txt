################################################################################
# Device-resident DFSPH backend: CCCL kernels + CUDASTF orchestration.
#
# Built as an isolated CUDA / C++17 static library. It exposes only a CUDA-free
# C++ interface (DFSPHCudaBackend.h, FeatureProbes.h) so the main SPlisHSPlasH
# library (compiled at the project standard) links against it without pulling in
# any CUDA/CCCL headers.
################################################################################

set(DFSPH_CUDA_SOURCES
	FeatureProbes.cu
	Bender2019MapCUDA.cu
	DFSPHKernels.cu
	DFSPHCudaBackend.cu
)

set(DFSPH_CUDA_HEADERS
	DFSPHCudaBackend.h
	FeatureProbes.h
	DFSPHDeviceState.cuh
	DFSPHKernels.cuh
	Bender2019MapCUDA.cuh
)

add_library(SPlisHSPlasH_CUDA STATIC ${DFSPH_CUDA_SOURCES} ${DFSPH_CUDA_HEADERS})

set_target_properties(SPlisHSPlasH_CUDA PROPERTIES
	CUDA_STANDARD 17
	CUDA_STANDARD_REQUIRED ON
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
	CUDA_ARCHITECTURES "${CUDA_DFSPH_ARCHITECTURES}"
	POSITION_INDEPENDENT_CODE ON
)

# CCCL (libcudacxx / cub / thrust / cudax-STF) from the local checkout.
target_include_directories(SPlisHSPlasH_CUDA PRIVATE
	${CCCL_DIR}/libcudacxx/include
	${CCCL_DIR}/cub
	${CCCL_DIR}/thrust
	${CCCL_DIR}/cudax/include
)

# OpenGL headers for the CUDA/GL interop render path (cuda_gl_interop.h needs
# GL/gl.h; the symbols themselves live in cudart / the exe's GL link).
find_package(OpenGL QUIET)
if (OPENGL_INCLUDE_DIR)
	target_include_directories(SPlisHSPlasH_CUDA PRIVATE ${OPENGL_INCLUDE_DIR})
endif()

target_compile_options(SPlisHSPlasH_CUDA PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
	$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

if (USE_DOUBLE_PRECISION)
	target_compile_definitions(SPlisHSPlasH_CUDA PUBLIC USE_DOUBLE)
endif()

# CUDASTF uses the CUDA driver API.
target_link_libraries(SPlisHSPlasH_CUDA PRIVATE CUDA::cudart CUDA::cuda_driver)

# Standalone runner for the feature probes.
add_executable(DFSPHCudaProbe probe_main.cpp)
target_link_libraries(DFSPHCudaProbe PRIVATE SPlisHSPlasH_CUDA CUDA::cudart CUDA::cuda_driver)
set_target_properties(DFSPHCudaProbe PROPERTIES CUDA_ARCHITECTURES "${CUDA_DFSPH_ARCHITECTURES}")
