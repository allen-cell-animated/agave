target_include_directories(renderlib PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}"
	${Boost_INCLUDE_DIRS}
)
find_package(CUDA)
add_library(renderlib_cuda STATIC
	"${CMAKE_CURRENT_SOURCE_DIR}/Camera2.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/Camera2Impl.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/BasicVolumeRender.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/BasicVolumeRender.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/Checkerboard.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/Checkerboard.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/Checkerboard2.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/Checkerboard2.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/Core.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/Core.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/Denoise.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/DenoiseParams.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/Estimate.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/Lighting2.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/Lighting2Impl.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/helper_math.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/MonteCarlo.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/NearestIntersection.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/RayMarching.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/RNG.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/Sample.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/Shader.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/SingleScattering.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/ToneMap.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/Transport.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/Utilities.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/CudaUtilities.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/CudaUtilities.h"
)
target_include_directories(renderlib_cuda PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}"
	"${CMAKE_CURRENT_SOURCE_DIR}/.."  # a couple dependencies on utility classes down in renderlib
	${Boost_INCLUDE_DIRS}
	${GLAD_DIR}
	${CUDA_TOOLKIT_INCLUDE}
	${CUDA_INCLUDE_DIRS}
	${GLM_INCLUDE_DIRS}
)

target_compile_definitions(renderlib_cuda PUBLIC
	-DBOOST_ALL_DYN_LINK
)

target_sources(renderlib PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/BoundingBox.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/BoundingBox.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Defines.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Enumerations.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Geometry.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Geometry.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Spectrum.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Spectrum.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/Variance.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Variance.h"
)
