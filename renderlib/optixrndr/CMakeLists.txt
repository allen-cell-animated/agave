
set(CMAKE_CUDA_FLAGS
	# Pascal target
	"-gencode=arch=compute_61,code=sm_61 -Wno-deprecated-gpu-targets"
)

add_library(CudaPTX OBJECT
	hello.cu
	parallelogram.cu
	pathTracer.cu
	phong.cu
	triangle_mesh.cu
)

set_property(TARGET CudaPTX PROPERTY CUDA_PTX_COMPILATION ON)

target_include_directories(CudaPTX PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}"
	${CUDA_SDK_ROOT_DIR}/common/inc
	${CUDA_TOOLKIT_INCLUDE}
	${OptiX_INCLUDE}
)

# drop ptx files into install target location
install(TARGETS CudaPTX
   OBJECTS DESTINATION ${INSTALLDIR}/ptx 
)

