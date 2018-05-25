
add_library(CudaPTX OBJECT
	hello.cu
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
   OBJECTS DESTINATION bin/ptx 
)

