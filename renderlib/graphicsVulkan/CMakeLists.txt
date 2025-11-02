# Vulkan graphics library
set(VULKAN_GRAPHICS_SOURCES
    IVulkanRenderWindow.cpp
    RenderVK.cpp
    RenderVKPT.cpp
    FramebufferVK.cpp
    ImageXyzcGpuVK.cpp
    GestureGraphicsVK.cpp
    vk/Util.cpp
    vk/Image3DVK.cpp
    vk/Image2DVK.cpp
    vk/FontVK.cpp
)

set(VULKAN_GRAPHICS_HEADERS
    IVulkanRenderWindow.h
    RenderVK.h
    RenderVKPT.h
    FramebufferVK.h
    ImageXyzcGpuVK.h
    GestureGraphicsVK.h
    vk/Util.h
    vk/Image3DVK.h
    vk/Image2DVK.h
    vk/FontVK.h
    vk/FSQ.h
)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Create Vulkan graphics library
add_library(vulkan_graphics STATIC ${VULKAN_GRAPHICS_SOURCES} ${VULKAN_GRAPHICS_HEADERS})

target_include_directories(vulkan_graphics PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/vk
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(vulkan_graphics 
    ${Vulkan_LIBRARIES}
    glm
)

# Compile shader sources to SPIR-V
find_program(GLSLC glslc HINTS ${Vulkan_GLSLC_EXECUTABLE})

if(GLSLC)
    # Add custom target to compile shaders
    file(GLOB SHADER_SOURCES "spirv/*.vert" "spirv/*.frag" "spirv/*.comp")
    
    foreach(SHADER ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER} NAME)
        set(SHADER_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/spirv/${SHADER_NAME}.spv")
        
        add_custom_command(
            OUTPUT ${SHADER_OUTPUT}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/spirv"
            COMMAND ${GLSLC} ${SHADER} -o ${SHADER_OUTPUT}
            DEPENDS ${SHADER}
            COMMENT "Compiling ${SHADER_NAME}"
        )
        
        list(APPEND SPIRV_SHADERS ${SHADER_OUTPUT})
    endforeach()
    
    add_custom_target(vulkan_shaders DEPENDS ${SPIRV_SHADERS})
    add_dependencies(vulkan_graphics vulkan_shaders)
else()
    message(WARNING "glslc not found - Vulkan shaders will not be compiled")
endif()

# Set preprocessor definition to indicate Vulkan support
target_compile_definitions(vulkan_graphics PUBLIC AGAVE_USE_VULKAN=1)