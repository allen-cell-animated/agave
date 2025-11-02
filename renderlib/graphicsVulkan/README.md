# Vulkan Graphics Implementation

This directory contains a complete Vulkan-based reimplementation of the OpenGL graphics rendering system in `renderlib/graphics/`. It provides an alternative rendering backend that can be used instead of OpenGL for modern graphics applications.

## Directory Structure

```
renderlib/
├── graphics/           # Original OpenGL implementation
│   ├── gl/            # OpenGL-specific utilities
│   ├── glsl/          # GLSL shaders
│   ├── IRenderWindow.h # OpenGL render window interface
│   ├── RenderGL.cpp   # OpenGL renderer
│   └── ...
├── graphicsVulkan/    # New Vulkan implementation (mirrors graphics/)
│   ├── vk/            # Vulkan-specific utilities
│   ├── spirv/         # SPIR-V shaders
│   ├── IVulkanRenderWindow.h # Vulkan render window interface
│   ├── RenderVK.cpp   # Vulkan renderer
│   └── ...
└── renderlib.cpp      # Main library (conditionally compiled)
```

## Implementation Strategy

### 1. Parallel Implementation

- **`graphics/`** contains all OpenGL code (existing)
- **`graphicsVulkan/`** contains equivalent Vulkan implementations
- Same class names and interfaces, different implementations
- No abstraction layer complexity

### 2. Compile-Time Selection

```cmake
option(AGAVE_USE_VULKAN "Use Vulkan graphics backend" OFF)

if(AGAVE_USE_VULKAN)
    target_compile_definitions(renderlib PUBLIC AGAVE_USE_VULKAN=1)
    target_sources(renderlib PRIVATE graphicsVulkan/RenderVK.cpp)
    target_link_libraries(renderlib vulkan_graphics)
else()
    target_sources(renderlib PRIVATE graphics/RenderGL.cpp)
    target_link_libraries(renderlib gl_graphics)
endif()
```

### 3. Conditional Headers

```cpp
#ifdef AGAVE_USE_VULKAN
#include "graphicsVulkan/RenderVK.h"
using RendererImpl = RenderVK;
#else
#include "graphics/RenderGL.h"
using RendererImpl = RenderGL;
#endif
```

## Benefits of This Approach

### ✅ Advantages

1. **No Abstraction Overhead** - Direct API usage, maximum performance
2. **Independent Development** - Work on Vulkan without affecting OpenGL
3. **Simple Migration** - Copy and modify existing classes
4. **Easy Maintenance** - Separate codebases, easier debugging
5. **Compile-Time Safety** - No runtime API switching complexity
6. **Resource Efficiency** - Only one graphics API linked per build

### 🔄 Migration Process

1. **Phase 1**: Copy `graphics/` classes to `graphicsVulkan/`
2. **Phase 2**: Replace OpenGL calls with Vulkan equivalents
3. **Phase 3**: Implement Vulkan-specific features (compute, ray tracing)
4. **Phase 4**: Optimize for Vulkan strengths (multi-threading, explicit control)

### 📋 Class Mappings

| OpenGL Class          | Vulkan Equivalent         | Purpose                 |
| --------------------- | ------------------------- | ----------------------- |
| `IRenderWindow`       | `IVulkanRenderWindow`     | Render window interface |
| `RenderGL`            | `RenderVK`                | Main renderer           |
| `RenderGLPT`          | `RenderVKPT`              | Path tracing renderer   |
| `GLFramebufferObject` | `VulkanFramebufferObject` | Framebuffer             |
| `Image3D`             | `Image3DVK`               | Volume renderer         |
| `RectImage2D`         | `RectImage2DVK`           | 2D image renderer       |
| `BoundingBoxDrawable` | `BoundingBoxDrawableVK`   | Debug rendering         |

## Example Usage

### CMake Configuration

```bash
# Build with OpenGL (default)
cmake .. -DAGAVE_USE_VULKAN=OFF

# Build with Vulkan
cmake .. -DAGAVE_USE_VULKAN=ON
```

### Code Example

```cpp
// Factory method automatically returns correct renderer
auto renderer = renderlib::createRenderer(renderlib::RendererType_Raymarch);

// Runtime query
if (renderlib::isVulkanEnabled()) {
    // Vulkan-specific features
} else {
    // OpenGL fallback
}
```

## Implementation Status

### ✅ Completed

- Directory structure created
- Basic Vulkan renderer stub (`RenderVK`)
- Vulkan utilities header (`vk/Util.h`)
- SPIR-V shader examples
- CMake build configuration
- Conditional compilation setup

### 🚧 In Progress

- Complete Vulkan renderer implementation
- Vulkan resource management classes
- SPIR-V shader compilation

### ❌ TODO

- Full Vulkan pipeline implementation
- Memory management utilities
- Descriptor set management
- Multi-threading support
- Vulkan validation layers integration
- Performance benchmarking

## Shader Management

### OpenGL (GLSL)

```glsl
#version 410 core
layout(location = 0) in vec2 aPos;
out vec2 texCoord;
void main() { /* ... */ }
```

### Vulkan (SPIR-V)

```glsl
#version 450
layout(location = 0) in vec2 inPosition;
layout(location = 0) out vec2 fragTexCoord;
void main() { /* ... */ }
```

Vulkan shaders are compiled to SPIR-V bytecode at build time using `glslc`.

This approach provides a clean, maintainable way to support both graphics APIs without the complexity of runtime abstraction layers.
