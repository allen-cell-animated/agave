#version 450

layout(location = 0) in vec3 inPosition;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 volumeScale;
    vec4 rayParams;
} ubo;

layout(location = 0) out vec3 rayDir;
layout(location = 1) out vec3 rayStart;

void main() {
    vec4 worldPos = ubo.model * vec4(inPosition, 1.0);
    gl_Position = ubo.proj * ubo.view * worldPos;
    
    // Calculate ray direction and starting position for volume rendering
    rayStart = inPosition * 0.5 + 0.5; // Transform to [0,1] cube
    rayDir = normalize(inPosition);
}