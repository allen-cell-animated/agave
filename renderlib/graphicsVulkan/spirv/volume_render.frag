#version 450

layout(location = 0) in vec3 rayDir;
layout(location = 1) in vec3 rayStart;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 volumeScale;
    vec4 rayParams; // x: step size, y: max steps, z: density scale, w: unused
} ubo;

layout(binding = 1) uniform sampler3D volumeTexture;
layout(binding = 2) uniform sampler1D transferFunction;

vec4 sampleVolume(vec3 pos) {
    if (any(lessThan(pos, vec3(0.0))) || any(greaterThan(pos, vec3(1.0)))) {
        return vec4(0.0);
    }
    
    float density = texture(volumeTexture, pos).r;
    return texture(transferFunction, density);
}

void main() {
    vec3 rayDirection = normalize(rayDir);
    vec3 currentPos = rayStart;
    
    float stepSize = ubo.rayParams.x;
    int maxSteps = int(ubo.rayParams.y);
    float densityScale = ubo.rayParams.z;
    
    vec4 color = vec4(0.0);
    float alpha = 0.0;
    
    // Ray marching through volume
    for (int i = 0; i < maxSteps && alpha < 0.99; i++) {
        vec4 sampl = sampleVolume(currentPos);
        sampl.a *= densityScale;
        
        // Front-to-back alpha blending
        color.rgb += (1.0 - alpha) * sampl.a * sampl.rgb;
        alpha += (1.0 - alpha) * sampl.a;

        currentPos += rayDirection * stepSize;
    }
    
    outColor = vec4(color.rgb, alpha);
}