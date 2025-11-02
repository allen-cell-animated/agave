#version 450

// Display fragment shader for final screen output

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D colorTexture;

// Exposure and gamma correction uniforms
layout(set = 0, binding = 1) uniform DisplayUniforms {
    float exposure;
    float gamma;
    float brightness;
    float contrast;
} display;

vec3 tonemap(vec3 color) {
    // Reinhard tone mapping
    color *= display.exposure;
    color = color / (color + vec3(1.0));
    
    // Gamma correction
    color = pow(color, vec3(1.0 / display.gamma));
    
    // Brightness and contrast adjustment
    color = (color - 0.5) * display.contrast + 0.5 + display.brightness;
    
    return clamp(color, 0.0, 1.0);
}

void main() {
    vec4 color = texture(colorTexture, fragTexCoord);
    
    // Apply tone mapping and color correction
    vec3 finalColor = tonemap(color.rgb);
    
    outColor = vec4(finalColor, color.a);
}