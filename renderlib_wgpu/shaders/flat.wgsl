// Vertex shader
struct Uniform {
    mvp: mat4x4<f32>,
    color: vec4<f32>,
};
@group(1) @binding(0) // 1.
var<uniform> u: Uniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) f_color: vec4<f32>,
};

@vertex
fn vs_main(
    v: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4(v.position, 1.0);
    out.f_colour = u.colour;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.f_color;
}
