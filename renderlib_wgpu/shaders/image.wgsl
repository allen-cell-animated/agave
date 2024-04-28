
struct Uniform {
    mvp: mat4x4<f32>,
};
@group(1) @binding(0) // 1.
var<uniform> u: Uniform;

struct VertexInput {
    @location(0) coord2d: vec2<f32>,
    @location(1) texcoord: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) f_texcoord: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4<f32>(in.coord2d, 0.0, 1.0);
    out.f_texcoord = in.texcoord;
    return out;
}

@group(0) @binding(0)
var tex: texture_2d<f32>;
@group(0) @binding(1)
var s: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(tex, s, in.f_texcoord);
}
