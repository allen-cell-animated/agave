struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) vUv: vec2<f32>,
};

@vertex
fn vs_main(
    v: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.vUv = v.uv;
    out.clip_position = vec4<f32>(v.position, 1.0);
    return out;
}


@group(0) @binding(0)
var tTexture0: texture_2d<f32>;
@group(0) @binding(1)
var s: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(tTexture0, s, in.vUv);
}
