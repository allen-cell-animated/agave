// Vertex shader
struct CameraUniform {
    modelViewMatrix: mat4x4<f32>,
    projectionMatrix: mat4x4<f32>,
};
@group(1) @binding(0) // 1.
var<uniform> camera: CameraUniform;


struct VertexInput {
    @location(0) position: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) pObj: vec2<f32>,
};

@vertex
fn vs_main(
    v: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.pObj = v.position;
    out.clip_position = vec4<f32>(v.position, 0.0, 1.0);
    return out;
}


@group(0) @binding(0)
var textureRender: texture_2d<f32>;
@group(0) @binding(1)
var textureAccum: texture_2d<f32>;
@group(0) @binding(2)
var s_tex: sampler;
@group(0) @binding(3)
var<uniform> numIterations: i32;

fn CumulativeMovingAverage(A: vec4<f32>, Ax: vec4<f32>, N: i32) -> vec4<f32> {
    return A + ((Ax - A) / max(f32(N), 1.0f));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var accum: vec4<f32> = textureSampleLevel(textureAccum, s_tex, (in.pObj + 1.0) / 2.0, 0.0).rgba;
    var render: vec4<f32> = textureSampleLevel(textureRender, s_tex, (in.pObj + 1.0) / 2.0, 0.0).rgba;

    return CumulativeMovingAverage(accum, render, numIterations);
}
