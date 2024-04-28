struct VertexInput {
    @location(0) vPos: vec3<f32>,
    @location(1) vUV: vec2<f32>,
    @location(2) vCol: vec4<f32>,
    @location(3) vCode: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) Frag_color: vec4<f32>,
    @location(1) Frag_UV: vec2<f32>,
};

struct Uniform {
    projection: mat4x4<f32>,
    picking: i32, //< draw for display or for picking? Picking has no texture.
};
@group(1) @binding(0) // 1.
var<uniform> u: Uniform;

@vertex
fn vs_main(
    v: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.Frag_UV = v.vUV;
    if u.picking == 1 {
        out.Frag_color = vec4(f32(v.vCode & 0xffu) / 255.0,
            f32((v.vCode >> 8) & 0xffu) / 255.0,
            f32((v.vCode >> 16) & 0xffu) / 255.0,
            1.0);
    } else {
        out.Frag_color = v.vCol;
    }
    out.clip_position = u.projection * vec4<f32>(v.vPos, 1.0);
    return out;
}

@group(0) @binding(0)
var Texture: texture_2d<f32>;
@group(0) @binding(1)
var s: sampler;

const EPSILON:f32 = 0.1;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var result = in.Frag_color;

        // When drawing selection codes, everything is opaque.
    if u.picking == 1 {
        result.w = 1.0;
    }

        // Gesture geometry handshake: any uv value below -64 means
        // no texture lookup. Check VertsCode::k_noTexture
        // (add an epsilon to fix some fp errors.
        // TODO check to see if highp would have helped)
    if u.picking == 0 && in.Frag_UV.x > -64 + EPSILON {
        result *= textureSample(Texture, s, in.Frag_UV.xy);
    }

        // Gesture geometry handshake: any uv equal to -128 means
        // overlay a checkerboard pattern. Check VertsCode::k_marqueePattern
    if in.Frag_UV.x == -128.0 {
            // Create a pixel checkerboard pattern used for marquee
            // selection
        var x = i32(in.clip_position.x);
        var y = i32(in.clip_position.y);
        if ((x + y) & 1) == 0 {result = vec4<f32>(0, 0, 0, 1);}
    }
    return result;
}
