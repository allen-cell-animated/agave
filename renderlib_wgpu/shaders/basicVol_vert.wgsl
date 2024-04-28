
struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct CameraUniform {
    modelViewMatrix: mat4x4<f32>,
    projectionMatrix: mat4x4<f32>,
};
@group(1) @binding(0) // 1.
var<uniform> camera: CameraUniform;


struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) pObj: vec3<f32>,
};



@vertex
fn vs_main(
    v: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.pObj = v.position;
    out.clip_position = projectionMatrix * modelViewMatrix * vec4<f32>(v.position, 1.0);
    return out;
}



@group(0) @binding(0)
var textureAtlas: texture_3d<f32>;
@group(0) @binding(1)
var textureAtlasMask: texture_2d<f32>;
@group(0) @binding(2)
var s_texAtlas: sampler;
@group(0) @binding(3)
var s_texAtlasMask: sampler;

struct VolumeUniforms {
    inverseModelViewMatrix: mat4x4<f32>,
    iResolution: vec2<f32>,
    isPerspective: f32,
    orthoScale: f32,
    GAMMA_MIN: f32,
    GAMMA_MAX: f32,
    GAMMA_SCALE: f32,
    BRIGHTNESS: f32,
    DENSITY: f32,
    maskAlpha: f32,
    BREAK_STEPS: i32,
    AABB_CLIP_MIN: vec3<f32>,
    AABB_CLIP_MAX: vec3<f32>,
    dataRangeMin: f32, // 0..1 (mapped from 0..uint16_max)
    dataRangeMax: f32, // 0..1 (mapped from 0..uint16_max)
};
@group(1) @binding(0) // 1.
var<uniform> v: VolumeUniforms;

fn powf(a: f32, b: f32) -> f32 {
    return pow(a, b);
}

fn rand(co: vec2<f32>) -> f32 {
    var threadId = gl_FragCoord.x / (gl_FragCoord.y + 1.0);
    var bigVal = threadId * 1299721.0 / 911.0;
    var smallVal: vec2<f32> = vec2(threadId * 7927.0 / 577.0, threadId * 104743.0 / 1039.0);
    return fract(sin(dot(co, smallVal)) * bigVal);
}

fn luma2Alpha(color: vec4<f32>, vmin: f32, vmax: f32, C: f32) -> vec4<f32> {
    var x = max(color[2], max(color[0], color[1]));
    var xi = (x - vmin) / (vmax - vmin);
    xi = clamp(xi, 0.0, 1.0);
    var y = pow(xi, C);
    y = clamp(y, 0.0, 1.0);
    color[3] = y;
    return(color);
}

fn sampleAs3DTexture(tex: texture_3d<f32>, s: sampler, pos: vec4<f32>) -> vec4<f32> {
    f32bounds = f32(pos[0] > 0.001 && pos[0] < 0.999 && pos[1] > 0.001 && pos[1] < 0.999 && pos[2] > 0.001 && pos[2] < 0.999);

    vec4texval = textureLod(tex, pos.xyz, 0).rgba;
    vec4retval = vec4(texval.rgb, 1.0);

//    f32 texval = textureLod(tex, pos.xyz, 0).r;
//	texval = (texval - dataRangeMin) / (dataRangeMax - dataRangeMin);
//	vec4<f32> retval = vec4(texval, texval, texval, 1.0);
    return bounds * retval;
}

fn sampleStack(tex: texture_3d<f32>, s: sampler, pos: vec4<f32>) -> vec4<f32> {
    vec4col = sampleAs3DTexture(tex, pos);
    col = luma2Alpha(col, GAMMA_MIN, GAMMA_MAX, GAMMA_SCALE);
    return col;
}

//->intersect AXIS-ALIGNED box routine
//
fn intersectBox(r_o: vec3<f32>, r_d: vec3<f32>, boxMin: vec3<f32>, boxMax: vec3<f32>, tnear: ptr<function, f32>, tfar: ptr<function,f32>) -> bool {
    vec3invR = vec3(1.0, 1.0, 1.0) / r_d;
    vec3tbot = invR * (boxMin - r_o);
    vec3ttop = invR * (boxMax - r_o);
    vec3tmin = min(ttop, tbot);
    vec3tmax = max(ttop, tbot);
    f32largest_tmin = max(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
    f32smallest_tmax = min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));
    tnear = largest_tmin;
    tfar = smallest_tmax;
    return(smallest_tmax > largest_tmin);
}

fn integrateVolume(eye_o: vec4<f32>, eye_d: vec4<f32>, tnear: f32, tfar: f32, clipNear: f32, clipFar: f32, textureAtlas: texture_3d<f32>) -> vec4<f32> {
    vec4C = vec4(0.0);
    f32tend = min(tfar, clipFar);
    f32tbegin = tnear;
    const maxSteps: i32 = 512;
    f32csteps = clamp(f32(BREAK_STEPS), 1.0, f32(maxSteps));
    f32invstep = 1.0 / csteps;
    f32r = 0.5 - 1.0 * rand(eye_d.xy);
    f32tstep = invstep;
    f32tfarsurf = r * tstep;
    f32overflow = mod((tfarsurf - tend), tstep);
    f32t = tbegin + overflow;
    t += r * tstep;
    f32tdist = 0.0;
    int numSteps = 0;

    var pos: vec4<f32>; var col: vec4<f32>;
    f32s = 0.5 * f32(maxSteps) / csteps;
    for (int i = 0; i < maxSteps; i++) {
            pos = eye_o + eye_d * t;
            pos.xyz = (pos.xyz + 0.5);//0.5 * (pos + 1.0); // map position from [boxMin, boxMax] to [0, 1] coordinates
            col = sampleStack(textureAtlas, pos);

		//Finish up by adding brightness/density
            col.xyz *= BRIGHTNESS;
            col.w *= DENSITY;
            f32stepScale = (1.0 - powf((1.0 - col.w), s));
            col.w = stepScale;
            col.xyz *= col.w;
            col = clamp(col, 0.0, 1.0);

            C = (1.0 - C.w) * col + C;
            t += tstep;
            numSteps = i;
            if t > tend
                break;
                if C.w > 1.0
                    break;
	}
                return C;
}

            @fragment
            fn fs_main(in,: VertexOutput) -> @location(0) vec4<f32>{
                var outputColour: vec4<f32> = vec4(1.0, 0.0, 0.0, 1.0);
	// gl_FragCoord defaults to 0,0 at lower left
                vec2vUv = gl_FragCoord.xy / iResolution.xy;

                vec3eyeRay_o, eyeRay_d;
                if isPerspective != 0.0 {
		// camera position in camera space is 0,0,0!
                    eyeRay_o = (inverseModelViewMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
                    eyeRay_d = normalize(inData.pObj - eyeRay_o);
                } else {
                    f32zDist = 2.0;
                    eyeRay_d = (inverseModelViewMatrix * vec4(0.0, 0.0, -zDist, 0.0)).xyz;
                    vec4ray_o = vec4(2.0 * vUv - 1.0, 1.0, 1.0);
                    ray_o.xy *= orthoScale;
                    ray_o.x *= iResolution.x / iResolution.y;
                    eyeRay_o = (inverseModelViewMatrix * ray_o).xyz;
                }

                vec3boxMin = AABB_CLIP_MIN;
                vec3boxMax = AABB_CLIP_MAX;
                f32tnear, tfar;
                boolhit = intersectBox(eyeRay_o, eyeRay_d, boxMin, boxMax, tnear, tfar);
                if !hit {
                    outputColour = vec4(1.0, 0.0, 1.0, 0.0);
                    return outputColour;
                }
//else {
//		outputColour = vec4(1.0, 1.0, 1.0, 1.0);
//		return;
//}
                f32clipNear = 0.0;//-(dot(eyeRay_o.xyz, eyeNorm) + dNear) / dot(eyeRay_d.xyz, eyeNorm);
                f32clipFar = 10000.0;//-(dot(eyeRay_o.xyz,-eyeNorm) + dFar ) / dot(eyeRay_d.xyz,-eyeNorm);

                vec4C = integrateVolume(vec4(eyeRay_o, 1.0), vec4(eyeRay_d, 0.0),
                    tnear, tfar,
                    clipNear, clipFar,
                    textureAtlas);
                C = clamp(C, 0.0, 1.0);
                outputColour = C;
                return outputColour;
}
