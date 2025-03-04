
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
    out.clip_position = camera.projectionMatrix * camera.modelViewMatrix * vec4<f32>(v.position, 1.0);
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

fn rand(co: vec2<f32>, fragCoord: vec2<f32>) -> f32 {
    var threadId = fragCoord.x / (fragCoord.y + 1.0);
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
    return vec4<f32>(color[0], color[1], color[2], y);
}

fn sampleAs3DTexture(tex: texture_3d<f32>, s: sampler, pos: vec4<f32>) -> vec4<f32> {
    var bounds = f32(pos[0] > 0.001 && pos[0] < 0.999 && pos[1] > 0.001 && pos[1] < 0.999 && pos[2] > 0.001 && pos[2] < 0.999);

    var texval = textureSampleLevel(tex, s, pos.xyz, 0.0).rgba;
    var retval = vec4(texval.rgb, 1.0);

//    f32 texval = textureLod(tex, pos.xyz, 0).r;
//	texval = (texval - dataRangeMin) / (dataRangeMax - dataRangeMin);
//	vec4<f32> retval = vec4(texval, texval, texval, 1.0);
    return bounds * retval;
}

fn sampleStack(tex: texture_3d<f32>, s: sampler, pos: vec4<f32>) -> vec4<f32> {
    var col = sampleAs3DTexture(tex, s, pos);
    col = luma2Alpha(col, v.GAMMA_MIN, v.GAMMA_MAX, v.GAMMA_SCALE);
    return col;
}

//->intersect AXIS-ALIGNED box routine
//
fn intersectBox(r_o: vec3<f32>, r_d: vec3<f32>, boxMin: vec3<f32>, boxMax: vec3<f32>, tnear: ptr<function, f32>, tfar: ptr<function,f32>) -> bool {
    var invR = vec3(1.0, 1.0, 1.0) / r_d;
    var tbot = invR * (boxMin - r_o);
    var ttop = invR * (boxMax - r_o);
    var tmin = min(ttop, tbot);
    var tmax = max(ttop, tbot);
    var largest_tmin = max(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
    var smallest_tmax = min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));
    *tnear = largest_tmin;
    *tfar = smallest_tmax;
    return(smallest_tmax > largest_tmin);
}

fn mymod(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

    const maxSteps: i32 = 512;
fn integrateVolume(fragCoord: vec2<f32>, eye_o: vec4<f32>, eye_d: vec4<f32>, tnear: f32, tfar: f32, clipNear: f32, clipFar: f32, textureAtlas: texture_3d<f32>) -> vec4<f32> {
    var C = vec4(0.0);
    var tend = min(tfar, clipFar);
    var tbegin = tnear;
    var csteps = clamp(f32(v.BREAK_STEPS), 1.0, f32(maxSteps));
    var invstep = 1.0 / csteps;
    var r = 0.5 - 1.0 * rand(eye_d.xy, fragCoord);
    var tstep = invstep;
    var tfarsurf = r * tstep;
    var overflow = mymod((tfarsurf - tend), tstep);
    var t = tbegin + overflow;
    t += r * tstep;
    var tdist = 0.0;
    var numSteps: i32 = 0;

    var pos: vec4<f32>; var col: vec4<f32>;
    var s = 0.5 * f32(maxSteps) / csteps;
    for (var i: i32 = 0; i < maxSteps; i++) {
        pos = eye_o + eye_d * t;
        pos = vec4<f32>(pos.xyz + 0.5, pos.w);//0.5 * (pos + 1.0); // map position from [boxMin, boxMax] to [0, 1] coordinates
        col = sampleStack(textureAtlas, s_texAtlas, pos);

		//Finish up by adding brightness/density
        col = vec4<f32>(col.xyz * v.BRIGHTNESS, col.w * v.DENSITY);
        var stepScale = (1.0 - powf((1.0 - col.w), s));
        col = vec4<f32>(col.xyz * stepScale, stepScale);
        col = clamp(col, vec4(0.0), vec4(1.0));

        C = (1.0 - C.w) * col + C;
        t += tstep;
        numSteps = i;
        if t > tend {  break;}
        if C.w > 1.0 {break;}
    }
    return C;
}

    @fragment
fn fs_main(
    //@builtin(position) position: vec4<f32>,
    inData: VertexOutput
) -> @location(0) vec4<f32> {
    var outputColour: vec4<f32> = vec4(1.0, 0.0, 0.0, 1.0);
	// gl_FragCoord defaults to 0,0 at lower left
    var vUv: vec2<f32> = inData.clip_position.xy / v.iResolution.xy;

    var eyeRay_o: vec3<f32>; var eyeRay_d: vec3<f32>;
    if v.isPerspective != 0.0 {
		// camera position in camera space is 0,0,0!
        eyeRay_o = (v.inverseModelViewMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
        eyeRay_d = normalize(inData.pObj - eyeRay_o);
    } else {
        var zDist = 2.0;
        eyeRay_d = (v.inverseModelViewMatrix * vec4(0.0, 0.0, -zDist, 0.0)).xyz;
        var ray_o: vec4<f32> = vec4(2.0 * vUv - 1.0, 1.0, 1.0);
        ray_o = ray_o * vec4(v.orthoScale * v.iResolution.x / v.iResolution.y, v.orthoScale, 1.0, 1.0);
//        ray_o.xy *= v.orthoScale;
//        ray_o.x *= v.iResolution.x / v.iResolution.y;
        eyeRay_o = (v.inverseModelViewMatrix * ray_o).xyz;
    }

    var boxMin = v.AABB_CLIP_MIN;
    var boxMax = v.AABB_CLIP_MAX;
    var tnear: f32; var tfar: f32;
    var hit = intersectBox(eyeRay_o, eyeRay_d, boxMin, boxMax, &tnear, &tfar);
    if !hit {
        outputColour = vec4(1.0, 0.0, 1.0, 0.0);
        return outputColour;
    }
//else {
//		outputColour = vec4(1.0, 1.0, 1.0, 1.0);
//		return;
//}
    var clipNear = 0.0;//-(dot(eyeRay_o.xyz, eyeNorm) + dNear) / dot(eyeRay_d.xyz, eyeNorm);
    var clipFar = 10000.0;//-(dot(eyeRay_o.xyz,-eyeNorm) + dFar ) / dot(eyeRay_d.xyz,-eyeNorm);

    var C = integrateVolume(inData.clip_position.xy, vec4(eyeRay_o, 1.0), vec4(eyeRay_d, 0.0),
        tnear, tfar,
        clipNear, clipFar,
        textureAtlas);
    C = clamp(C, vec4(0.0), vec4(1.0));
    outputColour = C;
    return outputColour;
}
