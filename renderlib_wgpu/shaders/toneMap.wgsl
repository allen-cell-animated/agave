struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) vUv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 1.0);
    out.vUv = in.uv;
    return out;
}


@group(1) @binding(0) var<uniform> gInvExposure:f32;

@group(0) @binding(0)
var tTexture0: texture_2d<f32>;
@group(0) @binding(1)
var s: sampler;


fn XYZtoRGB(xyz: vec3<f32>) -> vec3<f32> {
    return vec3(
        3.240479f * xyz[0] - 1.537150f * xyz[1] - 0.498535f * xyz[2],
        -0.969256f * xyz[0] + 1.875991f * xyz[1] + 0.041556f * xyz[2],
        0.055648f * xyz[0] - 0.204043f * xyz[1] + 1.057311f * xyz[2]
    );
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  // assumes texture stores colors in XYZ color space
    var pixelColor: vec4<f32> = textureSample(tTexture0, s, in.vUv);
    pixelColor = vec4<f32>(XYZtoRGB(pixelColor.xyz), pixelColor.w);
    pixelColor = vec4<f32>(vec3(1.0) - exp(-pixelColor.xyz * gInvExposure), pixelColor.w);
    pixelColor = clamp(pixelColor, vec4(0.0), vec4(1.0));
    return pixelColor;
}

/*
    /////////////////////
    /////////////////////
    /////////////////////
    /////////////////////
    //// DENOISING FILTER

    vec4 clr00 = pixelColor;
    vec4 rgbsample = clr00;
    // convert XYZ to RGB here.
    rgbsample.rgb = XYZtoRGB(clr00.rgb);
    // tone map!
    rgbsample.rgb = 1.0 - exp(-rgbsample.rgb * gInvExposure);
    rgbsample = clamp(rgbsample, 0.0, 1.0);
    clr00 = rgbsample;

    float fCount = 0.0;
    float SumWeights = 0.0;
    vec3 clr = vec3(0.0, 0.0, 0.0);

    for (int i = -gDenoiseWindowRadius; i <= gDenoiseWindowRadius; i++) {
      for (int j = -gDenoiseWindowRadius; j <= gDenoiseWindowRadius; j++) {
        // sad face...
        if (Y + i < 0)
          continue;
        if (X + j < 0)
          continue;
        if (Y + i >= gFilmHeight)
          continue;
        if (X + j >= gFilmWidth)
          continue;

        vec4 clrIJ = texture(tTexture0, vUv + vec2(i,j));
        rgbsample.rgb = XYZToRGB(clrIJ.rgb);
        // tone map!
        rgbsample.rgb = 1.0 - exp(-rgbsample.rgb * gInvExposure);
        rgbsample = clamp(rgbsample, 0.0, 1.0);

        clrIJ = rgbsample;

        float distanceIJ = vecLen(clr00, clrIJ);

        // gDenoiseNoise = 1/h^2
        //
        float weightIJ = expf(-(distanceIJ * gDenoiseNoise + (float)(i * i + j * j) * gDenoiseInvWindowArea));

        clr.x += clrIJ.x * weightIJ;
        clr.y += clrIJ.y * weightIJ;
        clr.z += clrIJ.z * weightIJ;

        SumWeights += weightIJ;

        fCount += (weightIJ > gDenoiseWeightThreshold) ? gDenoiseInvWindowArea : 0;
      }
    }

    SumWeights = 1.0f / SumWeights;

    clr.rgb *= SumWeights;

    float LerpQ = (fCount > gDenoiseLerpThreshold) ? gDenoiseLerpC : 1.0f - gDenoiseLerpC;

    clr.rgb = mix(clr.rgb, clr00.rgb, LerpQ);
    clr.rgb = clamp(clr.rgb, 0.0, 1.0);

    out_FragColor = vec4(clr.rgb, clr00.a);
*/
