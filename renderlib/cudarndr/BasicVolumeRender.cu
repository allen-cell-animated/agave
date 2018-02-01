#include "BasicVolumeRender.cuh"

#include "CudaUtilities.h"
#include "helper_math.cuh"
//#include "Camera.cuh"
//#include "Geometry.h"

CD int			gFilmWidth3;
CD int			gFilmHeight3;

#define KRNL_CH_BLOCK_W		16
#define KRNL_CH_BLOCK_H		8
#define KRNL_CH_BLOCK_SIZE	KRNL_CH_BLOCK_W * KRNL_CH_BLOCK_H

typedef struct
{
	float4 m[3];
} float3x4;
typedef struct
{
	float4 m[4];
} float4x4;

CD float4x4 c_invViewMatrix;  // inverse view matrix

struct Ray
{
	float3 o;   // origin
	float3 d;   // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

// transform vector by matrix (no translation)
DEV
float3 mul(const float3x4 &M, const float3 &v)
{
	float3 r;
	r.x = dot(v, make_float3(M.m[0]));
	r.y = dot(v, make_float3(M.m[1]));
	r.z = dot(v, make_float3(M.m[2]));
	return r;
}

// transform vector by matrix with translation
DEV
float4 mul(const float3x4 &M, const float4 &v)
{
	float4 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	r.w = 1.0f;
	return r;
}

// transform vector by matrix (no translation)
DEV
float3 mul(const float4x4 &M, const float3 &v)
{
	float3 r;
	r.x = dot(v, make_float3(M.m[0]));
	r.y = dot(v, make_float3(M.m[1]));
	r.z = dot(v, make_float3(M.m[2]));
	return r;
}

// transform vector by matrix with translation
DEV
float4 mul(const float4x4 &M, const float4 &v)
{
	float4 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	r.w = dot(v, M.m[3]);
	return r;
}

DEV
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
	// compute intersection of ray with all six bbox planes
	float3 invR = make_float3(1.0f) / r.d;
	float3 tbot = invR * (boxmin - r.o);
	float3 ttop = invR * (boxmax - r.o);

	// re-order intersections to find smallest and largest on each axis
	float3 tmin = fminf(ttop, tbot);
	float3 tmax = fmaxf(ttop, tbot);

	// find the largest tmin and the smallest tmax
	float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
	float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}


KERNEL void KrnlCh(cudaTextureObject_t volumeTex, float* outbuf, float density, float brightness, float dataRangeMin, float dataRangeMax)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	//const int TID	= threadIdx.y * blockDim.x + threadIdx.x;

	// bounds check.
	if (X >= gFilmWidth3 || Y >= gFilmHeight3)
		return;

	// pixel offset of this thread
	int pixoffset = Y*(gFilmWidth3) + (X);
	int floatoffset = pixoffset*4;

	// background color
	outbuf[floatoffset] = 0.75;
	outbuf[floatoffset+1] = 0.0;
	outbuf[floatoffset+2] = 0.25;
	outbuf[floatoffset+3] = 1.0;

	const int maxSteps = 500;
	const float opacityThreshold = 0.95f;

	const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
	const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

	float u = (X / (float)gFilmWidth3)*2.0f - 1.0f;
	float v = (Y / (float)gFilmHeight3)*2.0f - 1.0f;

	// calculate eye ray in OBJECT(BOX) space
	Ray eyeRay;
	eyeRay.o = make_float3(0.0f, 0.0f, 5.0f);
	//eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
	eyeRay.d = normalize(make_float3(u, v, -2.0f));
	//eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

	if (!hit) {
		return;
	}

	if (tnear < 0.0f) {
		tnear = 0.0f;     // clamp to near plane
	}

	// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;

	//float tstep = 0.01;
	//float tstep = (tfar - tnear) / maxSteps;
	// diagonal of cube of side length 2
	float tstep = (2.0*sqrt(3.0)) / maxSteps;

	float3 step = eyeRay.d*tstep;

	for (int i = 0; i<maxSteps; i++)
	{
		// read from 3D texture
		// remap position to [0, 1] coordinates
		float sample = tex3DLod<float>(volumeTex, pos.x*0.5f + 0.5f, pos.y*0.5f + 0.5f, pos.z*0.5f + 0.5f, 0.0f);

		sample = (sample - dataRangeMin) / (dataRangeMax - dataRangeMin);
		//sample *= 64.0f;    // scale for 10-bit data

		// lookup in transfer function texture
		float4 col = make_float4(sample, sample, sample, sample);// tex1D(transferTex, (sample - transferOffset)*transferScale);
		col.w *= density;

		// "under" operator for back-to-front blending
		//sum = lerp(sum, col, col.w);

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;
		// "over" operator for front-to-back blending
		sum = sum + col*(1.0f - sum.w);

		// exit early if opaque
		if (sum.w > opacityThreshold) {
			break;
		}

		t += tstep;

		if (t > tfar) {
			break;
		}

		pos += step;
	}

	sum *= brightness;

	// write output color
	outbuf[floatoffset] = sum.x;
	outbuf[floatoffset + 1] = sum.y;
	outbuf[floatoffset + 2] = sum.z;
	outbuf[floatoffset + 3] = sum.w;

}

void RayMarchVolume(float* outbuf, cudaTextureObject_t volumeTex, cudaTextureObject_t gradientVolumeTex, int w, int h, float density, float brightness, float* invViewMatrix, float texmin, float texmax)
{
	// init some input vars for kernel
	HandleCudaError(cudaMemcpyToSymbol(gFilmWidth3, &w, sizeof(int)));
	HandleCudaError(cudaMemcpyToSymbol(gFilmHeight3, &h, sizeof(int)));
	HandleCudaError(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeof(float)*12));

	// launch kernel
	const dim3 KernelBlock(KRNL_CH_BLOCK_W, KRNL_CH_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)w / (float)KernelBlock.x), (int)ceilf((float)h / (float)KernelBlock.y));
	KrnlCh<<<KernelGrid, KernelBlock>>>(volumeTex, outbuf, density, brightness, texmin, texmax);
	
	// wait for end kernel
	cudaDeviceSynchronize();
	HandleCudaKernelError(cudaGetLastError(), "RayMarchVolume");
}

////////////////
////////////////
////////////////

CD float gInvExposure1;

#define KRNL_TM_BLOCK_W		8
#define KRNL_TM_BLOCK_H		8
#define KRNL_TM_BLOCK_SIZE	KRNL_TM_BLOCK_W * KRNL_TM_BLOCK_H

KERNEL void KrnlToneMap_Basic(float* inbuf, cudaSurfaceObject_t surfaceObj)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= gFilmWidth3 || Y >= gFilmHeight3)
		return;

	int pixoffset = Y*(gFilmWidth3) + (X);

	float4 sample = reinterpret_cast<float4*>(inbuf)[pixoffset];

	sample.x = __saturatef(1.0f - expf(-(sample.x * gInvExposure1)));
	sample.y = __saturatef(1.0f - expf(-(sample.y * gInvExposure1)));
	sample.z = __saturatef(1.0f - expf(-(sample.z * gInvExposure1)));
	sample.w = __saturatef(sample.w);

	uchar4 pixel = make_uchar4(sample.x*255.0, sample.y*255.0, sample.z*255.0, sample.w*255.0);
	surf2Dwrite(pixel, surfaceObj, X*4, Y);
}

void ToneMap_Basic(float* inbuf, cudaSurfaceObject_t surfaceObj, int w, int h)
{
	// init some input vars for kernel
	//float invexposure = 1.0 / 1.0;
	float invexposure = 1.0 / 4.0;
	HandleCudaError(cudaMemcpyToSymbol(gInvExposure1, &invexposure, sizeof(float)));

	const dim3 KernelBlock(KRNL_TM_BLOCK_W, KRNL_TM_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)w / (float)KernelBlock.x), (int)ceilf((float)h / (float)KernelBlock.y));
	KrnlToneMap_Basic<<<KernelGrid, KernelBlock>>>(inbuf, surfaceObj);

	cudaDeviceSynchronize();
	HandleCudaKernelError(cudaGetLastError(), "Tone Map");
}
