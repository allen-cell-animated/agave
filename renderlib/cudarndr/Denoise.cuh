#include "Utilities.cuh"
#include "CudaUtilities.h"

#define KRNL_DENOISE_BLOCK_W		8
#define KRNL_DENOISE_BLOCK_H		8
#define KRNL_DENOISE_BLOCK_SIZE	KRNL_DENOISE_BLOCK_W * KRNL_DENOISE_BLOCK_H

DEV float lerpf(float a, float b, float c){
	return a + (b - a) * c;
}

DEV float vecLen(float4 a, float4 b)
{
    return ((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y) + (b.z - a.z) * (b.z - a.z));
}

DEV inline void XYZToRGB(const float4& xyz, float4& rgb) {
	XYZToRGB(&xyz.x, &rgb.x);
	rgb.w = xyz.w;
}

// inbuf is runningestimatergba
KERNEL void KrnlDenoise(float* inbuf, cudaSurfaceObject_t surfaceObj)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	int pixoffset = Y*(gFilmWidth)+(X);
	float4 clr00 = reinterpret_cast<float4*>(inbuf)[pixoffset];
	float4 rgbsample;
	// convert XYZ to RGB here.
	XYZToRGB(clr00, rgbsample);
	// tone map!
	rgbsample.x = __saturatef(1.0f - expf(-(rgbsample.x * gInvExposure)));
	rgbsample.y = __saturatef(1.0f - expf(-(rgbsample.y * gInvExposure)));
	rgbsample.z = __saturatef(1.0f - expf(-(rgbsample.z * gInvExposure)));
	clr00 = rgbsample;

//	if (gDenoiseEnabled && gDenoiseLerpC > 0.0f && gDenoiseLerpC < 1.0f)
//	{
        float			fCount		= 0;
        float			SumWeights	= 0;
        float3			clr			= { 0, 0, 0 };
        		
        for (int i = -gDenoiseWindowRadius; i <= gDenoiseWindowRadius; i++)
		{
            for (int j = -gDenoiseWindowRadius; j <= gDenoiseWindowRadius; j++)
            {
				// sad face...
				if (Y + i < 0)
					continue;
				if (X + j < 0)
					continue;
				if (Y + i >= gFilmHeight)
					continue;
				if (X + j >= gFilmWidth)
					continue;

				float4 clrIJ = reinterpret_cast<float4*>(inbuf)[((Y+i)*gFilmWidth) + X+j];
				XYZToRGB(clrIJ, rgbsample);
				// tone map!
				rgbsample.x = __saturatef(1.0f - expf(-(rgbsample.x * gInvExposure)));
				rgbsample.y = __saturatef(1.0f - expf(-(rgbsample.y * gInvExposure)));
				rgbsample.z = __saturatef(1.0f - expf(-(rgbsample.z * gInvExposure)));
				clrIJ = rgbsample;

				//const float4 clrIJ = tex2D(gTexRunningEstimateRgba, x + j, y + i);
				const float distanceIJ = vecLen(clr00, clrIJ);

				// gDenoiseNoise = 1/h^2
				// 
                const float weightIJ = expf(-(distanceIJ * gDenoiseNoise + (float)(i * i + j * j) * gDenoiseInvWindowArea));

                clr.x += clrIJ.x * weightIJ;
                clr.y += clrIJ.y * weightIJ;
                clr.z += clrIJ.z * weightIJ;

                SumWeights += weightIJ;

                fCount += (weightIJ > gDenoiseWeightThreshold) ? gDenoiseInvWindowArea : 0;
            }
		}
		
		SumWeights = 1.0f / SumWeights;

		clr.x *= SumWeights;
		clr.y *= SumWeights;
		clr.z *= SumWeights;

		const float LerpQ = (fCount > gDenoiseLerpThreshold) ? gDenoiseLerpC : 1.0f - gDenoiseLerpC;

		clr.x = lerpf(clr.x, clr00.x, LerpQ);
		clr.y = lerpf(clr.y, clr00.y, LerpQ);
		clr.z = lerpf(clr.z, clr00.z, LerpQ);

		//CColorRgbLdr cx(255 * clr.x, 255 * clr.y, 255 * clr.z);
		//pView->m_DisplayEstimateRgbLdr.Set(cx, X, Y);
		uchar4 pixel = make_uchar4(__saturatef(clr.x)*255.0, __saturatef(clr.y)*255.0, __saturatef(clr.z)*255.0, __saturatef(clr00.w)*255.0);
		surf2Dwrite(pixel, surfaceObj, X * 4, Y);
//	}
//	else
//	{
//		// blit the tonemapped result.  this is nonsensical.  the conditional should be before invoking these kernels.
//		// now inbuf is expected to be something else entirely, "estimatergbaldr"
//		//const CColorRgbaLdr RGBA = pView->m_EstimateRgbaLdr.Get(X, Y);
//		//CColorRgbLdr cx(RGBA.r, RGBA.g, RGBA.b);
//		//pView->m_DisplayEstimateRgbLdr.Set(cx, X, Y);
//
//		const float4 clr = reinterpret_cast<float4*>(inbuf)[pixoffset];
//		uchar4 pixel = make_uchar4(clr.x*255.0, clr.y*255.0, clr.z*255.0, clr.w*255.0);
//		surf2Dwrite(pixel, surfaceObj, X * 4, Y);
//	}
}

void Denoise(float* inbuf, cudaSurfaceObject_t surfaceObj, int w, int h, float lerpC)
{
	HandleCudaError(cudaMemcpyToSymbol(gDenoiseLerpC, &lerpC, sizeof(float)));

	const dim3 KernelBlock(KRNL_DENOISE_BLOCK_W, KRNL_DENOISE_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)w / (float)KernelBlock.x), (int)ceilf((float)h / (float)KernelBlock.y));
	KrnlDenoise << <KernelGrid, KernelBlock >> >(inbuf, surfaceObj);

	cudaDeviceSynchronize();
	HandleCudaKernelError(cudaGetLastError(), "Noise Reduction");
}
