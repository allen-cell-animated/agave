#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "phong.h"

using namespace optix;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtBuffer<float4, 2>   result_buffer;
//rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(float3, draw_color, , );

rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );

rtDeclareVariable(PerRayData_radiance,
	prd_radiance, rtPayload, );

RT_PROGRAM void miss()
{
	//result_buffer[launch_index] = make_float4(prd.result, 0.0f);
	prd_radiance.result = draw_color;
}
RT_PROGRAM void exception()
{
	result_buffer[launch_index] = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
}

RT_PROGRAM void pinhole_camera()
{
	size_t2 screen = result_buffer.size();

	float2 d = make_float2(launch_index) /
		make_float2(screen) * 2.f - 1.f;
	float3 ray_origin = eye;
	float3 ray_direction = normalize(d.x*U + d.y*V + W);

	optix::Ray ray(ray_origin, ray_direction,
		0, scene_epsilon);
	    //radiance_ray_type, scene_epsilon);
	PerRayData_radiance prd;
	prd.importance = 1.f;
	prd.depth = 0;
	//prd.result = draw_color;

	rtTrace(top_object, ray, prd);

	result_buffer[launch_index] = make_float4(prd.result, 1.0f);
	//result_buffer[launch_index] = make_float4(draw_color, 0.f);
}
