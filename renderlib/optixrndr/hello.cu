#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtBuffer<float4, 2>   result_buffer;

rtDeclareVariable(float3,                draw_color, , );

RT_PROGRAM void draw_solid_color()
{
  result_buffer[launch_index] = make_float4(draw_color, 0.f);
}
