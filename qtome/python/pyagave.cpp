#include "pybind11/embed.h"

#include "pyrenderer.h"

namespace py = pybind11;

PYBIND11_EMBEDDED_MODULE(agave, m)
{
  m.doc() = "agave plugin"; // optional module docstring

  py::class_<OffscreenRenderer>(m, "renderer")
    //.def(py::init<>())
    .def("session", &OffscreenRenderer::Session)
    .def("asset_path", &OffscreenRenderer::AssetPath)
    .def("load_ome_tif", &OffscreenRenderer::LoadOmeTif)
    .def("eye", &OffscreenRenderer::Eye)
    .def("target", &OffscreenRenderer::Target)
    .def("up", &OffscreenRenderer::Up)
    .def("aperture", &OffscreenRenderer::Aperture)
    .def("camera_projection", &OffscreenRenderer::CameraProjection)
    .def("focaldist", &OffscreenRenderer::Focaldist)
    .def("exposure", &OffscreenRenderer::Exposure)
    .def("mat_diffuse", &OffscreenRenderer::MatDiffuse)
    .def("mat_specular", &OffscreenRenderer::MatSpecular)
    .def("mat_emissive", &OffscreenRenderer::MatEmissive)
    .def("render_iterations", &OffscreenRenderer::RenderIterations)
    .def("stream_mode", &OffscreenRenderer::StreamMode)
    .def("redraw", &OffscreenRenderer::Redraw)
    .def("set_resolution", &OffscreenRenderer::SetResolution)
    .def("density", &OffscreenRenderer::Density)
    .def("frame_scene", &OffscreenRenderer::FrameScene)
    .def("mat_glossiness", &OffscreenRenderer::MatGlossiness)
    .def("enable_channel", &OffscreenRenderer::EnableChannel)
    .def("set_window_level", &OffscreenRenderer::SetWindowLevel)
    .def("orbit_camera", &OffscreenRenderer::OrbitCamera)
    .def("skylight_top_color", &OffscreenRenderer::SkylightTopColor)
    .def("skylight_middle_color", &OffscreenRenderer::SkylightMiddleColor)
    .def("skylight_bottom_color", &OffscreenRenderer::SkylightBottomColor)
    .def("light_pos", &OffscreenRenderer::LightPos)
    .def("light_color", &OffscreenRenderer::LightColor)
    .def("light_size", &OffscreenRenderer::LightSize)
    .def("set_clip_region", &OffscreenRenderer::SetClipRegion)
    .def("set_voxel_scale", &OffscreenRenderer::SetVoxelScale)
    .def("auto_threshold", &OffscreenRenderer::AutoThreshold)
    .def("set_percentile_threshold", &OffscreenRenderer::SetPercentileThreshold)
    .def("mat_opacity", &OffscreenRenderer::MatOpacity)
    .def("set_primary_ray_step_size", &OffscreenRenderer::SetPrimaryRayStepSize)
    .def("set_secondary_ray_step_size", &OffscreenRenderer::SetSecondaryRayStepSize)
    .def("background_color", &OffscreenRenderer::BackgroundColor);
}
