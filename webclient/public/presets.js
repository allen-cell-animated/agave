function updateGui() {
  for (var i in gui.__controllers) {
    gui.__controllers[i].updateDisplay();
  }
  for (var i in gui.__folders) {
    for (var j in gui.__folders[i].__controllers) {
      gui.__folders[i].__controllers[j].updateDisplay();
    }
  }
}

function applyPresetObj(obj) {
  // skip obj.resolution
  gCamera.position.x = obj.camera.eye[0];
  gCamera.position.y = obj.camera.eye[1];
  gCamera.position.z = obj.camera.eye[2];
  gCamera.up.x = obj.camera.up[0];
  gCamera.up.y = obj.camera.up[1];
  gCamera.up.z = obj.camera.up[2];
  gControls.target.x = obj.camera.target[0];
  gControls.target.y = obj.camera.target[1];
  gControls.target.z = obj.camera.target[2];
  gControls.target0 = gControls.target.clone();

  effectController.density = obj.density;
  effectController.exposure = obj.camera.exposure;
  effectController.fov = obj.camera.fovY;
  effectController.focal_distance = obj.camera.focalDistance;

  effectController.xmin = obj.clipRegion[0][0];
  effectController.xmax = obj.clipRegion[0][1];
  effectController.ymin = obj.clipRegion[1][0];
  effectController.ymax = obj.clipRegion[1][1];
  effectController.zmin = obj.clipRegion[2][0];
  effectController.zmax = obj.clipRegion[2][1];

  for (
    var i = 0;
    i <
    Math.min(effectController.infoObj.channelGui.length, obj.channels.length);
    ++i
  ) {
    let channel = effectController.infoObj.channelGui[i];
    channel.window = obj.channels[i].window;
    channel.level = obj.channels[i].level;
    channel.colorD = obj.channels[i].diffuseColor;
    channel.colorS = obj.channels[i].specularColor;
    channel.colorE = obj.channels[i].emissiveColor;

    channel.roughness = obj.channels[i].glossiness;
    channel.enabled = obj.channels[i].enabled;
  }
}

function preset0() {
  gCamera.position.x = 0.541147;
  gCamera.position.y = 0.370615;
  gCamera.position.z = 0.867704;
  gCamera.up.x = -0.00219161;
  gCamera.up.y = 0.999123;
  gCamera.up.z = -0.0418183;
  gControls.target.x = 0.5;
  gControls.target.y = 0.337662;
  gControls.target.z = 0.0825674;
  gControls.target0 = gControls.target.clone();

  effectController.density = 100.0;
  effectController.exposure = 0.8179;
  effectController.infoObj.channelGui[0].window = 1;
  effectController.infoObj.channelGui[0].level = 0.5806;
  effectController.infoObj.channelGui[1].window = 1;
  effectController.infoObj.channelGui[1].level = 0.668;
  effectController.infoObj.channelGui[2].window = 1;
  effectController.infoObj.channelGui[2].level = 0.7408;

  updateGui();

  agave.set_clip_region(0, 1, 0, 1, 0, 1);
  agave.eye(0.541147, 0.370615, 0.867704);
  agave.target(0.5, 0.337662, 0.0825674);
  agave.up(-0.00219161, 0.999123, -0.0418183);
  agave.camera_projection(0, 55);
  agave.exposure(0.8179);
  agave.density(100);
  agave.aperture(0);
  agave.focaldist(0.75);
  agave.enable_channel(0, 1);
  agave.mat_diffuse(0, 1, 0, 1, 1.0);
  agave.mat_specular(0, 0.423529, 0.423529, 0.423529, 0.0);
  agave.mat_emissive(0, 0, 0, 0, 0.0);
  agave.mat_glossiness(0, 100);
  agave.set_window_level(0, 1, 0.5806);
  agave.enable_channel(1, 1);
  agave.mat_diffuse(1, 1, 1, 1, 1.0);
  agave.mat_specular(1, 0.368627, 0.368627, 0.368627, 0.0);
  agave.mat_emissive(1, 0, 0, 0, 0.0);
  agave.mat_glossiness(1, 100);
  agave.set_window_level(1, 1, 0.668);
  agave.enable_channel(2, 1);
  agave.mat_diffuse(2, 0, 1, 1, 1.0);
  agave.mat_specular(2, 0, 0, 0, 0.0);
  agave.mat_emissive(2, 0, 0, 0, 0.0);
  agave.mat_glossiness(2, 0);
  agave.set_window_level(2, 1, 0.7408);
  agave.enable_channel(3, 0);
  agave.mat_diffuse(3, 1, 0, 0, 1.0);
  agave.mat_specular(3, 0, 0, 0, 0.0);
  agave.mat_emissive(3, 0, 0, 0, 0.0);
  agave.mat_glossiness(3, 0);
  agave.set_window_level(3, 0.380392, 0.527451);
  agave.enable_channel(4, 0);
  agave.mat_diffuse(4, 0, 0.291844, 1, 1.0);
  agave.mat_specular(4, 0, 0, 0, 0.0);
  agave.mat_emissive(4, 0, 0, 0, 0.0);
  agave.mat_glossiness(4, 0);
  agave.set_window_level(4, 0, 1);
  agave.enable_channel(5, 0);
  agave.mat_diffuse(5, 0.583673, 1, 0, 1.0);
  agave.mat_specular(5, 0, 0, 0, 0.0);
  agave.mat_emissive(5, 0, 0, 0, 0.0);
  agave.mat_glossiness(5, 0);
  agave.set_window_level(5, 0.921569, 0.539216);
  agave.enable_channel(6, 0);
  agave.mat_diffuse(6, 1, 0, 0.875334, 1.0);
  agave.mat_specular(6, 0, 0, 0, 0.0);
  agave.mat_emissive(6, 0, 0, 0, 0.0);
  agave.mat_glossiness(6, 0);
  agave.set_window_level(6, 0.921569, 0.539216);
  agave.enable_channel(7, 0);
  agave.mat_diffuse(7, 0, 1, 0.832837, 1.0);
  agave.mat_specular(7, 0, 0, 0, 0.0);
  agave.mat_emissive(7, 0, 0, 0, 0.0);
  agave.mat_glossiness(7, 0);
  agave.set_window_level(7, 0.921569, 0.539216);
  agave.enable_channel(8, 0);
  agave.mat_diffuse(8, 1, 0.541009, 0, 1.0);
  agave.mat_specular(8, 0, 0, 0, 0.0);
  agave.mat_emissive(8, 0, 0, 0, 0.0);
  agave.mat_glossiness(8, 0);
  agave.set_window_level(8, 0.921569, 0.539216);
  agave.skylight_top_color(0.5, 0.5, 0.5);
  agave.skylight_middle_color(0.5, 0.5, 0.5);
  agave.skylight_bottom_color(0.5, 0.5, 0.5);
  agave.light_pos(0, 10, 0, 0);
  agave.light_color(0, 100, 100, 100);
  agave.light_size(0, 1, 1);
  agave.flushCommandBuffer();
}
function preset1() {
  gCamera.position.x = 0.5;
  gCamera.position.y = 0.337662;
  gCamera.position.z = 1.26718;
  gCamera.up.x = 0;
  gCamera.up.y = 1;
  gCamera.up.z = 0;
  gControls.target.x = 0.5;
  gControls.target.y = 0.337662;
  gControls.target.z = 0.0941558;
  gControls.target0 = gControls.target.clone();

  effectController.density = 100.0;
  effectController.exposure = 0.8179;

  effectController.infoObj.channelGui[0].window = 1;
  effectController.infoObj.channelGui[0].level = 0.5689;
  effectController.infoObj.channelGui[1].window = 1;
  effectController.infoObj.channelGui[1].level = 0.6301;
  effectController.infoObj.channelGui[2].window = 1;
  effectController.infoObj.channelGui[2].level = 0.732;

  updateGui();

  agave.eye(0.5, 0.337662, 1.26718);
  agave.target(0.5, 0.337662, 0.0941558);
  agave.up(0, 1, 0);
  agave.camera_projection(0, 55);
  agave.exposure(0.8179);
  agave.density(100);
  agave.aperture(0);
  agave.focaldist(0.75);
  agave.enable_channel(0, 1);
  agave.mat_diffuse(0, 1, 0, 1, 1.0);
  agave.mat_specular(0, 0, 0, 0, 0.0);
  agave.mat_emissive(0, 0, 0, 0, 0.0);
  agave.mat_glossiness(0, 0);
  agave.set_window_level(0, 1, 0.5689);
  agave.enable_channel(1, 1);
  agave.mat_diffuse(1, 1, 1, 1, 1.0);
  agave.mat_specular(1, 0.415686, 0.415686, 0.415686, 0.0);
  agave.mat_emissive(1, 0, 0, 0, 0.0);
  agave.mat_glossiness(1, 100);
  agave.set_window_level(1, 1, 0.6301);
  agave.enable_channel(2, 1);
  agave.mat_diffuse(2, 0, 1, 1, 1.0);
  agave.mat_specular(2, 0, 0, 0, 0.0);
  agave.mat_emissive(2, 0, 0, 0, 0.0);
  agave.mat_glossiness(2, 0);
  agave.set_window_level(2, 1, 0.732);
  agave.enable_channel(3, 0);
  agave.mat_diffuse(3, 1, 0, 0, 1.0);
  agave.mat_specular(3, 0, 0, 0, 0.0);
  agave.mat_emissive(3, 0, 0, 0, 0.0);
  agave.mat_glossiness(3, 0);
  agave.set_window_level(3, 0.290196, 0.364706);
  agave.enable_channel(4, 0);
  agave.mat_diffuse(4, 0, 0.291844, 1, 1.0);
  agave.mat_specular(4, 0, 0, 0, 0.0);
  agave.mat_emissive(4, 0, 0, 0, 0.0);
  agave.mat_glossiness(4, 0);
  agave.set_window_level(4, 0, 1);
  agave.enable_channel(5, 0);
  agave.mat_diffuse(5, 0.583673, 1, 0, 1.0);
  agave.mat_specular(5, 0, 0, 0, 0.0);
  agave.mat_emissive(5, 0, 0, 0, 0.0);
  agave.mat_glossiness(5, 0);
  agave.set_window_level(5, 0.898039, 0.55098);
  agave.enable_channel(6, 0);
  agave.mat_diffuse(6, 1, 0, 0.875334, 1.0);
  agave.mat_specular(6, 0, 0, 0, 0.0);
  agave.mat_emissive(6, 0, 0, 0, 0.0);
  agave.mat_glossiness(6, 0);
  agave.set_window_level(6, 0.898039, 0.55098);
  agave.enable_channel(7, 0);
  agave.mat_diffuse(7, 0, 1, 0.832837, 1.0);
  agave.mat_specular(7, 0, 0, 0, 0.0);
  agave.mat_emissive(7, 0, 0, 0, 0.0);
  agave.mat_glossiness(7, 0);
  agave.set_window_level(7, 0.898039, 0.55098);
  agave.skylight_top_color(0.5, 0.5, 0.5);
  agave.skylight_middle_color(0.5, 0.5, 0.5);
  agave.skylight_bottom_color(0.5, 0.5, 0.5);
  agave.light_pos(0, 10, 0, 0);
  agave.light_color(0, 100, 100, 100);
  agave.light_size(0, 1, 1);
  agave.flushCommandBuffer();
}
function preset2() {
  gCamera.position.x = 0.535138;
  gCamera.position.y = -0.22972;
  gCamera.position.z = 0.728428;
  gCamera.up.x = 0.0337808;
  gCamera.up.y = 0.763795;
  gCamera.up.z = 0.644573;
  gControls.target.x = 0.5;
  gControls.target.y = 0.337662;
  gControls.target.z = 0.0579421;
  gControls.target0 = gControls.target.clone();

  effectController.density = 100.0;
  effectController.exposure = 0.8179;

  updateGui();

  agave.set_clip_region(0, 1, 0, 1, 0, 1);
  agave.eye(0.535138, -0.22972, 0.728428);
  agave.target(0.5, 0.337662, 0.0579421);
  agave.up(0.0337808, 0.763795, 0.644573);
  agave.camera_projection(0, 55);
  agave.exposure(0.8179);
  agave.density(100);
  agave.aperture(0);
  agave.focaldist(0.75);
  agave.enable_channel(0, 1);
  agave.mat_diffuse(0, 1, 0, 1, 1.0);
  agave.mat_specular(0, 0, 0, 0, 0.0);
  agave.mat_emissive(0, 0, 0, 0, 0.0);
  agave.mat_glossiness(0, 0);
  agave.set_window_level(0, 1, 0.6854);
  agave.enable_channel(1, 1);
  agave.mat_diffuse(1, 1, 1, 1, 1.0);
  agave.mat_specular(1, 0, 0, 0, 0.0);
  agave.mat_emissive(1, 0, 0, 0, 0.0);
  agave.mat_glossiness(1, 0);
  agave.set_window_level(1, 1, 0.601);
  agave.enable_channel(2, 1);
  agave.mat_diffuse(2, 0, 1, 1, 1.0);
  agave.mat_specular(2, 0, 0, 0, 0.0);
  agave.mat_emissive(2, 0, 0, 0, 0.0);
  agave.mat_glossiness(2, 0);
  agave.set_window_level(2, 1, 0.7757);
  agave.enable_channel(3, 0);
  agave.mat_diffuse(3, 1, 0, 0, 1.0);
  agave.mat_specular(3, 0, 0, 0, 0.0);
  agave.mat_emissive(3, 0, 0, 0, 0.0);
  agave.mat_glossiness(3, 0);
  agave.set_window_level(3, 0.65098, 0.67451);
  agave.enable_channel(4, 0);
  agave.mat_diffuse(4, 0, 0.291844, 1, 1.0);
  agave.mat_specular(4, 0, 0, 0, 0.0);
  agave.mat_emissive(4, 0, 0, 0, 0.0);
  agave.mat_glossiness(4, 0);
  agave.set_window_level(4, 0, 1);
  agave.enable_channel(5, 0);
  agave.mat_diffuse(5, 0.583673, 1, 0, 1.0);
  agave.mat_specular(5, 0, 0, 0, 0.0);
  agave.mat_emissive(5, 0, 0, 0, 0.0);
  agave.mat_glossiness(5, 0);
  agave.set_window_level(5, 0.811765, 0.594118);
  agave.enable_channel(6, 0);
  agave.mat_diffuse(6, 1, 0, 0.875334, 1.0);
  agave.mat_specular(6, 0, 0, 0, 0.0);
  agave.mat_emissive(6, 0, 0, 0, 0.0);
  agave.mat_glossiness(6, 0);
  agave.set_window_level(6, 0.811765, 0.594118);
  agave.enable_channel(7, 0);
  agave.mat_diffuse(7, 0, 1, 0.832837, 1.0);
  agave.mat_specular(7, 0, 0, 0, 0.0);
  agave.mat_emissive(7, 0, 0, 0, 0.0);
  agave.mat_glossiness(7, 0);
  agave.set_window_level(7, 0.811765, 0.594118);
  agave.enable_channel(8, 0);
  agave.mat_diffuse(8, 1, 0.541009, 0, 1.0);
  agave.mat_specular(8, 0, 0, 0, 0.0);
  agave.mat_emissive(8, 0, 0, 0, 0.0);
  agave.mat_glossiness(8, 0);
  agave.set_window_level(8, 0.811765, 0.594118);
  agave.skylight_top_color(0.5, 0.5, 0.5);
  agave.skylight_middle_color(0.5, 0.5, 0.5);
  agave.skylight_bottom_color(0.5, 0.5, 0.5);
  agave.light_pos(0, 10, 0, 0);
  agave.light_color(0, 100, 100, 100);
  agave.light_size(0, 1, 1);
  agave.flushCommandBuffer();
}

const pathprefix = "/agavedata/";
const presets = [
  { name: "AICS-12_881.ome.tif", f: preset0 },
  { name: "AICS-11_409.ome.tif", f: preset1 },
  { name: "AICS-13_319.ome.tif", f: preset2 },
];
let loading_preset = null;
function executePreset(index) {
  // load image and wait and then execute the rest of the command buffer...
  const fpath = pathprefix + presets[index].name;
  agave.load_data(fpath, 0, 0, 0, [], []);
  agave.flushCommandBuffer();
  _stream_mode_suspended = true;

  loading_preset = presets[index].f;
}
function applyPresets() {
  if (loading_preset) {
    loading_preset();
    loading_preset = null;
  }
}
