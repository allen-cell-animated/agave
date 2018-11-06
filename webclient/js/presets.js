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

function applyPresetObj (obj) {
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

    //effectController.file = "//allen/aics/animated-cell/Allen-Cell-Explorer/Allen-Cell-Explorer_1.2.0/Cell-Viewer_Data/2017_05_15_tubulin/AICS-12/AICS-12_881.ome.tif",
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

    for (var i = 0; i < Math.min(effectController.infoObj.channelGui.length, obj.channels.length); ++i) {
        let channel = effectController.infoObj.channelGui[i];
        channel.window = obj.channels[i].window;
        channel.level = obj.channels[i].level;    
        channel.colorD = obj.channels[i].diffuseColor;
        channel.colorS = obj.channels[i].specularColor;
        channel.colorE = obj.channels[i].emissiveColor;

        channel.roughness = obj.channels[i].glossiness;
        channel.enabled = obj.channels[i].enabled;
    }

    // skyTopIntensity: 1.0,
    // skyMidIntensity: 1.0,
    // skyBotIntensity: 1.0,
    // skyTopColor: [255, 255, 255],
    // skyMidColor: [255, 255, 255],
    // skyBotColor: [255, 255, 255],
    // lightColor: [255, 255, 255],
    // lightIntensity: 100.0,
    // lightDistance: 10.0,
    // lightTheta: 0.0,
    // lightPhi: 0.0,
    // lightSize: 1.0,


    // "camera": {
    //     "aperture": 0,
    //     "exposure": 0.75,
    //     "focalDistance": 0.75,
    //     "fovY": 55,
    // },
    // "clipRegion": [
    //     [
    //         0,
    //         1
    //     ],
    //     [
    //         0,
    //         1
    //     ],
    //     [
    //         0,
    //         1
    //     ]
    // ],
    // "density": 50,
    // "lights": [
    //     {
    //         "bottomColor": [
    //             0.5,
    //             0.5,
    //             0.5
    //         ],
    //         "middleColor": [
    //             0.5,
    //             0.5,
    //             0.5
    //         ],
    //         "topColor": [
    //             0.5,
    //             0.5,
    //             0.5
    //         ],
    //         "type": 0
    //     },
    //     {
    //         "color": [
    //             100,
    //             100,
    //             100
    //         ],
    //         "distance": 10,
    //         "height": 1,
    //         "phi": 0,
    //         "theta": 0,
    //         "type": 1,
    //         "width": 1
    //     }
    // ],
    // "name": "C:/Users/danielt.ALLENINST/Downloads/AICS-12_881.ome.tif",
    // "renderIterations": 277,
    // "resolution": [
    //     1325,
    //     1054
    // ]
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

    //effectController.file = "//allen/aics/animated-cell/Allen-Cell-Explorer/Allen-Cell-Explorer_1.2.0/Cell-Viewer_Data/2017_05_15_tubulin/AICS-12/AICS-12_881.ome.tif",
    effectController.density = 100.0;
    effectController.exposure = 0.8179;
    effectController.infoObj.channelGui[0].window = 1;
    effectController.infoObj.channelGui[0].level = 0.5806;
    effectController.infoObj.channelGui[1].window = 1;
    effectController.infoObj.channelGui[1].level = 0.668;
    effectController.infoObj.channelGui[2].window = 1;
    effectController.infoObj.channelGui[2].level = 0.7408;

    updateGui();

    var cb = new commandBuffer();
    // cb.addCommand("LOAD_OME_TIF", "C:/Users/danielt.ALLENINST/Downloads/AICS-12_881.ome.tif");
    // cb.addCommand("SET_RESOLUTION", 1447, 1175);
    // cb.addCommand("RENDER_ITERATIONS", 239);
    cb.addCommand("SET_CLIP_REGION", 0, 1, 0, 1, 0, 1);
    cb.addCommand("EYE", 0.541147, 0.370615, 0.867704);
    cb.addCommand("TARGET", 0.5, 0.337662, 0.0825674);
    cb.addCommand("UP", -0.00219161, 0.999123, -0.0418183);
    cb.addCommand("FOV_Y", 55);
    cb.addCommand("EXPOSURE", 0.8179);
    cb.addCommand("DENSITY", 100);
    cb.addCommand("APERTURE", 0);
    cb.addCommand("FOCALDIST", 0.75);
    cb.addCommand("ENABLE_CHANNEL", 0, 1);
    cb.addCommand("MAT_DIFFUSE", 0, 1, 0, 1, 1.0);
    cb.addCommand("MAT_SPECULAR", 0, 0.423529, 0.423529, 0.423529, 0.0);
    cb.addCommand("MAT_EMISSIVE", 0, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 0, 100);
    cb.addCommand("SET_WINDOW_LEVEL", 0, 1, 0.5806);
    cb.addCommand("ENABLE_CHANNEL", 1, 1);
    cb.addCommand("MAT_DIFFUSE", 1, 1, 1, 1, 1.0);
    cb.addCommand("MAT_SPECULAR", 1, 0.368627, 0.368627, 0.368627, 0.0);
    cb.addCommand("MAT_EMISSIVE", 1, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 1, 100);
    cb.addCommand("SET_WINDOW_LEVEL", 1, 1, 0.668);
    cb.addCommand("ENABLE_CHANNEL", 2, 1);
    cb.addCommand("MAT_DIFFUSE", 2, 0, 1, 1, 1.0);
    cb.addCommand("MAT_SPECULAR", 2, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 2, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 2, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 2, 1, 0.7408);
    cb.addCommand("ENABLE_CHANNEL", 3, 0);
    cb.addCommand("MAT_DIFFUSE", 3, 1, 0, 0, 1.0);
    cb.addCommand("MAT_SPECULAR", 3, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 3, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 3, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 3, 0.380392, 0.527451);
    cb.addCommand("ENABLE_CHANNEL", 4, 0);
    cb.addCommand("MAT_DIFFUSE", 4, 0, 0.291844, 1, 1.0);
    cb.addCommand("MAT_SPECULAR", 4, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 4, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 4, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 4, 0, 1);
    cb.addCommand("ENABLE_CHANNEL", 5, 0);
    cb.addCommand("MAT_DIFFUSE", 5, 0.583673, 1, 0, 1.0);
    cb.addCommand("MAT_SPECULAR", 5, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 5, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 5, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 5, 0.921569, 0.539216);
    cb.addCommand("ENABLE_CHANNEL", 6, 0);
    cb.addCommand("MAT_DIFFUSE", 6, 1, 0, 0.875334, 1.0);
    cb.addCommand("MAT_SPECULAR", 6, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 6, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 6, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 6, 0.921569, 0.539216);
    cb.addCommand("ENABLE_CHANNEL", 7, 0);
    cb.addCommand("MAT_DIFFUSE", 7, 0, 1, 0.832837, 1.0);
    cb.addCommand("MAT_SPECULAR", 7, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 7, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 7, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 7, 0.921569, 0.539216);
    cb.addCommand("ENABLE_CHANNEL", 8, 0);
    cb.addCommand("MAT_DIFFUSE", 8, 1, 0.541009, 0, 1.0);
    cb.addCommand("MAT_SPECULAR", 8, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 8, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 8, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 8, 0.921569, 0.539216);
    cb.addCommand("SKYLIGHT_TOP_COLOR", 0.5, 0.5, 0.5);
    cb.addCommand("SKYLIGHT_MIDDLE_COLOR", 0.5, 0.5, 0.5);
    cb.addCommand("SKYLIGHT_BOTTOM_COLOR", 0.5, 0.5, 0.5);
    cb.addCommand("LIGHT_POS", 0, 10, 0, 0);
    cb.addCommand("LIGHT_COLOR", 0, 100, 100, 100);
    cb.addCommand("LIGHT_SIZE", 0, 1, 1);
    flushCommandBuffer(cb);
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

    //effectController.file = "//allen/aics/animated-cell/Allen-Cell-Explorer/Allen-Cell-Explorer_1.2.0/Cell-Viewer_Data/2017_05_15_tubulin/AICS-12/AICS-12_881.ome.tif",
    effectController.density = 100.0;
    effectController.exposure = 0.8179;
    
    effectController.infoObj.channelGui[0].window = 1;
    effectController.infoObj.channelGui[0].level = 0.5689;
    effectController.infoObj.channelGui[1].window = 1;
    effectController.infoObj.channelGui[1].level = 0.6301;
    effectController.infoObj.channelGui[2].window = 1;
    effectController.infoObj.channelGui[2].level = 0.732;

    updateGui();
    
    var cb = new commandBuffer();
    //cb.addCommand("LOAD_OME_TIF", "C:/Users/danielt.ALLENINST/Downloads/AICS-11_409.ome.tif");
    // cb.addCommand("SET_RESOLUTION", 1447, 1175);
    // cb.addCommand("RENDER_ITERATIONS", 384);
    // cb.addCommand("SET_CLIP_REGION", 0, 1, 0, 1, 0, 1);
    cb.addCommand("EYE", 0.5, 0.337662, 1.26718);
    cb.addCommand("TARGET", 0.5, 0.337662, 0.0941558);
    cb.addCommand("UP", 0, 1, 0);
    cb.addCommand("FOV_Y", 55);
    cb.addCommand("EXPOSURE", 0.8179);
    cb.addCommand("DENSITY", 100);
    cb.addCommand("APERTURE", 0);
    cb.addCommand("FOCALDIST", 0.75);
    cb.addCommand("ENABLE_CHANNEL", 0, 1);
    cb.addCommand("MAT_DIFFUSE", 0, 1, 0, 1, 1.0);
    cb.addCommand("MAT_SPECULAR", 0, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 0, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 0, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 0, 1, 0.5689);
    cb.addCommand("ENABLE_CHANNEL", 1, 1);
    cb.addCommand("MAT_DIFFUSE", 1, 1, 1, 1, 1.0);
    cb.addCommand("MAT_SPECULAR", 1, 0.415686, 0.415686, 0.415686, 0.0);
    cb.addCommand("MAT_EMISSIVE", 1, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 1, 100);
    cb.addCommand("SET_WINDOW_LEVEL", 1, 1, 0.6301);
    cb.addCommand("ENABLE_CHANNEL", 2, 1);
    cb.addCommand("MAT_DIFFUSE", 2, 0, 1, 1, 1.0);
    cb.addCommand("MAT_SPECULAR", 2, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 2, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 2, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 2, 1, 0.732);
    cb.addCommand("ENABLE_CHANNEL", 3, 0);
    cb.addCommand("MAT_DIFFUSE", 3, 1, 0, 0, 1.0);
    cb.addCommand("MAT_SPECULAR", 3, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 3, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 3, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 3, 0.290196, 0.364706);
    cb.addCommand("ENABLE_CHANNEL", 4, 0);
    cb.addCommand("MAT_DIFFUSE", 4, 0, 0.291844, 1, 1.0);
    cb.addCommand("MAT_SPECULAR", 4, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 4, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 4, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 4, 0, 1);
    cb.addCommand("ENABLE_CHANNEL", 5, 0);
    cb.addCommand("MAT_DIFFUSE", 5, 0.583673, 1, 0, 1.0);
    cb.addCommand("MAT_SPECULAR", 5, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 5, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 5, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 5, 0.898039, 0.55098);
    cb.addCommand("ENABLE_CHANNEL", 6, 0);
    cb.addCommand("MAT_DIFFUSE", 6, 1, 0, 0.875334, 1.0);
    cb.addCommand("MAT_SPECULAR", 6, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 6, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 6, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 6, 0.898039, 0.55098);
    cb.addCommand("ENABLE_CHANNEL", 7, 0);
    cb.addCommand("MAT_DIFFUSE", 7, 0, 1, 0.832837, 1.0);
    cb.addCommand("MAT_SPECULAR", 7, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 7, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 7, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 7, 0.898039, 0.55098);
    cb.addCommand("SKYLIGHT_TOP_COLOR", 0.5, 0.5, 0.5);
    cb.addCommand("SKYLIGHT_MIDDLE_COLOR", 0.5, 0.5, 0.5);
    cb.addCommand("SKYLIGHT_BOTTOM_COLOR", 0.5, 0.5, 0.5);
    cb.addCommand("LIGHT_POS", 0, 10, 0, 0);
    cb.addCommand("LIGHT_COLOR", 0, 100, 100, 100);
    cb.addCommand("LIGHT_SIZE", 0, 1, 1);
    flushCommandBuffer(cb);
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

    //effectController.file = "//allen/aics/animated-cell/Allen-Cell-Explorer/Allen-Cell-Explorer_1.2.0/Cell-Viewer_Data/2017_05_15_tubulin/AICS-12/AICS-12_881.ome.tif",
    effectController.density = 100.0;
    effectController.exposure = 0.8179;

    updateGui();

    var cb = new commandBuffer();
    //cb.addCommand("LOAD_OME_TIF", "C:/Users/danielt.ALLENINST/Downloads/AICS-13_319.ome.tif");
    //cb.addCommand("SET_RESOLUTION", 1447, 1175);
    //cb.addCommand("RENDER_ITERATIONS", 222);
    cb.addCommand("SET_CLIP_REGION", 0, 1, 0, 1, 0, 1);
    cb.addCommand("EYE", 0.535138, -0.22972, 0.728428);
    cb.addCommand("TARGET", 0.5, 0.337662, 0.0579421);
    cb.addCommand("UP", 0.0337808, 0.763795, 0.644573);
    cb.addCommand("FOV_Y", 55);
    cb.addCommand("EXPOSURE", 0.8179);
    cb.addCommand("DENSITY", 100);
    cb.addCommand("APERTURE", 0);
    cb.addCommand("FOCALDIST", 0.75);
    cb.addCommand("ENABLE_CHANNEL", 0, 1);
    cb.addCommand("MAT_DIFFUSE", 0, 1, 0, 1, 1.0);
    cb.addCommand("MAT_SPECULAR", 0, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 0, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 0, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 0, 1, 0.6854);
    cb.addCommand("ENABLE_CHANNEL", 1, 1);
    cb.addCommand("MAT_DIFFUSE", 1, 1, 1, 1, 1.0);
    cb.addCommand("MAT_SPECULAR", 1, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 1, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 1, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 1, 1, 0.601);
    cb.addCommand("ENABLE_CHANNEL", 2, 1);
    cb.addCommand("MAT_DIFFUSE", 2, 0, 1, 1, 1.0);
    cb.addCommand("MAT_SPECULAR", 2, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 2, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 2, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 2, 1, 0.7757);
    cb.addCommand("ENABLE_CHANNEL", 3, 0);
    cb.addCommand("MAT_DIFFUSE", 3, 1, 0, 0, 1.0);
    cb.addCommand("MAT_SPECULAR", 3, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 3, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 3, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 3, 0.65098, 0.67451);
    cb.addCommand("ENABLE_CHANNEL", 4, 0);
    cb.addCommand("MAT_DIFFUSE", 4, 0, 0.291844, 1, 1.0);
    cb.addCommand("MAT_SPECULAR", 4, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 4, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 4, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 4, 0, 1);
    cb.addCommand("ENABLE_CHANNEL", 5, 0);
    cb.addCommand("MAT_DIFFUSE", 5, 0.583673, 1, 0, 1.0);
    cb.addCommand("MAT_SPECULAR", 5, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 5, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 5, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 5, 0.811765, 0.594118);
    cb.addCommand("ENABLE_CHANNEL", 6, 0);
    cb.addCommand("MAT_DIFFUSE", 6, 1, 0, 0.875334, 1.0);
    cb.addCommand("MAT_SPECULAR", 6, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 6, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 6, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 6, 0.811765, 0.594118);
    cb.addCommand("ENABLE_CHANNEL", 7, 0);
    cb.addCommand("MAT_DIFFUSE", 7, 0, 1, 0.832837, 1.0);
    cb.addCommand("MAT_SPECULAR", 7, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 7, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 7, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 7, 0.811765, 0.594118);
    cb.addCommand("ENABLE_CHANNEL", 8, 0);
    cb.addCommand("MAT_DIFFUSE", 8, 1, 0.541009, 0, 1.0);
    cb.addCommand("MAT_SPECULAR", 8, 0, 0, 0, 0.0);
    cb.addCommand("MAT_EMISSIVE", 8, 0, 0, 0, 0.0);
    cb.addCommand("MAT_GLOSSINESS", 8, 0);
    cb.addCommand("SET_WINDOW_LEVEL", 8, 0.811765, 0.594118);
    cb.addCommand("SKYLIGHT_TOP_COLOR", 0.5, 0.5, 0.5);
    cb.addCommand("SKYLIGHT_MIDDLE_COLOR", 0.5, 0.5, 0.5);
    cb.addCommand("SKYLIGHT_BOTTOM_COLOR", 0.5, 0.5, 0.5);
    cb.addCommand("LIGHT_POS", 0, 10, 0, 0);
    cb.addCommand("LIGHT_COLOR", 0, 100, 100, 100);
    cb.addCommand("LIGHT_SIZE", 0, 1, 1);
    flushCommandBuffer(cb);
}
const pathprefix = "/allen/aics/animated-cell/Allen-Cell-Explorer/Allen-Cell-Explorer_1.2.0/Cell-Viewer_Data/";
const presets = [
    {name:"2017_05_15_tubulin/AICS-12/AICS-12_881.ome.tif", f:preset0},
    {name:"2017_07_21_Tom20/AICS-11/AICS-11_409.ome.tif", f:preset1},
    {name:"2017_06_28_lamin/AICS-13/AICS-13_319.ome.tif", f:preset2}
]
let loading_preset = null;
function executePreset(index) {
    // load image and wait and then execute the rest of the command buffer...
    const fpath = pathprefix + presets[index].name;
    var cb = new commandBuffer();
    cb.addCommand("LOAD_OME_TIF", fpath);
    flushCommandBuffer(cb);
    _stream_mode_suspended = true;

    loading_preset = presets[index].f;
}
function applyPresets() {
    if (loading_preset) {
        loading_preset();
        loading_preset = null;
    }
}
