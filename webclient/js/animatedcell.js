//var wsUri = "ws://localhost:1235";
var wsUri = "ws://dev-aics-dtp-001:1235";

var binarysocket0 = null; //handles requests for image streaming target #1
//var binarysocket1 = null; //handles requests for image streaming target #2
var jsonsocket0 = null; //handles requests for dynamically populating the menu entries based on server feedback

var dragFlag = 0; //for dragging in the render view
var selectDragFlag = 0; //for dragging in the cell structure visibility widget
var initialMouseX = 0;
var initialMouseY = 0;
var mouseSensi = 0.2;
var img_width = 0;
var img_height = 0;

//quaternions
var rotation;
var oldRotation;
var rotationDelta;
var tempold;
var slider_drag = false;

var _stream_mode = false;
var _stream_mode_suspended = false;
var enqueued_image_data = null;
var waiting_for_image = false;


/**
 * switches the supplied element to (in)visible
 * @param element
 * @param visible
 */
function toggleDivVisibility(element, visible) {
    element.style.visibility = (visible ? "visible" : "hidden");
}


var binarysock, jsonsock;
var gui;

function setupGui() {

    effectController = {
        resolution: "512x512",
        file: "//allen/aics/animated-cell/Allen-Cell-Explorer/Allen-Cell-Explorer_1.2.0/Cell-Viewer_Data/2017_05_15_tubulin/AICS-12/AICS-12_881.ome.tif",
        density: 50.0,
        exposure: 0.75,
        aperture: 0.0,
        fov: 55,
        focal_distance: 4.0,
        stream: true,
        skyTopIntensity: 1.0,
        skyMidIntensity: 1.0,
        skyBotIntensity: 1.0,
        skyTopColor: [255, 255, 255],
        skyMidColor: [255, 255, 255],
        skyBotColor: [255, 255, 255],
        lightColor: [255, 255, 255],
        lightIntensity: 100.0,
        lightDistance: 10.0,
        lightTheta: 0.0,
        lightPhi: 0.0,
        lightSize: 1.0,
        xmin: 0.0,
        ymin: 0.0,
        zmin: 0.0,
        xmax: 1.0,
        ymax: 1.0,
        zmax: 1.0,
        resetCamera: resetCamera,
        preset0: function() { executePreset(0); },
        preset1: function() { executePreset(1); },
        preset2: function() { executePreset(2); }
    };

    gui = new dat.GUI();
    //gui = new dat.GUI({autoPlace:false, width:200});

    gui.add(effectController, 'preset0').name('Tubulin');
    gui.add(effectController, 'preset1').name('TOM20');
    gui.add(effectController, 'preset2').name('LaminB');

    // gui.add(effectController, "file").onFinishChange(function (value) {
    //     var cb = new commandBuffer();
    //     cb.addCommand("LOAD_OME_TIF", value);
    //     flushCommandBuffer(cb);
    //     _stream_mode_suspended = true;
    // });

    gui.add(effectController, "resolution", ["256x256", "512x512", "1024x1024", "1024x768"]).onChange(function (value) {
        var res = value.match(/(\d+)x(\d+)/);
        if (res.length === 3) {
            res[0] = parseInt(res[1]);
            res[1] = parseInt(res[2]);
            var imgholder = document.getElementById("imageA");
            imgholder.width = res[0];
            imgholder.height = res[1];
            imgholder.style.width = res[0];
            imgholder.style.height = res[1];

            var cb = new commandBuffer();
            cb.addCommand("SET_RESOLUTION", res[0], res[1]);
            flushCommandBuffer(cb);
        }
    });

    gui.add(effectController, "resetCamera");
    //allen/aics/animated-cell/Allen-Cell-Explorer/Allen-Cell-Explorer_1.2.0/Cell-Viewer_Data/2017_05_15_tubulin/AICS-12/AICS-12_790.ome.tif
    gui.add(effectController, "stream").onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("STREAM_MODE", value ? 1 : 0);
        flushCommandBuffer(cb);
        // BUG THIS SHOULD NOT BE NEEDED.
        var cb2 = new commandBuffer();
        cb2.addCommand("REDRAW");
        flushCommandBuffer(cb2);
        _stream_mode = value;
    });
    gui.add(effectController, "density").max(100.0).min(0.0).step(0.001).onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("DENSITY", value);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });


    var cameragui = gui.addFolder("Camera");
    cameragui.add(effectController, "exposure").max(1.0).min(0.0).step(0.001).onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("EXPOSURE", value);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });
    cameragui.add(effectController, "aperture").max(0.1).min(0.0).step(0.001).onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("APERTURE", value);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });
    cameragui.add(effectController, "focal_distance").max(5.0).min(0.1).step(0.001).onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("FOCALDIST", value);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });
    cameragui.add(effectController, "fov").max(90.0).min(0.0).step(0.001).onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("FOV_Y", value || 0.01);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });

    var clipping = gui.addFolder("Clipping Box");
    clipping.add(effectController, "xmin").max(1.0).min(0.0).step(0.001).onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("SET_CLIP_REGION", effectController.xmin, effectController.xmax, effectController.ymin, effectController.ymax, effectController.zmin, effectController.zmax);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });
    clipping.add(effectController, "xmax").max(1.0).min(0.0).step(0.001).onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("SET_CLIP_REGION", effectController.xmin, effectController.xmax, effectController.ymin, effectController.ymax, effectController.zmin, effectController.zmax);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });
    clipping.add(effectController, "ymin").max(1.0).min(0.0).step(0.001).onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("SET_CLIP_REGION", effectController.xmin, effectController.xmax, effectController.ymin, effectController.ymax, effectController.zmin, effectController.zmax);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });
    clipping.add(effectController, "ymax").max(1.0).min(0.0).step(0.001).onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("SET_CLIP_REGION", effectController.xmin, effectController.xmax, effectController.ymin, effectController.ymax, effectController.zmin, effectController.zmax);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });
    clipping.add(effectController, "zmin").max(1.0).min(0.0).step(0.001).onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("SET_CLIP_REGION", effectController.xmin, effectController.xmax, effectController.ymin, effectController.ymax, effectController.zmin, effectController.zmax);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });
    clipping.add(effectController, "zmax").max(1.0).min(0.0).step(0.001).onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("SET_CLIP_REGION", effectController.xmin, effectController.xmax, effectController.ymin, effectController.ymax, effectController.zmin, effectController.zmax);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });


    var lighting = gui.addFolder("Lighting");
    lighting.addColor(effectController, "skyTopColor").name("Sky Top").onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("SKYLIGHT_TOP_COLOR",
            effectController["skyTopIntensity"] * value[0] / 255.0,
            effectController["skyTopIntensity"] * value[1] / 255.0,
            effectController["skyTopIntensity"] * value[2] / 255.0);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });
    lighting.add(effectController, "skyTopIntensity").max(100.0).min(0.01).step(0.1).onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("SKYLIGHT_TOP_COLOR",
            effectController["skyTopColor"][0] / 255.0 * value,
            effectController["skyTopColor"][1] / 255.0 * value,
            effectController["skyTopColor"][2] / 255.0 * value);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });

    lighting.addColor(effectController, "skyMidColor").name("Sky Mid").onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("SKYLIGHT_MIDDLE_COLOR",
            effectController["skyMidIntensity"] * value[0] / 255.0,
            effectController["skyMidIntensity"] * value[1] / 255.0,
            effectController["skyMidIntensity"] * value[2] / 255.0);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });
    lighting.add(effectController, "skyMidIntensity").max(100.0).min(0.01).step(0.1).onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("SKYLIGHT_MIDDLE_COLOR",
            effectController["skyMidColor"][0] / 255.0 * value,
            effectController["skyMidColor"][1] / 255.0 * value,
            effectController["skyMidColor"][2] / 255.0 * value);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });
    lighting.addColor(effectController, "skyBotColor").name("Sky Bottom").onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("SKYLIGHT_BOTTOM_COLOR",
            effectController["skyBotIntensity"] * value[0] / 255.0,
            effectController["skyBotIntensity"] * value[1] / 255.0,
            effectController["skyBotIntensity"] * value[2] / 255.0);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });
    lighting.add(effectController, "skyBotIntensity").max(100.0).min(0.01).step(0.1).onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("SKYLIGHT_BOTTOM_COLOR",
            effectController["skyBotColor"][0] / 255.0 * value,
            effectController["skyBotColor"][1] / 255.0 * value,
            effectController["skyBotColor"][2] / 255.0 * value);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });
    lighting.add(effectController, "lightDistance").max(100.0).min(0.0).step(0.1).onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("LIGHT_POS", 0, value, effectController["lightTheta"] * 180.0 / 3.14159265, effectController["lightPhi"] * 180.0 / 3.14159265);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });
    lighting.add(effectController, "lightTheta").max(180.0).min(-180.0).step(1).onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("LIGHT_POS", 0, effectController["lightDistance"], value * 180.0 / 3.14159265, effectController["lightPhi"] * 180.0 / 3.14159265);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });
    lighting.add(effectController, "lightPhi").max(180.0).min(0.0).step(1).onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("LIGHT_POS", 0, effectController["lightDistance"], effectController["lightTheta"] * 180.0 / 3.14159265, value * 180.0 / 3.14159265);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });
    lighting.add(effectController, "lightSize").max(100.0).min(0.01).step(0.1).onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("LIGHT_SIZE", 0, value, value);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });
    lighting.add(effectController, "lightIntensity").max(100.0).min(0.01).step(0.1).onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("LIGHT_COLOR", 0, effectController["lightColor"][0] / 255.0 * value, effectController["lightColor"][1] / 255.0 * value, effectController["lightColor"][2] / 255.0 * value);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });
    lighting.addColor(effectController, "lightColor").name("lightcolor").onChange(function (value) {
        var cb = new commandBuffer();
        cb.addCommand("LIGHT_COLOR", 0, value[0] / 255.0 * effectController["lightIntensity"], value[1] / 255.0 * effectController["lightIntensity"], value[2] / 255.0 * effectController["lightIntensity"]);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function (value) {
        _stream_mode_suspended = false;
    });


    //  var customContainer = document.getElementById('my-gui-container');
    //  customContainer.appendChild(gui.domElement);
}

dat.GUI.prototype.removeFolder = function (name) {
    var folder = this.__folders[name];
    if (!folder) {
        return;
    }
    folder.close();
    this.__ul.removeChild(folder.domElement.parentNode);
    delete this.__folders[name];
    this.onResize();
}

function resetCamera() {
    // set up positions based on sizes.
    var x = effectController.infoObj.pixel_size_x * effectController.infoObj.x;
    var y = effectController.infoObj.pixel_size_y * effectController.infoObj.y;
    var z = effectController.infoObj.pixel_size_z * effectController.infoObj.z;
    var maxdim = Math.max(x, Math.max(y, z));
    const camdist = 1.5;
    gCamera.position.x = 0.5 * x / maxdim;
    gCamera.position.y = 0.5 * y / maxdim;
    gCamera.position.z = camdist + (0.5 * z / maxdim);
    gCamera.up.x = 0.0;
    gCamera.up.y = 1.0;
    gCamera.up.z = 0.0;
    gControls.target.x = 0.5 * x / maxdim;
    gControls.target.y = 0.5 * y / maxdim;
    gControls.target.z = 0.5 * z / maxdim;
    gControls.target0 = gControls.target.clone();
    effectController.focal_distance = camdist;
    sendCameraUpdate();
}

function onNewImage(infoObj) {
    effectController.infoObj = infoObj;
    resetCamera();
    setupChannelsGui();
    applyPresets();
}

function setupChannelsGui() {
    if (effectController && effectController.channelFolderNames) {
        for (var i = 0; i < effectController.channelFolderNames.length; ++i) {
            gui.removeFolder(effectController.channelFolderNames[i]);
        }
    }

    //var infoObj = effectController.infoObj;
    effectController.infoObj.channelGui = [];
    initcolors = [
        [255, 0, 255],
        [255, 255, 255],
        [0, 255, 255]
    ];
    effectController.channelFolderNames = []
    for (var i = 0; i < effectController.infoObj.c; ++i) {
        effectController.infoObj.channelGui.push({
            colorD: (i < 3) ? initcolors[i] : [255, 255, 255],
            colorS: [0, 0, 0],
            colorE: [0, 0, 0],
            window: 1.0,
            level: 0.5,
            roughness: 0.0,
            enabled: (i < 3) ? true : false
        });
        var f = gui.addFolder("Channel " + effectController.infoObj.channel_names[i]);
        effectController.channelFolderNames.push("Channel " + effectController.infoObj.channel_names[i]);
        f.add(effectController.infoObj.channelGui[i], "enabled").onChange(function (j) {
            return function (value) {
                var cb = new commandBuffer();
                cb.addCommand("ENABLE_CHANNEL", j, value ? 1 : 0);
                flushCommandBuffer(cb);
            };
        }(i));
        f.addColor(effectController.infoObj.channelGui[i], "colorD").name("Diffuse").onChange(function (j) {
            return function (value) {
                var cb = new commandBuffer();
                cb.addCommand("MAT_DIFFUSE", j, value[0] / 255.0, value[1] / 255.0, value[2] / 255.0, 1.0);
                flushCommandBuffer(cb);
            };
        }(i));
        f.addColor(effectController.infoObj.channelGui[i], "colorS").name("Specular").onChange(function (j) {
            return function (value) {
                var cb = new commandBuffer();
                cb.addCommand("MAT_SPECULAR", j, value[0] / 255.0, value[1] / 255.0, value[2] / 255.0, 1.0);
                flushCommandBuffer(cb);
            };
        }(i));
        f.addColor(effectController.infoObj.channelGui[i], "colorE").name("Emissive").onChange(function (j) {
            return function (value) {
                var cb = new commandBuffer();
                cb.addCommand("MAT_EMISSIVE", j, value[0] / 255.0, value[1] / 255.0, value[2] / 255.0, 1.0);
                flushCommandBuffer(cb);
            };
        }(i));
        f.add(effectController.infoObj.channelGui[i], "window").max(1.0).min(0.0).step(0.001).onChange(function (j) {
                return function (value) {
                    if (!waiting_for_image) {
                        var cb = new commandBuffer();
                        cb.addCommand("STREAM_MODE", 0);
                        cb.addCommand("SET_WINDOW_LEVEL", j, value, effectController.infoObj.channelGui[j].level);
                        flushCommandBuffer(cb);
                        waiting_for_image = true;
                    }
                    _stream_mode_suspended = true;
                }
            }(i))
            .onFinishChange(function (value) {
                var cb = new commandBuffer();
                cb.addCommand("STREAM_MODE", 1);
                flushCommandBuffer(cb);
                var cb2 = new commandBuffer();
                cb2.addCommand("REDRAW");
                flushCommandBuffer(cb2);
                _stream_mode_suspended = false;
            });

        f.add(effectController.infoObj.channelGui[i], "level").max(1.0).min(0.0).step(0.001).onChange(function (j) {
                return function (value) {
                    if (!waiting_for_image) {
                        var cb = new commandBuffer();
                        cb.addCommand("STREAM_MODE", 0);
                        cb.addCommand("SET_WINDOW_LEVEL", j, effectController.infoObj.channelGui[j].window, value);
                        flushCommandBuffer(cb);
                        waiting_for_image = true;
                    }
                    _stream_mode_suspended = true;
                }
            }(i))
            .onFinishChange(function (value) {
                var cb = new commandBuffer();
                cb.addCommand("STREAM_MODE", 1);
                flushCommandBuffer(cb);
                var cb2 = new commandBuffer();
                cb2.addCommand("REDRAW");
                flushCommandBuffer(cb2);
                _stream_mode_suspended = false;
            });
        f.add(effectController.infoObj.channelGui[i], "roughness").max(100.0).min(0.0).onChange(function (j) {
                return function (value) {
                    if (!waiting_for_image) {
                        var cb = new commandBuffer();
                        cb.addCommand("MAT_GLOSSINESS", j, value);
                        flushCommandBuffer(cb);
                        waiting_for_image = true;
                    }
                    _stream_mode_suspended = true;
                }
            }(i))
            .onFinishChange(function (value) {
                _stream_mode_suspended = false;
            });

    }

    var cb = new commandBuffer();
    for (var i = 0; i < effectController.infoObj.c; ++i) {
        var ch = effectController.infoObj.channelGui[i];
        cb.addCommand("ENABLE_CHANNEL", i, ch.enabled ? 1 : 0);
        cb.addCommand("MAT_DIFFUSE", i, ch.colorD[0] / 255.0, ch.colorD[1] / 255.0, ch.colorD[2] / 255.0, 1.0);
        cb.addCommand("MAT_SPECULAR", i, ch.colorS[0] / 255.0, ch.colorS[1] / 255.0, ch.colorS[2] / 255.0, 1.0);
        cb.addCommand("MAT_EMISSIVE", i, ch.colorE[0] / 255.0, ch.colorE[1] / 255.0, ch.colorE[2] / 255.0, 1.0);
        //cb.addCommand("SET_WINDOW_LEVEL", i, ch.window, ch.level);
    }
    flushCommandBuffer(cb);

}
/**
 *
 */
function init() {
    binarysocket0 = new WebSocket(wsUri); //handles requests for image streaming target #1
    binarysock = new binarysocket(0);
    binarysocket0.binaryType = "arraybuffer";
    //socket connection for image stream #1
    binarysocket0.onopen = binarysock.open;
    binarysocket0.onclose = binarysock.close;
    binarysocket0.onmessage = binarysock.message0; //linked to message0
    binarysocket0.onerror = binarysock.error;

    //    jsonsocket0 = new WebSocket(wsUri); //handles requests for image streaming target #1
    //    jsonsock = new jsonsocket();
    //    jsonsocket0.binaryType = "arraybuffer";
    //socket connection for json message requests
    //    jsonsocket0.onopen = jsonsock.open;
    //    jsonsocket0.onclose = jsonsock.close;
    //    jsonsocket0.onmessage = jsonsock.message;
    //    jsonsocket0.onerror = jsonsock.error;

    //setup tooltips
    //readTextFile("data/tooltip.csv");

    var streamedImg = document.getElementsByClassName("streamed_img");

    //set up first tab
    var streamimg1 = document.getElementById("imageA");

    toggleDivVisibility(streamimg1, true);

    setupGui();

    animate();
}

function animate() {
    requestAnimationFrame(animate);
    // look for new image to show
    if (enqueued_image_data) {
        binarysock.draw();
    }
}
/**
 * socket that exclusively receives binary data for streaming jpg images
 * @param channelnumber = 0 or 1 for left or right image => currently message0 or message1 are used since channelnumber cannot always be set via the constructor for some reason
 */
function binarysocket(channelnumber = 0) {
    this.channelnum = channelnumber;
    this.open = function (evt) {

        var cb = new commandBuffer();
        //cb.addCommand("LOAD_OME_TIF", effectController.file);
        cb.addCommand("SET_RESOLUTION", 512, 512);
        //cb.addCommand("FRAME_SCENE");
        cb.addCommand("APERTURE", effectController.aperture);
        cb.addCommand("EXPOSURE", effectController.exposure);
        cb.addCommand("SKYLIGHT_TOP_COLOR",
            effectController.skyTopIntensity * effectController.skyTopColor[0] / 255.0,
            effectController.skyTopIntensity * effectController.skyTopColor[1] / 255.0,
            effectController.skyTopIntensity * effectController.skyTopColor[2] / 255.0);
        cb.addCommand("SKYLIGHT_MIDDLE_COLOR",
            effectController.skyMidIntensity * effectController.skyMidColor[0] / 255.0,
            effectController.skyMidIntensity * effectController.skyMidColor[1] / 255.0,
            effectController.skyMidIntensity * effectController.skyMidColor[2] / 255.0);
        cb.addCommand("SKYLIGHT_BOTTOM_COLOR",
            effectController.skyBotIntensity * effectController.skyBotColor[0] / 255.0,
            effectController.skyBotIntensity * effectController.skyBotColor[1] / 255.0,
            effectController.skyBotIntensity * effectController.skyBotColor[2] / 255.0);
        cb.addCommand("LIGHT_POS", 0, effectController.lightDistance, effectController.lightTheta, effectController.lightPhi);
        cb.addCommand("LIGHT_COLOR", 0,
            effectController.lightColor[0] / 255.0 * effectController.lightIntensity,
            effectController.lightColor[1] / 255.0 * effectController.lightIntensity,
            effectController.lightColor[2] / 255.0 * effectController.lightIntensity,
        );
        cb.addCommand("LIGHT_SIZE", 0, effectController.lightSize, effectController.lightSize);
        cb.addCommand("STREAM_MODE", 1);
        flushCommandBuffer(cb);
        // var cb2 = new commandBuffer();
        // cb2.addCommand("REDRAW");
        // flushCommandBuffer(cb2);

        // init camera
        var streamimg1 = document.getElementById("imageA");
        gCamera = new THREE.PerspectiveCamera(55.0, 1.0, 0.001, 20);
        gCamera.position.x = 0.5;
        gCamera.position.y = 0.5 * 0.675;
        gCamera.position.z = 1.5 + (0.5 * 0.133);
        gCamera.up.x = 0.0;
        gCamera.up.y = 1.0;
        gCamera.up.z = 0.0;
        gControls = new AICStrackballControls(gCamera, streamimg1);
        gControls.target.x = 0.5;
        gControls.target.y = 0.5 * 0.675;
        gControls.target.z = 0.5 * 0.133;
        gControls.target0 = gControls.target.clone();
        gControls.rotateSpeed = 4.0 / window.devicePixelRatio;
        gControls.autoRotate = false;
        gControls.staticMoving = true;
        gControls.length = 10;
        gControls.enabled = true; //turn off mouse moments by setting to false

        gControls.addEventListener("change", sendCameraUpdate);
        gControls.addEventListener("start", function () {
            let cb = new commandBuffer();
            cb.addCommand("STREAM_MODE", 0);
            flushCommandBuffer(cb);
        });
        gControls.addEventListener("end", function () {
            let cb = new commandBuffer();
            cb.addCommand("STREAM_MODE", 1);
            flushCommandBuffer(cb);
            let cb1 = new commandBuffer();
            cb1.addCommand("REDRAW");
            flushCommandBuffer(cb1);
        });

    };
    this.close = function (evt) {
        setTimeout(function () {
            //window.location.href = 'index.html';
            console.warn("connection failed. refresh to retry.");
        }, 3000);
        //document.write('Socket disconnected. Restarting..');
    };
    this.message = function (evt) {
        var bytes = new Uint8Array(evt.data),
            binary = "",
            len = bytes.byteLength,
            i;

        for (i = 0; i < len; i++)
            binary += String.fromCharCode(bytes[i]);

        //console.log("msg received");
        screenImage.set(binary, this.channelnum);

    };

    this.message0 = function (evt) {

        if (typeof (evt.data) === "string") {
            var returnedObj = JSON.parse(evt.data);
            if (returnedObj.commandId === COMMANDS.LOAD_OME_TIF[0]) {
                console.log(returnedObj);
                // set up gui!
                onNewImage(returnedObj);
            }
            return;
        }


        // new data will be used to obliterate the previous data if it exists.
        // in this way, two consecutive images between redraws, will not both be drawn.
        // TODO:enqueue this...?
        enqueued_image_data = evt.data;

        // the this ptr is not what I want here.
        //binarysock.draw();

        if (!_stream_mode_suspended && _stream_mode && !dragFlag) {
            // let cb = new commandBuffer();
            // cb.addCommand("REDRAW");
            // flushCommandBuffer(cb);
        }

        // why should this code be slower?
        // var reader = new FileReader();
        // reader.onload = function(e) {
        //   screenImage.set(e.target.result, 0);
        //   console.timeEnd('recv');
        // };
        // reader.readAsDataURL(new Blob([new Uint8Array(evt.data)]));

    };

    this.draw = function () {
        //console.time('decode_img');
        var bytes = new Uint8Array(enqueued_image_data),
            binary = "",
            len = bytes.byteLength,
            i;
        for (i = 0; i < len; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        binary = window.btoa(binary);
        //console.timeEnd('decode_img');

        //console.time('set_img');
        screenImage.set("data:image/png;base64," + binary, 0);
        //console.timeEnd('set_img');  

        // nothing else to draw for now.
        enqueued_image_data = null;
        waiting_for_image = false;
    }
    this.error = function (evt) {
        console.log('error', evt);
    }
}
var lastevent;
var filestructure = {};


/**
 * socket that receives & handles json messages - used for setting up the client interface
 */
function jsonsocket() {

    this.open = function (evt) {
        //console.log("opening json socket");

    };

    this.close = function (evt) {
        //setTimeout(function () { window.location.href = 'index.html'; }, 3000);
        //document.write('Socket disconnected. Restarting..');
        console.log('json socket closed', evt);
    };

    this.message = function (evt) {
        lastevent = evt;

        //parse incoming json
        filestructure = JSON.parse(evt.data);
        jsonfilestruct = filestructure;
    };

    this.error = function (evt) {
        console.log('error', evt);
    }
}
//todo: test if function is deprecated

function send(msg) {}

var lastmsg;

function flushCommandBuffer(cmdbuf) {
    var buf = cmdbuf.prebufferToBuffer();
    binarysocket0.send(buf);
}

/**
 * calls the "init" method upon page load
 */
window.addEventListener("load", init, false);

function sendCameraUpdate() {
    if (!waiting_for_image) {
        cb = new commandBuffer();
        cb.addCommand("EYE", gCamera.position.x, gCamera.position.y, gCamera.position.z);
        cb.addCommand("TARGET", gControls.target.x, gControls.target.y, gControls.target.z);
        cb.addCommand("UP", gCamera.up.x, gCamera.up.y, gCamera.up.z);
        cb.addCommand("REDRAW");
        flushCommandBuffer(cb);
        waiting_for_image = true;
    }
}

/**
 * this object holds the image that is received from the server
 * @type {{set: screenImage.set}}
 */
var screenImage = {

    /**
     * sets the image and the events. called from the websocket "message" signal
     * @param binary
     * @param channelnumber
     */
    set: function (binary, channelnumber) {

        //get all the divs with the streamed_img tag and set the received binary data to the image's source
        var tabs;
        tabs = document.getElementsByClassName("streamed_img" + " img" + channelnumber);

        if (tabs.length > 0) {
            for (var i = 0; i < tabs.length; i++) {
                tabs[i].src = binary;

                img_width = tabs[i].width;
                img_height = tabs[i].height;
            }
        } else {
            console.warn("div 'streamed_img' not found :(");
        }
    }
};