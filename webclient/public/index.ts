import { PerspectiveCamera } from "three";

import AICSTrackballControls from "./AICStrackballControls.js";
import * as dat from "./dat.gui.min.js";
import { AgaveClient } from "../src";

//var wsUri = "ws://localhost:1235";
//var wsUri = "ws://dev-aics-dtp-001.corp.alleninstitute.org:1235";
const wsUri = "ws://ec2-54-245-184-76.us-west-2.compute.amazonaws.com:1235";
let agave: AgaveClient; // = new AgaveClient(wsUri);
let gCamera = new PerspectiveCamera(55.0, 1.0, 0.001, 20);
const streamimg1 = document.getElementById("imageA");
let gControls = new AICSTrackballControls(gCamera, streamimg1);

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
  element.style.visibility = visible ? "visible" : "hidden";
}

var gui;
interface ChannelGui {
  colorD: [number, number, number];
  colorS: [number, number, number];
  colorE: [number, number, number];
  window: number;
  level: number;
  roughness: number;
  enabled: boolean;
}
const effectController = {
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
  infoObj: {
    pixel_size_x: 1,
    pixel_size_y: 1,
    pixel_size_z: 1,
    z: 1,
    y: 1,
    x: 1,
    c: 1,
    channelGui: [] as ChannelGui[],
    channel_names: [] as string[],
  },
  channelFolderNames: [] as string[],
  resetCamera: resetCamera,
  preset0: function () {
    executePreset(0);
  },
  preset1: function () {
    executePreset(1);
  },
  preset2: function () {
    executePreset(2);
  },
};

function setupGui() {
  gui = new dat.GUI();
  //gui = new dat.GUI({autoPlace:false, width:200});

  gui.add(effectController, "preset0").name("Tubulin");
  gui.add(effectController, "preset1").name("TOM20");
  gui.add(effectController, "preset2").name("LaminB");

  // gui.add(effectController, "file").onFinishChange(function (value) {
  //     var cb = new commandBuffer();
  //     cb.addCommand("LOAD_OME_TIF", value);
  //     flushCommandBuffer(cb);
  //     _stream_mode_suspended = true;
  // });

  gui
    .add(effectController, "resolution", [
      "256x256",
      "512x512",
      "1024x1024",
      "1024x768",
    ])
    .onChange(function (value) {
      var res = value.match(/(\d+)x(\d+)/);
      if (res.length === 3) {
        res[0] = parseInt(res[1]);
        res[1] = parseInt(res[2]);
        var imgholder = document.getElementById("imageA") as HTMLImageElement;
        imgholder!.width = res[0];
        imgholder!.height = res[1];
        imgholder!.style.width = res[0];
        imgholder!.style.height = res[1];

        agave.set_resolution(res[0], res[1]);
        agave.flushCommandBuffer();
      }
    });

  gui.add(effectController, "resetCamera");
  //allen/aics/animated-cell/Allen-Cell-Explorer/Allen-Cell-Explorer_1.2.0/Cell-Viewer_Data/2017_05_15_tubulin/AICS-12/AICS-12_790.ome.tif
  gui.add(effectController, "stream").onChange(function (value) {
    agave.stream_mode(value);
    agave.flushCommandBuffer();
    // BUG THIS SHOULD NOT BE NEEDED.
    agave.redraw();
    agave.flushCommandBuffer();
    _stream_mode = value;
  });
  gui
    .add(effectController, "density")
    .max(100.0)
    .min(0.0)
    .step(0.001)
    .onChange(function (value) {
      agave.density(value);
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
      _stream_mode_suspended = false;
    });

  var cameragui = gui.addFolder("Camera");
  cameragui
    .add(effectController, "exposure")
    .max(1.0)
    .min(0.0)
    .step(0.001)
    .onChange(function (value) {
      agave.exposure(value);
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
      _stream_mode_suspended = false;
    });
  cameragui
    .add(effectController, "aperture")
    .max(0.1)
    .min(0.0)
    .step(0.001)
    .onChange(function (value) {
      agave.aperture(value);
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
      _stream_mode_suspended = false;
    });
  cameragui
    .add(effectController, "focal_distance")
    .max(5.0)
    .min(0.1)
    .step(0.001)
    .onChange(function (value) {
      agave.focaldist(value);
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
      _stream_mode_suspended = false;
    });
  cameragui
    .add(effectController, "fov")
    .max(90.0)
    .min(0.0)
    .step(0.001)
    .onChange(function (value) {
      agave.camera_projection(0, value || 0.01);
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
      _stream_mode_suspended = false;
    });

  var clipping = gui.addFolder("Clipping Box");
  clipping
    .add(effectController, "xmin")
    .max(1.0)
    .min(0.0)
    .step(0.001)
    .onChange(function (value) {
      agave.set_clip_region(
        effectController.xmin,
        effectController.xmax,
        effectController.ymin,
        effectController.ymax,
        effectController.zmin,
        effectController.zmax
      );
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
      _stream_mode_suspended = false;
    });
  clipping
    .add(effectController, "xmax")
    .max(1.0)
    .min(0.0)
    .step(0.001)
    .onChange(function (value) {
      agave.set_clip_region(
        effectController.xmin,
        effectController.xmax,
        effectController.ymin,
        effectController.ymax,
        effectController.zmin,
        effectController.zmax
      );
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
      _stream_mode_suspended = false;
    });
  clipping
    .add(effectController, "ymin")
    .max(1.0)
    .min(0.0)
    .step(0.001)
    .onChange(function (value) {
      agave.set_clip_region(
        effectController.xmin,
        effectController.xmax,
        effectController.ymin,
        effectController.ymax,
        effectController.zmin,
        effectController.zmax
      );
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
      _stream_mode_suspended = false;
    });
  clipping
    .add(effectController, "ymax")
    .max(1.0)
    .min(0.0)
    .step(0.001)
    .onChange(function (value) {
      agave.set_clip_region(
        effectController.xmin,
        effectController.xmax,
        effectController.ymin,
        effectController.ymax,
        effectController.zmin,
        effectController.zmax
      );
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
      _stream_mode_suspended = false;
    });
  clipping
    .add(effectController, "zmin")
    .max(1.0)
    .min(0.0)
    .step(0.001)
    .onChange(function (value) {
      agave.set_clip_region(
        effectController.xmin,
        effectController.xmax,
        effectController.ymin,
        effectController.ymax,
        effectController.zmin,
        effectController.zmax
      );
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
      _stream_mode_suspended = false;
    });
  clipping
    .add(effectController, "zmax")
    .max(1.0)
    .min(0.0)
    .step(0.001)
    .onChange(function (value) {
      agave.set_clip_region(
        effectController.xmin,
        effectController.xmax,
        effectController.ymin,
        effectController.ymax,
        effectController.zmin,
        effectController.zmax
      );
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
      _stream_mode_suspended = false;
    });

  var lighting = gui.addFolder("Lighting");
  lighting
    .addColor(effectController, "skyTopColor")
    .name("Sky Top")
    .onChange(function (value) {
      agave.skylight_top_color(
        (effectController["skyTopIntensity"] * value[0]) / 255.0,
        (effectController["skyTopIntensity"] * value[1]) / 255.0,
        (effectController["skyTopIntensity"] * value[2]) / 255.0
      );
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
      _stream_mode_suspended = false;
    });
  lighting
    .add(effectController, "skyTopIntensity")
    .max(100.0)
    .min(0.01)
    .step(0.1)
    .onChange(function (value) {
      agave.skylight_top_color(
        (effectController["skyTopColor"][0] / 255.0) * value,
        (effectController["skyTopColor"][1] / 255.0) * value,
        (effectController["skyTopColor"][2] / 255.0) * value
      );
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
      _stream_mode_suspended = false;
    });

  lighting
    .addColor(effectController, "skyMidColor")
    .name("Sky Mid")
    .onChange(function (value) {
      agave.skylight_middle_color(
        (effectController["skyMidIntensity"] * value[0]) / 255.0,
        (effectController["skyMidIntensity"] * value[1]) / 255.0,
        (effectController["skyMidIntensity"] * value[2]) / 255.0
      );
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
      _stream_mode_suspended = false;
    });
  lighting
    .add(effectController, "skyMidIntensity")
    .max(100.0)
    .min(0.01)
    .step(0.1)
    .onChange(function (value) {
      agave.skylight_middle_color(
        (effectController["skyMidColor"][0] / 255.0) * value,
        (effectController["skyMidColor"][1] / 255.0) * value,
        (effectController["skyMidColor"][2] / 255.0) * value
      );
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
      _stream_mode_suspended = false;
    });
  lighting
    .addColor(effectController, "skyBotColor")
    .name("Sky Bottom")
    .onChange(function (value) {
      agave.skylight_bottom_color(
        (effectController["skyBotIntensity"] * value[0]) / 255.0,
        (effectController["skyBotIntensity"] * value[1]) / 255.0,
        (effectController["skyBotIntensity"] * value[2]) / 255.0
      );
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
      _stream_mode_suspended = false;
    });
  lighting
    .add(effectController, "skyBotIntensity")
    .max(100.0)
    .min(0.01)
    .step(0.1)
    .onChange(function (value) {
      agave.skylight_bottom_color(
        (effectController["skyBotColor"][0] / 255.0) * value,
        (effectController["skyBotColor"][1] / 255.0) * value,
        (effectController["skyBotColor"][2] / 255.0) * value
      );
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
      _stream_mode_suspended = false;
    });
  lighting
    .add(effectController, "lightDistance")
    .max(100.0)
    .min(0.0)
    .step(0.1)
    .onChange(function (value) {
      agave.light_pos(
        0,
        value,
        (effectController["lightTheta"] * 180.0) / 3.14159265,
        (effectController["lightPhi"] * 180.0) / 3.14159265
      );
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
      _stream_mode_suspended = false;
    });
  lighting
    .add(effectController, "lightTheta")
    .max(180.0)
    .min(-180.0)
    .step(1)
    .onChange(function (value) {
      agave.light_pos(
        0,
        effectController["lightDistance"],
        (value * 180.0) / 3.14159265,
        (effectController["lightPhi"] * 180.0) / 3.14159265
      );
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
      _stream_mode_suspended = false;
    });
  lighting
    .add(effectController, "lightPhi")
    .max(180.0)
    .min(0.0)
    .step(1)
    .onChange(function (value) {
      agave.light_pos(
        0,
        effectController["lightDistance"],
        (effectController["lightTheta"] * 180.0) / 3.14159265,
        (value * 180.0) / 3.14159265
      );
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
      _stream_mode_suspended = false;
    });
  lighting
    .add(effectController, "lightSize")
    .max(100.0)
    .min(0.01)
    .step(0.1)
    .onChange(function (value) {
      agave.light_size(0, value, value);
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
      _stream_mode_suspended = false;
    });
  lighting
    .add(effectController, "lightIntensity")
    .max(100.0)
    .min(0.01)
    .step(0.1)
    .onChange(function (value) {
      agave.light_color(
        0,
        (effectController["lightColor"][0] / 255.0) * value,
        (effectController["lightColor"][1] / 255.0) * value,
        (effectController["lightColor"][2] / 255.0) * value
      );
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
      _stream_mode_suspended = false;
    });
  lighting
    .addColor(effectController, "lightColor")
    .name("lightcolor")
    .onChange(function (value) {
      agave.light_color(
        0,
        (value[0] / 255.0) * effectController["lightIntensity"],
        (value[1] / 255.0) * effectController["lightIntensity"],
        (value[2] / 255.0) * effectController["lightIntensity"]
      );
      agave.flushCommandBuffer();
      _stream_mode_suspended = true;
    })
    .onFinishChange(function (value) {
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
};

function resetCamera() {
  // set up positions based on sizes.
  var x = effectController.infoObj.pixel_size_x * effectController.infoObj.x;
  var y = effectController.infoObj.pixel_size_y * effectController.infoObj.y;
  var z = effectController.infoObj.pixel_size_z * effectController.infoObj.z;
  var maxdim = Math.max(x, Math.max(y, z));
  const camdist = 1.5;
  gCamera.position.x = (0.5 * x) / maxdim;
  gCamera.position.y = (0.5 * y) / maxdim;
  gCamera.position.z = camdist + (0.5 * z) / maxdim;
  gCamera.up.x = 0.0;
  gCamera.up.y = 1.0;
  gCamera.up.z = 0.0;
  gControls.target.x = (0.5 * x) / maxdim;
  gControls.target.y = (0.5 * y) / maxdim;
  gControls.target.z = (0.5 * z) / maxdim;
  gControls.target0 = gControls.target.clone();
  effectController.focal_distance = camdist;
  sendCameraUpdate();
}

/**
 * this object holds the image that is received from the server
 * @type {{set: screenImage.set}}
 */
const screenImage = {
  /**
   * sets the image and the events. called from the websocket "message" signal
   * @param binary
   */
  set: function (binary) {
    //get all the divs with the streamed_img tag and set the received binary data to the image's source
    var tabs;
    tabs = document.getElementsByClassName("streamed_img img0");

    if (tabs.length > 0) {
      for (var i = 0; i < tabs.length; i++) {
        tabs[i].src = binary;
      }
    } else {
      console.warn("div 'streamed_img' not found :(");
    }
  },
};

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
  const initcolors: [number, number, number][] = [
    [255, 0, 255],
    [255, 255, 255],
    [0, 255, 255],
  ];
  effectController.channelFolderNames = [];
  for (var i = 0; i < effectController.infoObj.c; ++i) {
    effectController.infoObj.channelGui.push({
      colorD: i < 3 ? initcolors[i] : [255, 255, 255],
      colorS: [0, 0, 0],
      colorE: [0, 0, 0],
      window: 1.0,
      level: 0.5,
      roughness: 0.0,
      enabled: i < 3 ? true : false,
    });
    var f = gui.addFolder(
      "Channel " + effectController.infoObj.channel_names[i]
    );
    effectController.channelFolderNames.push(
      "Channel " + effectController.infoObj.channel_names[i]
    );
    f.add(effectController.infoObj.channelGui[i], "enabled").onChange(
      (function (j) {
        return function (value) {
          agave.enable_channel(j, value ? 1 : 0);
          agave.flushCommandBuffer();
        };
      })(i)
    );
    f.addColor(effectController.infoObj.channelGui[i], "colorD")
      .name("Diffuse")
      .onChange(
        (function (j) {
          return function (value) {
            agave.mat_diffuse(
              j,
              value[0] / 255.0,
              value[1] / 255.0,
              value[2] / 255.0,
              1.0
            );
            agave.flushCommandBuffer();
          };
        })(i)
      );
    f.addColor(effectController.infoObj.channelGui[i], "colorS")
      .name("Specular")
      .onChange(
        (function (j) {
          return function (value) {
            agave.mat_specular(
              j,
              value[0] / 255.0,
              value[1] / 255.0,
              value[2] / 255.0,
              1.0
            );
            agave.flushCommandBuffer();
          };
        })(i)
      );
    f.addColor(effectController.infoObj.channelGui[i], "colorE")
      .name("Emissive")
      .onChange(
        (function (j) {
          return function (value) {
            agave.mat_emissive(
              j,
              value[0] / 255.0,
              value[1] / 255.0,
              value[2] / 255.0,
              1.0
            );
            agave.flushCommandBuffer();
          };
        })(i)
      );
    f.add(effectController.infoObj.channelGui[i], "window")
      .max(1.0)
      .min(0.0)
      .step(0.001)
      .onChange(
        (function (j) {
          return function (value) {
            if (!waiting_for_image) {
              agave.stream_mode(0);
              agave.set_window_level(
                j,
                value,
                effectController.infoObj.channelGui[j].level
              );
              agave.flushCommandBuffer();
              waiting_for_image = true;
            }
            _stream_mode_suspended = true;
          };
        })(i)
      )
      .onFinishChange(function (value) {
        agave.stream_mode(1);
        agave.flushCommandBuffer();
        agave.redraw();
        agave.flushCommandBuffer();
        _stream_mode_suspended = false;
      });

    f.add(effectController.infoObj.channelGui[i], "level")
      .max(1.0)
      .min(0.0)
      .step(0.001)
      .onChange(
        (function (j) {
          return function (value) {
            if (!waiting_for_image) {
              agave.stream_mode(0);
              agave.set_window_level(
                j,
                effectController.infoObj.channelGui[j].window,
                value
              );
              agave.flushCommandBuffer();
              waiting_for_image = true;
            }
            _stream_mode_suspended = true;
          };
        })(i)
      )
      .onFinishChange(function (value) {
        agave.stream_mode(1);
        agave.flushCommandBuffer();
        agave.redraw();
        agave.flushCommandBuffer();
        _stream_mode_suspended = false;
      });
    f.add(effectController.infoObj.channelGui[i], "roughness")
      .max(100.0)
      .min(0.0)
      .onChange(
        (function (j) {
          return function (value) {
            if (!waiting_for_image) {
              agave.mat_glossiness(j, value);
              agave.flushCommandBuffer();
              waiting_for_image = true;
            }
            _stream_mode_suspended = true;
          };
        })(i)
      )
      .onFinishChange(function (value) {
        _stream_mode_suspended = false;
      });
  }

  for (var i = 0; i < effectController.infoObj.c; ++i) {
    var ch = effectController.infoObj.channelGui[i];
    agave.enable_channel(i, ch.enabled ? 1 : 0);
    agave.mat_diffuse(
      i,
      ch.colorD[0] / 255.0,
      ch.colorD[1] / 255.0,
      ch.colorD[2] / 255.0,
      1.0
    );
    agave.mat_specular(
      i,
      ch.colorS[0] / 255.0,
      ch.colorS[1] / 255.0,
      ch.colorS[2] / 255.0,
      1.0
    );
    agave.mat_emissive(
      i,
      ch.colorE[0] / 255.0,
      ch.colorE[1] / 255.0,
      ch.colorE[2] / 255.0,
      1.0
    );
    //cb.addCommand("SET_WINDOW_LEVEL", i, ch.window, ch.level);
  }
  agave.flushCommandBuffer();
}

binarysock = new binarysocket();

// should this be a promise that runs after async init of agave client?
function onConnectionOpened() {
  binarysock.open();
}

function onConnectionClosed() {
  console.log("connection closed");
  binarysock.close();
}

function onImageReceived(imgdata) {
  // enqueue until redraw loop can pick it up?
  agave.enqueued_image_data = imgdata;
  agave.waiting_for_image = false;
}

function onJsonReceived(obj) {
  onNewImage(obj);
}

function init() {
  agave = new AgaveClient(
    wsUri,
    onConnectionOpened,
    onImageReceived,
    onJsonReceived,
    onConnectionClosed
  );

  toggleDivVisibility(streamimg1, true);

  setupGui();

  animate();
}

function animate() {
  requestAnimationFrame(animate);
  // look for new image to show
  if (agave.enqueued_image_data) {
    screenImage.set("data:image/png;base64," + agave.enqueued_image_data, 0);
    // nothing else to draw for now.
    agave.enqueued_image_data = "";
    agave.waiting_for_image = false;
  }
}
/**
 * socket that exclusively receives binary data for streaming jpg images
 */
function binarysocket() {
  this.open = function (evt) {
    agave.stream_mode(1);
    agave.set_resolution(512, 512);
    agave.aperture(effectController.aperture);
    agave.exposure(effectController.exposure);
    agave.sky_top_color(
      (effectController.skyTopIntensity * effectController.skyTopColor[0]) /
        255.0,
      (effectController.skyTopIntensity * effectController.skyTopColor[1]) /
        255.0,
      (effectController.skyTopIntensity * effectController.skyTopColor[2]) /
        255.0
    );
    agave.sky_middle_color(
      (effectController.skyMidIntensity * effectController.skyMidColor[0]) /
        255.0,
      (effectController.skyMidIntensity * effectController.skyMidColor[1]) /
        255.0,
      (effectController.skyMidIntensity * effectController.skyMidColor[2]) /
        255.0
    );
    agave.sky_bot_color(
      (effectController.skyBotIntensity * effectController.skyBotColor[0]) /
        255.0,
      (effectController.skyBotIntensity * effectController.skyBotColor[1]) /
        255.0,
      (effectController.skyBotIntensity * effectController.skyBotColor[2]) /
        255.0
    );
    agave.light_pos(
      0,
      effectController.lightDistance,
      effectController.lightTheta,
      effectController.lightPhi
    );
    agave.light_color(
      0,
      (effectController.lightColor[0] / 255.0) *
        effectController.lightIntensity,
      (effectController.lightColor[1] / 255.0) *
        effectController.lightIntensity,
      (effectController.lightColor[2] / 255.0) * effectController.lightIntensity
    );
    agave.light_size(0, effectController.lightSize, effectController.lightSize);
    agave.stream_mode(1);
    agave.flushCommandBuffer();

    // init camera
    var streamimg1 = document.getElementById("imageA");
    gCamera = new PerspectiveCamera(55.0, 1.0, 0.001, 20);
    gCamera.position.x = 0.5;
    gCamera.position.y = 0.5 * 0.675;
    gCamera.position.z = 1.5 + 0.5 * 0.133;
    gCamera.up.x = 0.0;
    gCamera.up.y = 1.0;
    gCamera.up.z = 0.0;
    gControls = new AICSTrackballControls(gCamera, streamimg1);
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
      agave.stream_mode(0);
      agave.flushCommandBuffer();
    });
    gControls.addEventListener("end", function () {
      agave.stream_mode(1);
      agave.flushCommandBuffer();
      agave.redraw();
      agave.flushCommandBuffer();
    });
  };
  this.close = function (evt) {
    setTimeout(function () {
      //window.location.href = 'index.html';
      console.warn("connection failed. refresh to retry.");
    }, 3000);
    //document.write('Socket disconnected. Restarting..');
  };
  this.message0 = function (evt) {
    if (typeof evt.data === "string") {
      var returnedObj = JSON.parse(evt.data);
      if (returnedObj.commandId === COMMANDS.LOAD_DATA[0]) {
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

    var bytes = new Uint8Array(enqueued_image_data),
      binary = "",
      len = bytes.byteLength,
      i;
    for (i = 0; i < len; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    enqueued_image_data = window.btoa(binary);

    // the this ptr is not what I want here.
    //binarysock.draw();

    if (!_stream_mode_suspended && _stream_mode) {
      // agave.redraw();
      // agave.flushCommandBuffer();
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
    //console.timeEnd('decode_img');

    //console.time('set_img');
    screenImage.set("data:image/png;base64," + enqueued_image_data, 0);
    //console.timeEnd('set_img');

    // nothing else to draw for now.
    enqueued_image_data = null;
    waiting_for_image = false;
  };
  this.error = function (evt) {
    console.log("error", evt);
  };
}

/**
 * calls the "init" method upon page load
 */
window.addEventListener("load", init, false);

function sendCameraUpdate() {
  if (!waiting_for_image) {
    agave.eye(gCamera.position.x, gCamera.position.y, gCamera.position.z);
    agave.target(gControls.target.x, gControls.target.y, gControls.target.z);
    agave.up(gCamera.up.x, gCamera.up.y, gCamera.up.z);
    agave.redraw();
    agave.flushCommandBuffer();
    waiting_for_image = true;
  }
}
