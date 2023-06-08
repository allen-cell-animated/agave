import { PerspectiveCamera } from "three";
import { TrackballControls } from "three/examples/jsm/controls/TrackballControls.js";

import * as dat from "dat.gui";
import { AgaveClient } from "../src";

function arrayBufferToImage(arraybuf) {
  const bytes = new Uint8Array(arraybuf);
  let binary = "";
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return window.btoa(binary);
}

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
  channelFolders: [] as dat.GUI[],
  resetCamera: () => {},
};

class AgaveApp {
  private agave: AgaveClient;
  private camera: PerspectiveCamera;
  private controls: TrackballControls;
  private streamimg1: HTMLImageElement;
  public enqueued_image_data: Blob | string | null;
  public waiting_for_image: boolean;
  private effectController: typeof effectController;
  private _stream_mode: boolean;
  private _stream_mode_suspended: boolean;
  private gui: dat.GUI;
  private nredraws: number = 0;
  private nwsmsgs: number = 0;
  private lastRedrawTime: number = 0;
  private lastWsMsgTime: number = 0;

  constructor() {
    this.gui = new dat.GUI();
    this._stream_mode = false;
    this._stream_mode_suspended = false;
    this.effectController = effectController;
    this.enqueued_image_data = null;
    this.waiting_for_image = false;
    const wsUri = "ws://localhost:1235";
    //const wsUri = "ws://dev-aics-dtp-001.corp.alleninstitute.org:1235";
    //const wsUri = "ws://ec2-54-245-184-76.us-west-2.compute.amazonaws.com:1235";
    this.agave = new AgaveClient(
      wsUri,
      "pathtrace",
      this._onConnectionOpened.bind(this),
      this._onJsonReceived.bind(this),
      this._onImageReceived.bind(this)
    );

    this.camera = new PerspectiveCamera(55.0, 1.0, 0.001, 20);
    this.streamimg1 = document.getElementById("imageA") as HTMLImageElement;
    this.controls = new TrackballControls(this.camera, this.streamimg1!);

    this._setupGui();
  }

  private _onImageReceived(imgdata) {
    // new data will be used to obliterate the previous data if it exists.
    // in this way, two consecutive images between redraws, will not both be drawn.
    // enqueue until redraw loop can pick it up?

    this.enqueued_image_data = imgdata;
    this.waiting_for_image = false;
    this.nwsmsgs++;

    const t = performance.now();
    const dt = t - this.lastWsMsgTime;
    this.lastWsMsgTime = t;
    console.log("MESSAGE TIME " + dt.toFixed(2) + " ms");
  }

  private _onJsonReceived(jsondata) {
    this.effectController.infoObj = jsondata;
    this._resetCamera();
    this._setupChannelsGui();
  }

  private _onConnectionOpened() {
    const agave = this.agave;
    const effectController = this.effectController;
    agave.load_data(
      "https://animatedcell-test-data.s3.us-west-2.amazonaws.com/variance/1.zarr",
      0,
      2,
      0,
      [],
      []
    );
    agave.stream_mode(1);
    agave.set_resolution(512, 512);
    agave.aperture(effectController.aperture);
    agave.exposure(effectController.exposure);
    agave.skylight_top_color(
      (effectController.skyTopIntensity * effectController.skyTopColor[0]) /
        255.0,
      (effectController.skyTopIntensity * effectController.skyTopColor[1]) /
        255.0,
      (effectController.skyTopIntensity * effectController.skyTopColor[2]) /
        255.0
    );
    agave.skylight_middle_color(
      (effectController.skyMidIntensity * effectController.skyMidColor[0]) /
        255.0,
      (effectController.skyMidIntensity * effectController.skyMidColor[1]) /
        255.0,
      (effectController.skyMidIntensity * effectController.skyMidColor[2]) /
        255.0
    );
    agave.skylight_bottom_color(
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
    this.camera = new PerspectiveCamera(55.0, 1.0, 0.001, 20);
    this.camera.position.x = 0.5;
    this.camera.position.y = 0.5 * 0.675;
    this.camera.position.z = 1.5 + 0.5 * 0.133;
    this.camera.up.x = 0.0;
    this.camera.up.y = 1.0;
    this.camera.up.z = 0.0;
    this.controls = new TrackballControls(this.camera, this.streamimg1!);
    this.controls.target.x = 0.5;
    this.controls.target.y = 0.5 * 0.675;
    this.controls.target.z = 0.5 * 0.133;
    this.controls.target0 = this.controls.target.clone();
    this.controls.rotateSpeed = 4.0 / window.devicePixelRatio;
    this.controls.staticMoving = true;
    this.controls.enabled = true; //turn off mouse moments by setting to false

    this.controls.addEventListener("change", () => {
      this._sendCameraUpdate();
    });
    this.controls.addEventListener("start", function () {
      agave.stream_mode(0);
      agave.flushCommandBuffer();
    });
    this.controls.addEventListener("end", function () {
      agave.stream_mode(1);
      agave.flushCommandBuffer();
      agave.redraw();
      agave.flushCommandBuffer();
    });
  }

  redraw() {
    this.controls.update();
    // look for new image to show
    if (this.enqueued_image_data) {
      // blob mode
      const dataurl = URL.createObjectURL(this.enqueued_image_data as Blob);
      // arraybuffer mode
      //const dataurl = "data:image/png;base64," + arrayBufferToImage(this.enqueued_image_data);
      this.streamimg1.src = dataurl;
      // nothing else to draw for now.
      this.enqueued_image_data = null;
      this.waiting_for_image = false;

      const t = performance.now();
      const dt = t - this.lastRedrawTime;
      this.lastRedrawTime = t;
      console.log("REDRAW TIME " + dt.toFixed(2) + " ms");
      this.nredraws++;
      // console.log("REDRAWS " + this.nredraws);
      // console.log("WS MESSAGES " + this.nwsmsgs);
      // this.nredraws = 0;
      // this.nwsmsgs = 0;
    }
  }

  private _resetCamera() {
    const effectController = this.effectController;
    // set up positions based on sizes.
    var x = effectController.infoObj.pixel_size_x * effectController.infoObj.x;
    var y = effectController.infoObj.pixel_size_y * effectController.infoObj.y;
    var z = effectController.infoObj.pixel_size_z * effectController.infoObj.z;
    var maxdim = Math.max(x, Math.max(y, z));
    const camdist = 1.5;
    this.camera.position.x = (0.5 * x) / maxdim;
    this.camera.position.y = (0.5 * y) / maxdim;
    this.camera.position.z = camdist + (0.5 * z) / maxdim;
    this.camera.up.x = 0.0;
    this.camera.up.y = 1.0;
    this.camera.up.z = 0.0;
    this.controls.target.x = (0.5 * x) / maxdim;
    this.controls.target.y = (0.5 * y) / maxdim;
    this.controls.target.z = (0.5 * z) / maxdim;
    this.controls.target0 = this.controls.target.clone();
    effectController.focal_distance = camdist;
    this._sendCameraUpdate();
  }

  private _sendCameraUpdate() {
    if (!this.waiting_for_image) {
      const agave = this.agave;
      agave.eye(
        this.camera.position.x,
        this.camera.position.y,
        this.camera.position.z
      );
      agave.target(
        this.controls.target.x,
        this.controls.target.y,
        this.controls.target.z
      );
      agave.up(this.camera.up.x, this.camera.up.y, this.camera.up.z);
      agave.redraw();
      agave.flushCommandBuffer();
      this.waiting_for_image = true;
    }
  }

  private _setupGui() {
    this.gui = new dat.GUI();
    const gui = this.gui;
    const agave = this.agave;
    gui
      .add(effectController, "resolution", [
        "256x256",
        "512x512",
        "1024x1024",
        "1024x768",
      ])
      .onChange((value) => {
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

    effectController.resetCamera = () => {
      this._resetCamera();
    };
    gui.add(effectController, "resetCamera");
    //allen/aics/animated-cell/Allen-Cell-Explorer/Allen-Cell-Explorer_1.2.0/Cell-Viewer_Data/2017_05_15_tubulin/AICS-12/AICS-12_790.ome.tif
    gui.add(effectController, "stream").onChange((value) => {
      agave.stream_mode(value);
      agave.flushCommandBuffer();
      // BUG THIS SHOULD NOT BE NEEDED.
      agave.redraw();
      agave.flushCommandBuffer();
      this._stream_mode = value;
    });
    gui
      .add(effectController, "density")
      .max(100.0)
      .min(0.0)
      .step(0.001)
      .onChange((value) => {
        agave.density(value);
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });

    var cameragui = gui.addFolder("Camera");
    cameragui
      .add(effectController, "exposure")
      .max(1.0)
      .min(0.0)
      .step(0.001)
      .onChange((value) => {
        agave.exposure(value);
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });
    cameragui
      .add(effectController, "aperture")
      .max(0.1)
      .min(0.0)
      .step(0.001)
      .onChange((value) => {
        agave.aperture(value);
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });
    cameragui
      .add(effectController, "focal_distance")
      .max(5.0)
      .min(0.1)
      .step(0.001)
      .onChange((value) => {
        agave.focaldist(value);
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });
    cameragui
      .add(effectController, "fov")
      .max(90.0)
      .min(0.0)
      .step(0.001)
      .onChange((value) => {
        agave.camera_projection(0, value || 0.01);
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });

    var clipping = gui.addFolder("Clipping Box");
    clipping
      .add(effectController, "xmin")
      .max(1.0)
      .min(0.0)
      .step(0.001)
      .onChange((value) => {
        agave.set_clip_region(
          effectController.xmin,
          effectController.xmax,
          effectController.ymin,
          effectController.ymax,
          effectController.zmin,
          effectController.zmax
        );
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });
    clipping
      .add(effectController, "xmax")
      .max(1.0)
      .min(0.0)
      .step(0.001)
      .onChange((value) => {
        agave.set_clip_region(
          effectController.xmin,
          effectController.xmax,
          effectController.ymin,
          effectController.ymax,
          effectController.zmin,
          effectController.zmax
        );
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });
    clipping
      .add(effectController, "ymin")
      .max(1.0)
      .min(0.0)
      .step(0.001)
      .onChange((value) => {
        agave.set_clip_region(
          effectController.xmin,
          effectController.xmax,
          effectController.ymin,
          effectController.ymax,
          effectController.zmin,
          effectController.zmax
        );
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });
    clipping
      .add(effectController, "ymax")
      .max(1.0)
      .min(0.0)
      .step(0.001)
      .onChange((value) => {
        agave.set_clip_region(
          effectController.xmin,
          effectController.xmax,
          effectController.ymin,
          effectController.ymax,
          effectController.zmin,
          effectController.zmax
        );
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });
    clipping
      .add(effectController, "zmin")
      .max(1.0)
      .min(0.0)
      .step(0.001)
      .onChange((value) => {
        agave.set_clip_region(
          effectController.xmin,
          effectController.xmax,
          effectController.ymin,
          effectController.ymax,
          effectController.zmin,
          effectController.zmax
        );
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });
    clipping
      .add(effectController, "zmax")
      .max(1.0)
      .min(0.0)
      .step(0.001)
      .onChange((value) => {
        agave.set_clip_region(
          effectController.xmin,
          effectController.xmax,
          effectController.ymin,
          effectController.ymax,
          effectController.zmin,
          effectController.zmax
        );
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });

    var lighting = gui.addFolder("Lighting");
    lighting
      .addColor(effectController, "skyTopColor")
      .name("Sky Top")
      .onChange((value) => {
        agave.skylight_top_color(
          (effectController["skyTopIntensity"] * value[0]) / 255.0,
          (effectController["skyTopIntensity"] * value[1]) / 255.0,
          (effectController["skyTopIntensity"] * value[2]) / 255.0
        );
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });
    lighting
      .add(effectController, "skyTopIntensity")
      .max(100.0)
      .min(0.01)
      .step(0.1)
      .onChange((value) => {
        agave.skylight_top_color(
          (effectController["skyTopColor"][0] / 255.0) * value,
          (effectController["skyTopColor"][1] / 255.0) * value,
          (effectController["skyTopColor"][2] / 255.0) * value
        );
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });

    lighting
      .addColor(effectController, "skyMidColor")
      .name("Sky Mid")
      .onChange((value) => {
        agave.skylight_middle_color(
          (effectController["skyMidIntensity"] * value[0]) / 255.0,
          (effectController["skyMidIntensity"] * value[1]) / 255.0,
          (effectController["skyMidIntensity"] * value[2]) / 255.0
        );
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });
    lighting
      .add(effectController, "skyMidIntensity")
      .max(100.0)
      .min(0.01)
      .step(0.1)
      .onChange((value) => {
        agave.skylight_middle_color(
          (effectController["skyMidColor"][0] / 255.0) * value,
          (effectController["skyMidColor"][1] / 255.0) * value,
          (effectController["skyMidColor"][2] / 255.0) * value
        );
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });
    lighting
      .addColor(effectController, "skyBotColor")
      .name("Sky Bottom")
      .onChange((value) => {
        agave.skylight_bottom_color(
          (effectController["skyBotIntensity"] * value[0]) / 255.0,
          (effectController["skyBotIntensity"] * value[1]) / 255.0,
          (effectController["skyBotIntensity"] * value[2]) / 255.0
        );
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });
    lighting
      .add(effectController, "skyBotIntensity")
      .max(100.0)
      .min(0.01)
      .step(0.1)
      .onChange((value) => {
        agave.skylight_bottom_color(
          (effectController["skyBotColor"][0] / 255.0) * value,
          (effectController["skyBotColor"][1] / 255.0) * value,
          (effectController["skyBotColor"][2] / 255.0) * value
        );
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });
    lighting
      .add(effectController, "lightDistance")
      .max(100.0)
      .min(0.0)
      .step(0.1)
      .onChange((value) => {
        agave.light_pos(
          0,
          value,
          (effectController["lightTheta"] * 180.0) / 3.14159265,
          (effectController["lightPhi"] * 180.0) / 3.14159265
        );
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });
    lighting
      .add(effectController, "lightTheta")
      .max(180.0)
      .min(-180.0)
      .step(1)
      .onChange((value) => {
        agave.light_pos(
          0,
          effectController["lightDistance"],
          (value * 180.0) / 3.14159265,
          (effectController["lightPhi"] * 180.0) / 3.14159265
        );
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });
    lighting
      .add(effectController, "lightPhi")
      .max(180.0)
      .min(0.0)
      .step(1)
      .onChange((value) => {
        agave.light_pos(
          0,
          effectController["lightDistance"],
          (effectController["lightTheta"] * 180.0) / 3.14159265,
          (value * 180.0) / 3.14159265
        );
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });
    lighting
      .add(effectController, "lightSize")
      .max(100.0)
      .min(0.01)
      .step(0.1)
      .onChange((value) => {
        agave.light_size(0, value, value);
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });
    lighting
      .add(effectController, "lightIntensity")
      .max(100.0)
      .min(0.01)
      .step(0.1)
      .onChange((value) => {
        agave.light_color(
          0,
          (effectController["lightColor"][0] / 255.0) * value,
          (effectController["lightColor"][1] / 255.0) * value,
          (effectController["lightColor"][2] / 255.0) * value
        );
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });
    lighting
      .addColor(effectController, "lightColor")
      .name("lightcolor")
      .onChange((value) => {
        agave.light_color(
          0,
          (value[0] / 255.0) * effectController["lightIntensity"],
          (value[1] / 255.0) * effectController["lightIntensity"],
          (value[2] / 255.0) * effectController["lightIntensity"]
        );
        agave.flushCommandBuffer();
        this._stream_mode_suspended = true;
      })
      .onFinishChange((value) => {
        this._stream_mode_suspended = false;
      });

    //  var customContainer = document.getElementById('my-gui-container');
    //  customContainer.appendChild(gui.domElement);
  }

  private _setupChannelsGui() {
    const gui = this.gui;
    const agave = this.agave;
    const effectController = this.effectController;
    if (effectController && effectController.channelFolders) {
      for (var i = 0; i < effectController.channelFolders.length; ++i) {
        gui.removeFolder(effectController.channelFolders[i]);
      }
    }

    effectController.infoObj.channelGui = [];
    const initcolors: [number, number, number][] = [
      [255, 0, 255],
      [255, 255, 255],
      [0, 255, 255],
    ];
    effectController.channelFolders = [];
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
      effectController.channelFolders.push(f);
      f.add(effectController.infoObj.channelGui[i], "enabled").onChange(
        (function (j) {
          return (value) => {
            agave.enable_channel(j, value ? 1 : 0);
            agave.flushCommandBuffer();
          };
        })(i)
      );
      f.addColor(effectController.infoObj.channelGui[i], "colorD")
        .name("Diffuse")
        .onChange(
          (function (j) {
            return (value) => {
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
            return (value) => {
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
            return (value) => {
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
          ((j) => {
            return (value) => {
              if (!this.waiting_for_image) {
                agave.stream_mode(0);
                agave.set_window_level(
                  j,
                  value,
                  effectController.infoObj.channelGui[j].level
                );
                agave.flushCommandBuffer();
                this.waiting_for_image = true;
              }
              this._stream_mode_suspended = true;
            };
          })(i)
        )
        .onFinishChange((value) => {
          agave.stream_mode(1);
          agave.flushCommandBuffer();
          agave.redraw();
          agave.flushCommandBuffer();
          this._stream_mode_suspended = false;
        });

      f.add(effectController.infoObj.channelGui[i], "level")
        .max(1.0)
        .min(0.0)
        .step(0.001)
        .onChange(
          ((j) => {
            return (value) => {
              if (!this.waiting_for_image) {
                agave.stream_mode(0);
                agave.set_window_level(
                  j,
                  effectController.infoObj.channelGui[j].window,
                  value
                );
                agave.flushCommandBuffer();
                this.waiting_for_image = true;
              }
              this._stream_mode_suspended = true;
            };
          })(i)
        )
        .onFinishChange((value) => {
          agave.stream_mode(1);
          agave.flushCommandBuffer();
          agave.redraw();
          agave.flushCommandBuffer();
          this._stream_mode_suspended = false;
        });
      f.add(effectController.infoObj.channelGui[i], "roughness")
        .max(100.0)
        .min(0.0)
        .onChange(
          ((j) => {
            return (value) => {
              if (!this.waiting_for_image) {
                agave.mat_glossiness(j, value);
                agave.flushCommandBuffer();
                this.waiting_for_image = true;
              }
              this._stream_mode_suspended = true;
            };
          })(i)
        )
        .onFinishChange((value) => {
          this._stream_mode_suspended = false;
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
}

function init() {
  const app = new AgaveApp();
  function redrawLoop() {
    app.redraw();
    requestAnimationFrame(redrawLoop);
  }
  requestAnimationFrame(redrawLoop);
}

/**
 * calls the "init" method upon page load
 */
window.addEventListener("load", init, false);
