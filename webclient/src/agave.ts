import { CommandBuffer, COMMANDS } from "./commandbuffer";

export class AgaveClient {
  private binarysocket0: WebSocket;
  private cb: CommandBuffer;
  private session_name: string;
  private onOpen: () => void;
  private onJson: (json: any) => void;
  private enqueued_image_data: string = "";
  private onImage: (data: string) => void;

  constructor(
    url = "ws://localhost:1235/",
    rendermode = "pathtrace",
    onOpen = () => {},
    onJson = (json: any) => {},
    onImage = (data: string) => {}
  ) {
    if (rendermode !== "pathtrace" && rendermode !== "raymarch") {
      rendermode = "pathtrace";
    }
    this.onOpen = onOpen;
    this.onJson = onJson;
    this.onImage = onImage;
    this.binarysocket0 = new WebSocket(url + "?mode=" + rendermode);
    this.binarysocket0.binaryType = "arraybuffer";
    this.binarysocket0.onopen = (_ev: Event) => {
      this.set_resolution(256, 256);
      // put agave in streaming mode from the get-go
      this.stream_mode(1);
      this.flushCommandBuffer();

      // user provided callback
      if (this.onOpen) {
        this.onOpen();
      }
    };
    this.binarysocket0.onclose = (_ev: CloseEvent) => {
      setTimeout(function () {
        console.warn("connection failed. refresh to retry.");
      }, 3000);
    };
    this.binarysocket0.onmessage = (evt: MessageEvent<any>) => {
      if (typeof evt.data === "string") {
        var returnedObj = JSON.parse(evt.data);
        if (returnedObj.commandId === COMMANDS.LOAD_DATA[0]) {
          console.log(returnedObj);
          // set up gui!
          if (this.onJson) {
            this.onJson(returnedObj);
          }
        }
        return;
      }

      // new data will be used to obliterate the previous data if it exists.
      // in this way, two consecutive images between redraws, will not both be drawn.
      // TODO:enqueue this...?
      const arraybuf = evt.data;
      var bytes = new Uint8Array(arraybuf),
        binary = "",
        len = bytes.byteLength,
        i;
      for (i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      // call btoa here or on the other side of this interface????
      this.enqueued_image_data = window.btoa(binary);
      if (this.onImage) {
        this.onImage(this.enqueued_image_data);
      }
    };
    this.binarysocket0.onerror = (evt: Event) => {
      console.log("error", evt);
    };

    this.cb = new CommandBuffer();
    this.session_name = "";
  }

  session(name: string) {
    /*
  Set the current session name.  Use the full path to the name of the output
  image here.

  Parameters
  ----------
  name: str
      This name is the full path to the output image, ending in .png or .jpg.
      Make sure the directory has already been created.
  */
    // 0
    this.cb.addCommand("SESSION", name);
    this.session_name = name;
  }
  asset_path(name: string) {
    /*
  Sets a search path for volume files. NOT YET IMPLEMENTED.

  Parameters
  ----------
  name: str
      This name is the path where volume images are located.
  */
    // 1
    this.cb.addCommand("ASSET_PATH", name);
  }
  // load_ome_tif(name: string) {
  //   /*
  // DEPRECATED. Use load_data
  // */
  //   // 2
  //   this.cb.addCommand("LOAD_OME_TIF", name);
  // }
  eye(x: number, y: number, z: number) {
    /*
  Set the viewer camera position.
  Default is (500,500,500).

  Parameters
  ----------
  x: number
      The x coordinate
  y: number
      The y coordinate
  z: number
      The z coordinate
      */
    // 3
    this.cb.addCommand("EYE", x, y, z);
  }
  target(x: number, y: number, z: number) {
    /*
  Set the viewer target position. This is a point toward which we are looking.
  Default is (0,0,0).

  Parameters
  ----------
  x: number
      The x coordinate
  y: number
      The y coordinate
  z: number
      The z coordinate
      */
    // 4
    this.cb.addCommand("TARGET", x, y, z);
  }

  up(x: number, y: number, z: number) {
    /*
  Set the viewer camera up direction.  This is a vector which should be nearly
  perpendicular to the view direction (target-eye), and defines the "roll" amount
  for the camera.
  Default is (0,0,1).

  Parameters
  ----------
  x: number
      The x coordinate
  y: number
      The y coordinate
  z: number
      The z coordinate
      */
    // 5
    this.cb.addCommand("UP", x, y, z);
  }

  aperture(x: number) {
    /*
  Set the viewer camera aperture size.

  Parameters
  ----------
  x: number
      The aperture size.
      This is a number between 0 and 1. 0 means no defocusing will occur, like a
      pinhole camera.  1 means maximum defocus. Default is 0.
      */
    // 6
    this.cb.addCommand("APERTURE", x);
  }

  camera_projection(projection_type: number, x: number) {
    /*
  Set the viewer camera projection type, along with a relevant parameter.

  Parameters
  ----------
  projection_type: number
      0 for Perspective, 1 for Orthographic.  Default: 0
  x: number
      If Perspective, then this is the vertical Field of View angle in degrees.
      If Orthographic, then this is the orthographic scale dimension.
      Default: 55.0 degrees. (default Orthographic scale is 0.5)
      */
    // 7
    this.cb.addCommand("CAMERA_PROJECTION", projection_type, x);
  }
  focaldist(x: number) {
    /*
  Set the viewer camera focal distance

  Parameters
  ----------
  x: number
      The focal distance.  Has no effect if aperture is 0.
      */
    // 8
    this.cb.addCommand("FOCALDIST", x);
  }
  exposure(x: number) {
    /*
  Set the exposure level

  Parameters
  ----------
  x: number
      The exposure level between 0 and 1. Default is 0.75.  Higher numbers are
      brighter.
      */
    // 9
    this.cb.addCommand("EXPOSURE", x);
  }
  mat_diffuse(channel: number, r: number, g: number, b: number, a: number) {
    /*
  Set the diffuse color of a channel

  Parameters
  ----------
  channel: number
      Which channel index, 0 based.
  r: number
      The red value between 0 and 1
  g: number
      The green value between 0 and 1
  b: number
      The blue value between 0 and 1
  a: number
      The alpha value between 0 and 1 (currently unused)
      */
    // 10
    this.cb.addCommand("MAT_DIFFUSE", channel, r, g, b, a);
  }
  mat_specular(channel: number, r: number, g: number, b: number, a: number) {
    /*
  Set the specular color of a channel (defaults to black, for no specular
  response)

  Parameters
  ----------
  channel: number
      Which channel index, 0 based.
  r: number
      The red value between 0 and 1
  g: number
      The green value between 0 and 1
  b: number
      The blue value between 0 and 1
  a: number
      The alpha value between 0 and 1 (currently unused)
      */
    // 11
    this.cb.addCommand("MAT_SPECULAR", channel, r, g, b, a);
  }
  mat_emissive(channel: number, r: number, g: number, b: number, a: number) {
    /*
  Set the emissive color of a channel (defaults to black, for no emission)

  Parameters
  ----------
  channel: number
      Which channel index, 0 based.
  r: number
      The red value between 0 and 1
  g: number
      The green value between 0 and 1
  b: number
      The blue value between 0 and 1
  a: number
      The alpha value between 0 and 1 (currently unused)
      */
    // 12
    this.cb.addCommand("MAT_EMISSIVE", channel, r, g, b, a);
  }
  render_iterations(x: number) {
    /*
  Set the number of paths per pixel to accumulate.

  Parameters
  ----------
  x: number
      How many paths per pixel. The more paths, the less noise in the image.
      */
    // 13
    this.cb.addCommand("RENDER_ITERATIONS", x);
  }
  stream_mode(x: number) {
    /*
  Turn stream mode on or off.  Stream mode will send an image back to the client
  on each iteration up to some server-defined amount.  This mode is useful for
  interactive client-server applications but not for batch-mode offline rendering.

  Parameters
  ----------
  x: number
      0 for off, 1 for on. Default is off.
      */
    // 14
    this.cb.addCommand("STREAM_MODE", x);
  }
  redraw() {
    /*
  Tell the server to process all commands and return an image
  TODO , and then save the image.
  TODO This function will block and wait for the image to be returned.
  TODO The image returned will be saved automatically using the session_name.
  TODO: a timeout is not yet implemented.
  */
    // 15
    // issue command buffer
    this.cb.addCommand("REDRAW");
    this.flushCommandBuffer();
    // //  and then WAIT for render to be completed
    // binarydata = this.ws.wait_for_image();
    // // and save image
    // im = Image.open(binarydata);
    // print(this.session_name);
    // im.save(this.session_name);
    // // ready for next frame
    // this.session_name = "";
  }
  set_resolution(x: number, y: number) {
    /*
  Set the image resolution in pixels.

  Parameters
  ----------
  x: number
      x resolution in pixels
  y: number
      y resolution in pixels
      */
    // 16
    this.cb.addCommand("SET_RESOLUTION", x, y);
  }
  density(x: number) {
    /*
  Set the scattering density.

  Parameters
  ----------
  x: number
      The scattering density, 0-100.  Higher values will make the volume seem
      more opaque. Default is 8.5, which is relatively transparent.
      */
    // 17
    this.cb.addCommand("DENSITY", x);
  }
  frame_scene() {
    /*
  Automatically set camera parameters so that the volume fills the view.
  Useful when you have insufficient information to position the camera accurately.
  */
    // 18
    this.cb.addCommand("FRAME_SCENE");
  }
  mat_glossiness(channel: number, glossiness: number) {
    /*
  Set the channel's glossiness.

  Parameters
  ----------
  channel: number
      Which channel index, 0 based.
  glossiness: number
      Sets the shininess, a number between 0 and 100.
      */
    // 19
    this.cb.addCommand("MAT_GLOSSINESS", channel, glossiness);
  }
  enable_channel(channel: number, enabled: number) {
    /*
  Show or hide a given channel

  Parameters
  ----------
  channel: number
      Which channel index, 0 based.
  enabled: number
      0 to hide, 1 to show
      */
    // 20
    this.cb.addCommand("ENABLE_CHANNEL", channel, enabled);
  }
  set_window_level(channel: number, window: number, level: number) {
    /*
  Set intensity threshold for a given channel based on Window/Level

  Parameters
  ----------
  channel: number
      Which channel index, 0 based.
  window: number
      Width of the window, from 0-1.
  level: number
      Intensity level mapped to middle of window, from 0-1
      */
    // 21
    this.cb.addCommand("SET_WINDOW_LEVEL", channel, window, level);
  }
  orbit_camera(theta: number, phi: number) {
    /*
  Rotate the camera around the volume by angle deltas

  Parameters
  ----------
  theta: number
      polar angle in degrees
  phi: number
      azimuthal angle in degrees
      */
    // 22
    this.cb.addCommand("ORBIT_CAMERA", theta, phi);
  }
  trackball_camera(theta: number, phi: number) {
    /*
  Rotate the camera around the volume by angle deltas

  Parameters
  ----------
  theta: number
      vertical screen angle in degrees
  phi: number
      horizontal screen angle in degrees
      */
    // 43
    this.cb.addCommand("TRACKBALL_CAMERA", theta, phi);
  }
  skylight_top_color(r: number, g: number, b: number) {
    /*
  Set the "north pole" color of the sky sphere

  Parameters
  ----------
  r: number
      The red value between 0 and 1
  g: number
      The green value between 0 and 1
  b: number
      The blue value between 0 and 1
      */
    // 23
    this.cb.addCommand("SKYLIGHT_TOP_COLOR", r, g, b);
  }
  skylight_middle_color(r: number, g: number, b: number) {
    /*
  Set the "equator" color of the sky sphere

  Parameters
  ----------
  r: number
      The red value between 0 and 1
  g: number
      The green value between 0 and 1
  b: number
      The blue value between 0 and 1
      */
    // 24
    this.cb.addCommand("SKYLIGHT_MIDDLE_COLOR", r, g, b);
  }
  skylight_bottom_color(r: number, g: number, b: number) {
    /*
  Set the "south pole" color of the sky sphere

  Parameters
  ----------
  r: number
      The red value between 0 and 1
  g: number
      The green value between 0 and 1
  b: number
      The blue value between 0 and 1
      */
    // 25
    this.cb.addCommand("SKYLIGHT_BOTTOM_COLOR", r, g, b);
  }
  light_pos(index: number, r: number, theta: number, phi: number) {
    /*
  Set the position of an area light, in spherical coordinates

  Parameters
  ----------
  index: number
      Which light to set.  Currently unused as there is only one area light.
  r: number
      The radius, as distance from the center of the volume
  theta: number
      The polar angle
  phi: number
      The azimuthal angle
      */
    // 26
    this.cb.addCommand("LIGHT_POS", index, r, theta, phi);
  }
  light_color(index: number, r: number, g: number, b: number) {
    /*
  Set the color of an area light. Overdrive the values higher than 1 to increase
  the light's intensity.

  Parameters
  ----------
  index: number
      Which light to set.  Currently unused as there is only one area light.
  r: number
      The red value between 0 and 1
  g: number
      The green value between 0 and 1
  b: number
      The blue value between 0 and 1
      */
    // 27
    this.cb.addCommand("LIGHT_COLOR", index, r, g, b);
  }
  light_size(index: number, x: number, y: number) {
    /*
  Set the size dimensions of a rectangular area light.

  Parameters
  ----------
  index: number
      Which light to set.  Currently unused as there is only one area light.
  x: number
      The width dimension of the area light
  y: number
      The height dimension of the area light
      */
    // 28
    this.cb.addCommand("LIGHT_SIZE", index, x, y);
  }
  set_clip_region(
    minx: number,
    maxx: number,
    miny: number,
    maxy: number,
    minz: number,
    maxz: number
  ) {
    /*
  Set the axis aligned region of interest of the volume. All axis values are
  relative, where 0 is one extent of the volume and 1 is the opposite extent.
  For example, (0,1, 0,1, 0,0.5) will select the lower half of the volume's z
  slices.

  Parameters
  ----------
  minx: number
      The lower x extent between 0 and 1
  maxx: number
      The higher x extent between 0 and 1
  miny: number
      The lower y extent between 0 and 1
  maxy: number
      The higher y extent between 0 and 1
  minz: number
      The lower z extent between 0 and 1
  maxz: number
      The higher z extent between 0 and 1
      */
    // 29
    this.cb.addCommand("SET_CLIP_REGION", minx, maxx, miny, maxy, minz, maxz);
  }
  set_voxel_scale(x: number, y: number, z: number) {
    /*
  Set the relative scale of the pixels in the volume. Typically this is filled in
  with the physical pixel dimensions from the microscope metadata.  Often the x
  and y scale will differ from the z scale.  Defaults to (1,1,1)

  Parameters
  ----------
  x: number
      x scale
  y: number
      y scale
  z: number
      z scale
      */
    // 30
    this.cb.addCommand("SET_VOXEL_SCALE", x, y, z);
  }
  auto_threshold(channel: number, method: number) {
    /*
  Automatically determine the intensity thresholds

  Parameters
  ----------
  channel: number
      Which channel index, 0 based.
  method: number
      Allowed values:
      0: Auto2
      1: Auto
      2: BestFit
      3: ChimeraX emulation
      4: between 0.5 percentile and 0.98 percentile
      */
    // 31
    this.cb.addCommand("AUTO_THRESHOLD", channel, method);
  }
  set_percentile_threshold(channel: number, pct_low: number, pct_high: number) {
    /*
  Set intensity thresholds based on percentiles of pixels to clip min and max
  intensity

  Parameters
  ----------
  channel: number
      Which channel index, 0 based.
  pct_low: number
      The low percentile to remap to 0(min) intensity
  pct_high: number
      The high percentile to remap to 1(max) intensity
      */
    // 32
    this.cb.addCommand("SET_PERCENTILE_THRESHOLD", channel, pct_low, pct_high);
  }
  mat_opacity(channel: number, opacity: number) {
    /*
  Set channel opacity. This is a multiplier against all intensity values in the
  channel.

  Parameters
  ----------
  channel: number
      Which channel index, 0 based.
  opacity: number
      A multiplier between 0 and 1. Default is 1
      */
    // 33
    this.cb.addCommand("MAT_OPACITY", channel, opacity);
  }
  set_primary_ray_step_size(step_size: number) {
    /*
  Set primary ray step size. This is an accuracy versus speed tradeoff.  Low
  values are more accurate. High values will render faster.
  Primary rays are the rays that are cast from the camera out into the volume.

  Parameters
  ----------
  step_size: number
      A value in voxels. Default is 4.  Minimum sensible value is 1.
      */
    // 34
    this.cb.addCommand("SET_PRIMARY_RAY_STEP_SIZE", step_size);
  }
  set_secondary_ray_step_size(step_size: number) {
    /*
  Set secondary ray step size. This is an accuracy versus speed tradeoff.  Low
  values are more accurate. High values will render faster.
  The secondary rays are rays which are cast toward lights after they have
  scattered within the volume.

  Parameters
  ----------
  step_size: number
      A value in voxels. Default is 4.  Minimum sensible value is 1.
      */
    // 35
    this.cb.addCommand("SET_SECONDARY_RAY_STEP_SIZE", step_size);
  }
  background_color(r: number, g: number, b: number) {
    /*
  Set the background color of the rendering

  Parameters
  ----------
  r: number
      The red value between 0 and 1
  g: number
      The green value between 0 and 1
  b: number
      The blue value between 0 and 1
      */
    // 36
    this.cb.addCommand("BACKGROUND_COLOR", r, g, b);
  }
  set_isovalue_threshold(channel: number, isovalue: number, isorange: number) {
    /*
  Set intensity thresholds based on values around an isovalue.

  Parameters
  ----------
  channel: number
      Which channel index, 0 based.
  isovalue: number
      The value to center at maximum intensity, between 0 and 1
  isorange: number
      A range around the isovalue to keep at constant intensity, between 0 and 1.
      Typically small, to select for a single isovalue.
      */
    // 37
    this.cb.addCommand("SET_ISOVALUE_THRESHOLD", channel, isovalue, isorange);
  }
  set_control_points(channel: number, data: number[]) {
    /*
  Set intensity thresholds based on a piecewise linear transfer function.

  Parameters
  ----------
  channel: number
      Which channel index, 0 based.
  data: List[float]
      An array of values.  5 floats per control point.  first is position (0-1),
      next four are rgba (all 0-1).  Only alpha is currently used as the remapped
      intensity value.  All others are linearly interpolated.
      */
    // 38
    this.cb.addCommand("SET_CONTROL_POINTS", channel, data);
  }
  // load_volume_from_file(path: string, scene: number, time: number) {
  //   /*
  // DEPRECATED. Use load_data
  // */
  //   // 39
  //   this.cb.addCommand("LOAD_VOLUME_FROM_FILE", path, scene, time);
  // }
  set_time(time: number) {
    /*
  Load a time from the current volume file

  Parameters
  ----------
  time: number
      zero-based index to select the time sample.  Defaults to 0
      */
    // 40
    this.cb.addCommand("SET_TIME", time);
  }
  bounding_box_color(r: number, g: number, b: number) {
    /*
  Set the color for the bounding box display

  Parameters
  ----------
  r: number
      the red value, from 0 to 1
  g: number
      the green value, from 0 to 1
  b: number
      the blue value, from 0 to 1
  */
    // 41
    this.cb.addCommand("SET_BOUNDING_BOX_COLOR", r, g, b);
  }
  show_bounding_box(on: number) {
    /*
  Turn bounding box display on or off

  Parameters
  ----------
  on: number
      0 to hide bounding box, 1 to show it
      */
    // 42
    this.cb.addCommand("SHOW_BOUNDING_BOX", on);
  }
  load_data(
    path: string,
    scene: number = 0,
    multiresolution_level: number = 0,
    time: number = 0,
    channels: number[] = [],
    region: number[] = []
  ) {
    /*
  Completely specify volume data to load

  Parameters
  ----------
  path: str
      URL or directory or file path to the data. The path must be locally
      accessible from the AGAVE server.

  scene: number
      zero-based index to select the scene, for multi-scene files. Defaults to 0

  multiresolution_level: number
      zero-based index to select the multiresolution level.  Defaults to 0

  time: number
      zero-based index to select the time sample.  Defaults to 0

  channels: List[int]
      zero-based indices to select the channels.  Defaults to all channels

  region: List[int]
      6 integers specifying the region to load.  Defaults to the entire volume.
      Any list length other than 0 or 6 is an error.
  */
    // 44
    this.cb.addCommand(
      "LOAD_DATA",
      path,
      scene,
      multiresolution_level,
      time,
      channels,
      region
    );
  }
  flushCommandBuffer() {
    const buf = this.cb.prebufferToBuffer();
    this.binarysocket0.send(buf);
    // assuming the buffer is sent, prepare a new one
    this.cb = new CommandBuffer();
  }
}
