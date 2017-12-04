// types: FLOAT32(f), INT32(i), STRING(s)=int32 and array of bytes
var types = {
  I32: 4,
  F32: 4,
  S: -1
}
// commands:
// SESSION s (what does this mean?)
// ASSET_PATH s
// LOAD_OME_TIF s : tell server to load cell in this session?
// EYE f f f : tell server where camera is
// TARGET f f f
// UP f f f
// APERTURE f
// FOV_Y f
// FOCALDIST f
// EXPOSURE f
// MAT.DIFFUSE f f f f
// MAT.SPECULAR f f f f
// MAT.EMISSIVE f f f f
// RENDER.ITERATIONS i
// STREAM_MODE i (continuous or on-demand frames)
// REDRAW

// command id will be int32 to future-proof it.
// note that the server needs to know these signatures too.
var COMMANDS = {
  // tell server to identify this session?
  SESSION: [0, "S"],
  // tell server where files might be (appends to existing)
  ASSET_PATH: [1, "S"],
  // load a volume
  LOAD_OME_TIF: [2, "S"],
  // set camera pos
  EYE: [3, "F32", "F32", "F32"],
  // set camera target pt
  TARGET: [4, "F32", "F32", "F32"],
  // set camera up direction
  UP: [5, "F32", "F32", "F32"],
  APERTURE: [6, "F32"],
  FOV_Y: [7, "F32"],
  FOCALDIST: [8, "F32"],
  EXPOSURE: [9, "F32"],
  MAT_DIFFUSE: [10, "F32", "F32", "F32", "F32"],
  MAT_SPECULAR: [11, "F32", "F32", "F32", "F32"],
  MAT_EMISSIVE: [12, "F32", "F32", "F32", "F32"],
  // set num render iterations
  RENDER_ITERATIONS: [13, "I32"],
  // (continuous or on-demand frames)
  STREAM_MODE: [14, "I32"],
  // request new image
  REDRAW: [15],
};

// strategy: add elements to prebuffer, and then traverse prebuffer to convert to binary before sending?
function commandBuffer() {
  // [command, args],...
  this.prebuffer = [];
  this.buffer = null;
}

commandBuffer.prototype = {
  prebufferToBuffer: function() {
    // iterate length of prebuffer to compute size.
    var bytesize = 0;
    for (var i = 0; i < this.prebuffer.length; ++i) {
      // for each command.

      // one i32 for the command id.
      bytesize += types.I32;

      var command = this.prebuffer[i];
      var commandCode = command[0];
      var signature = COMMANDS[commandCode];
      var nArgsExpected = signature.length-1;
      // for each arg:
      if (command.length-1 !== nArgsExpected) {
        console.error("BAD COMMAND: EXPECTED " + nArgsExpected + " args and got " + command.length-1);
      }

      for (var j = 0; j < nArgsExpected; ++j) {
        // get arg type
        var argtype = signature[j+1];
        if (argtype === "S") {
          // one int32 for string length
          bytesize += 4;
          // followed by one byte per char.
          bytesize += command[j+1].length;
        }
        else {
          bytesize += types[argtype];
        }

      }
    }
    // allocate arraybuffer and then fill it.
    this.buffer = new ArrayBuffer(bytesize);
    var dataview = new DataView(this.buffer);
    var offset = 0;
    var LITTLE_ENDIAN = true;
    for (var i = 0; i < this.prebuffer.length; ++i) {
      var cmd = this.prebuffer[i];
      var commandCode = cmd[0];
      var signature = COMMANDS[commandCode];
      var nArgsExpected = signature.length-1;

      // the numeric code for the command
      dataview.setInt32(offset, signature[0]);
      offset+=4;
      for (var j = 0; j < nArgsExpected; ++j) {
        // get arg type
        var argtype = signature[j+1];
        switch(argtype) {
          case "S":
            var str = cmd[j+1];
            dataview.setInt32(offset, str.length, LITTLE_ENDIAN);
            offset+=4;
            for (var k = 0; k < str.length; ++k) {
              dataview.setUint8(offset, str.charCodeAt(k));
              offset+=1;
            }
          break;
          case "F32":
            dataview.setFloat32(offset, cmd[j+1], LITTLE_ENDIAN);
            offset+=4;
          break;
          case "I32":
            dataview.setInt32(offset, cmd[j+1], LITTLE_ENDIAN);
            offset+=4;
          break;
        }
      }
    }
    // result is in this.buffer
    return this.buffer;
  },
  // commands are added by command code string name followed by appropriate signature args.
  addCommand: function() {
    var args = [].slice.call(arguments);
    // TODO: check against signature!!!
    this.prebuffer.push(args);
  },

};
