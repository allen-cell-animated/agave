
  //var wsUri = "ws://localhost:1235";
  var wsUri = "ws://dev-aics-dtp-001:1235";

  var binarysocket0 = null; //handles requests for image streaming target #1
  //var binarysocket1 = null; //handles requests for image streaming target #2
  var jsonsocket0 = null;  //handles requests for dynamically populating the menu entries based on server feedback

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


  /**
   * switches the supplied element to (in)visible
   * @param element
   * @param visible
   */
  function toggleDivVisibility(element, visible) {
      element.style.visibility = (visible ? "visible" : "hidden");
  }


var binarysock, jsonsock;

function setupGui() {

  effectController = {
    channel: 0,
    colorD0: [255, 0, 255],
    colorD1: [255, 255, 255],
    colorD2: [0, 255, 255],
    colorS0: [0, 0, 0],
    colorS1: [0, 0, 0],
    colorS2: [0, 0, 0],
    colorE0: [0, 0, 0],
    colorE1: [0, 0, 0],
    colorE2: [0, 0, 0],
    window0: 1.0,
    window1: 1.0,
    window2: 1.0,
    level0: 0.5,
    level1: 0.5,
    level2: 0.5,
    roughness0: 0.0,
    roughness1: 0.0,
    roughness2: 0.0,
    density: 50.0,
    exposure: 0.5,
    stream: false
  };

  var gui = new dat.GUI();
  //var gui = new dat.GUI({autoPlace:false, width:200});

  gui.add(effectController, "stream").onChange(function(value) {
    var cb = new commandBuffer();
    cb.addCommand("STREAM_MODE", value ? 1 : 0);
    flushCommandBuffer(cb);
    _stream_mode = value;
  });
  gui.add(effectController, "exposure").max(1.0).min(0.0).step(0.001).onChange(function(value) {
    var cb = new commandBuffer();
    cb.addCommand("EXPOSURE", value);
    flushCommandBuffer(cb);
    _stream_mode_suspended = true;
  }).onFinishChange(function(value) {
      _stream_mode_suspended = false;
  });
  gui.add(effectController, "density").max(100.0).min(0.0).step(0.001).onChange(function(value) {
        var cb = new commandBuffer();
        cb.addCommand("DENSITY", value);
        flushCommandBuffer(cb);
        _stream_mode_suspended = true;
    }).onFinishChange(function(value) {
        _stream_mode_suspended = false;
    });
  for (var i = 0; i < 3; ++i) {
    gui.addColor(effectController, "colorD"+i).name("Diffuse"+i).onChange(function(j) { 
        return function(value) {
                var cb = new commandBuffer();
                cb.addCommand("MAT_DIFFUSE", j, value[0]/255.0, value[1]/255.0, value[2]/255.0, 1.0);
                flushCommandBuffer(cb);
            };
    }(i));
    gui.addColor(effectController, "colorS"+i).name("Specular"+i).onChange(function(j) {
        return function(value) {
            var cb = new commandBuffer();
            cb.addCommand("MAT_SPECULAR", j, value[0]/255.0, value[1]/255.0, value[2]/255.0, 1.0);
            flushCommandBuffer(cb);
        };
    }(i));
    gui.addColor(effectController, "colorE"+i).name("Emissive"+i).onChange(function(j) {
        return function(value) {
            var cb = new commandBuffer();
            cb.addCommand("MAT_EMISSIVE", j, value[0]/255.0, value[1]/255.0, value[2]/255.0, 1.0);
            flushCommandBuffer(cb);
        };
    }(i));
    gui.add(effectController, "window"+i).max(1.0).min(0.0).step(0.001).onChange(function(j) {
        return function(value) {
            var cb = new commandBuffer();
            cb.addCommand("SET_WINDOW_LEVEL", j, value, effectController["level"+j]);
            flushCommandBuffer(cb);
            _stream_mode_suspended = true;
        }
    }(i))
    .onFinishChange(function(value) {
        _stream_mode_suspended = false;
    });

    gui.add(effectController, "level"+i).max(1.0).min(0.0).step(0.001).onChange(function(j) {
        return function(value) {
            var cb = new commandBuffer();
            cb.addCommand("SET_WINDOW_LEVEL", j, effectController["window"+j], value);
            flushCommandBuffer(cb);
            _stream_mode_suspended = true;
        }
    }(i))
    .onFinishChange(function(value) {
        _stream_mode_suspended = false;
    });
    gui.add(effectController, "roughness"+i).max(100.0).min(0.0).onChange(function(j) {
        return function(value) {
            var cb = new commandBuffer();
            cb.addCommand("MAT_GLOSSINESS", j, value);
            flushCommandBuffer(cb);
            _stream_mode_suspended = true;
        }
    }(i))
    .onFinishChange(function(value) {
        _stream_mode_suspended = false;
    });
    
  }

//  var customContainer = document.getElementById('my-gui-container');
//  customContainer.appendChild(gui.domElement);
}

  /**
   *
   */
  function init()
  {
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

    // camera manipulations
    for(var i=0; i<streamedImg.length; i++)
    {
        streamedImg[i].addEventListener("wheel", MouseWheelHandler, false);
        streamedImg[i].addEventListener("mousedown", MouseDownHandler, false);
        streamedImg[i].addEventListener('ondragstart', DragStartHandler, false);
    }


    //set up first tab
    var streamimg1 = document.getElementById("imageA");

    toggleDivVisibility(streamimg1, true);

    setupGui();
  }



  /**
   * socket that exclusively receives binary data for streaming jpg images
   * @param channelnumber = 0 or 1 for left or right image => currently message0 or message1 are used since channelnumber cannot always be set via the constructor for some reason
   */
  function binarysocket(channelnumber = 0) {
    this.channelnum = channelnumber;
    this.open = function (evt) {
        //send the initial camera & data query upon opening the connection


        var cb = new commandBuffer();
        cb.addCommand("LOAD_OME_TIF", "//allen/aics/animated-cell/Allen-Cell-Explorer/Allen-Cell-Explorer_1.1.1/Cell-Viewer_Data/2017_05_15_tubulin/AICS-12/AICS-12_790.ome.tif");
        cb.addCommand("SET_RESOLUTION", 512, 512);
        cb.addCommand("FRAME_SCENE");
        //cb.addCommand("EYE", 0.5, 0.408, 2.145);
        //cb.addCommand("TARGET", 0.5, 0.408, 0.145);
        cb.addCommand("MAT_DIFFUSE", 0, 1.0, 0.0, 1.0, 1.0);
        cb.addCommand("MAT_SPECULAR", 0, 0.0, 0.0, 0.0, 0.0);
        cb.addCommand("MAT_EMISSIVE", 0, 0.0, 0.0, 0.0, 0.0);
        cb.addCommand("MAT_DIFFUSE", 1, 1.0, 1.0, 1.0, 1.0);
        cb.addCommand("MAT_SPECULAR", 1, 0.0, 0.0, 0.0, 0.0);
        cb.addCommand("MAT_EMISSIVE", 1, 0.0, 0.0, 0.0, 0.0);
        cb.addCommand("MAT_DIFFUSE", 2, 0.0, 1.0, 1.0, 1.0);
        cb.addCommand("MAT_SPECULAR", 2, 0.0, 0.0, 0.0, 0.0);
        cb.addCommand("MAT_EMISSIVE", 2, 0.0, 0.0, 0.0, 0.0);
        cb.addCommand("APERTURE", 0.0);
        cb.addCommand("EXPOSURE", 0.5);
        flushCommandBuffer(cb);

        // init camera
        var streamimg1 = document.getElementById("imageA");
        gCamera = new THREE.PerspectiveCamera(55.0, 1.0, 0.001, 20);
        gCamera.position.x = 0.5;
        gCamera.position.y = 0.408;
        gCamera.position.z = 2.145;
        gCamera.up.x = 0.0;
        gCamera.up.y = 1.0;
        gCamera.up.z = 0.0;
        gControls = new AICStrackballControls(gCamera, streamimg1);
        gControls.target.x = 0.5;
        gControls.target.y = 0.408;
        gControls.target.z = 0.145;
        gControls.target0 = gControls.target.clone();
        gControls.rotateSpeed = 4.0/window.devicePixelRatio;
        gControls.autoRotate = false;
        gControls.staticMoving = true;
        gControls.length = 10;
        gControls.enabled = true; //turn off mouse moments by setting to false

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

        for (i=0; i<len; i++)
            binary += String.fromCharCode(bytes[i]);

        //console.log("msg received");
        screenImage.set(binary, this.channelnum);

    };

    this.message0 = function (evt) {
      console.time('recv');

      var bytes = new Uint8Array(evt.data),
        binary = "",
        len = bytes.byteLength,
        i;
      for (i=0; i<len; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      imgreceived = true;
      screenImage.set("data:image/png;base64,"+window.btoa( binary ), 0);
      console.timeEnd('recv');

      if (!_stream_mode_suspended && _stream_mode && !dragFlag) {
        // let cb = new commandBuffer();
        // cb.addCommand("REDRAW");
        // flushCommandBuffer(cb);
      }

// why should this code be slower?
      // var reader = new FileReader();
      // reader.onload = function(e) {
      //   imgreceived = true;
      //   screenImage.set(e.target.result, 0);
      //   console.timeEnd('recv');
      // };
      // reader.readAsDataURL(new Blob([new Uint8Array(evt.data)]));

    };

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

  function send(msg)
  {
      this.mouseMoveTimer = setTimeout(function () {
        binarysocket0.send(msg);
      }.bind(this), 200);
  }

  var lastmsg;

  function flushCommandBuffer(cmdbuf) {
    var buf = cmdbuf.prebufferToBuffer();
    binarysocket0.send(buf);
  }

  /**
   * calls the "init" method upon page load
   */
  window.addEventListener("load", init, false);


  var allText;

  /**
   * reads a file with the supplied path with an XMLHttpRequest, and calls a callback to handle the text
   * @param file = file path
   */
  function readTextFile(file, cb)
  {
      var xhr = new XMLHttpRequest();
      xhr.open("GET", file, true);
      xhr.onload = function (e) {
          if (xhr.readyState === 4) {
              if (xhr.status === 200) {
                  //console.log(xhr.responseText);
                  allText = xhr.responseText;

                  // DO SOMETHING WITH TEXT
                  if (cb) {
                    cb(allText);
                  }

              } else {
                  console.error(xhr.statusText);
              }
          }
      };
      xhr.onerror = function (e) {
          console.error(xhr.statusText);
      };
      xhr.send(null);
  }


var imgreceived = true;
var stopanimation = false;
var isplaying = false;

  /**
   * manages the animation state and triggers server requests
   * @param state = current animation state
   * @param step = normalized step that eventually sums up to finalstate
   * @param finalstate = finial animation state (should be 1 since the server expects a value between 0 and 1)
   * @param frequency = the waiting time between updates
   */
function playAnimation(state, step, finalstate, frequency)
{
    // console.log(stopanimation);
    // console.log(isplaying);
    if(imgreceived)
    {
        //imgreceived = false;
        messageobj.animationstate = state;
        //console.log("playback: " + state);
        triggerUpdate(messageobj);

        if(state == finalstate || stopanimation)
        {
            stopanimation = false;
            isplaying = false;
            messageobj.animationstate = 1;
            return;
        }
        isplaying = true;
        state += step;
        if(state > finalstate) state = finalstate;
    }

    setTimeout(function () {playAnimation(state, step, finalstate, frequency);}, frequency);
}


  /**
   * returns the index of the div in respect to its parent div
   * @param child
   * @returns {number}
   */
  function getIndexOfChild(child) {
    var selectedindex = 0;

    while (child = child.previousSibling) {
        if (child.nodeType === 1) { ++selectedindex}
    }
    return selectedindex;
}

  /**
   * adds or removes a given classname to a all children of the supplied parent container
   * @param parent
   * @param add (bool for toggling adding/removing)
   * @param classname
   */
  function changeChildAttribute(parent, add, classname)
{
    for(var i=0; i<parent.childElementCount; i++)
    {
        if(add) parent.children[i].classList.add(classname);
        else parent.children[i].classList.remove(classname);
    }
}


  /**
   * helper function to delete child elements of a div
   * @param elmnt
   */
  function clearChildren(elmnt)
{
    while (elmnt.firstChild) {
        elmnt.removeChild(elmnt.firstChild);
    }
}
