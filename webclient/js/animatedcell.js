
  var wsUri = "ws://localhost:1234";
  //var wsUri = "ws://dev-aics-dtp-001:1234";

  var binarysocket0 = null; //handles requests for image streaming target #1
  var binarysocket1 = null; //handles requests for image streaming target #2
  var jsonsocket0 = null;  //handles requests for dynamically populating the menu entries based on server feedback

  var dragFlag = 0; //for dragging in the render view
  var selectDragFlag = 0; //for dragging in the cell structure visibility widget
  var initialMouseX = 0;
  var initialMouseY = 0;
  var mouseSensi = 0.2;
  var img_width = 0;
  var img_height = 0;

  //vec3
  var rotationFrom;
  var rotationTo;
  var rotationAxis;
  var panVec;

  var modelView;

  //quaternions
  var rotation;
  var oldRotation;
  var rotationDelta;
  var tempold;
var slider_drag = false;

  // var bbbb = new commandBuffer();
  // bbbb.addCommand("EYE", 1, 1, 5);
  // bbbb.addCommand("TARGET", 3, 3, 0);
  // bbbb.addCommand("SESSION", "hello");
  // bbbb.addCommand("APERTURE", 7);
  // bbbb.prebufferToBuffer();
  // var bbbbview = new Uint8Array(bbbb.buffer);
  // console.log(bbbbview);




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
    color: [255, 255, 0]
  };

  var gui = new dat.GUI();
  //var gui = new dat.GUI({autoPlace:false, width:200});

  gui.add( effectController, "channel", [0,1,2,3,4,5,6,7]).name("Channel").onFinishChange(function(value) {
    var cb = new commandBuffer();
    cb.addCommand("CHANNEL", parseInt(value));
    flushCommandBuffer(cb);
  });
  gui.addColor(effectController, "color").name("Diffuse").onChange(function(value) {
    var cb = new commandBuffer();
    cb.addCommand("MAT_DIFFUSE", value[0]/255.0, value[1]/255.0, value[2]/255.0, 1.0);
    flushCommandBuffer(cb);
  });

//  var customContainer = document.getElementById('my-gui-container');
//  customContainer.appendChild(gui.domElement);
}

  /**
   *
   */
  function init()
  {
    binarysocket0 = new WebSocket(wsUri); //handles requests for image streaming target #1
    jsonsocket0 = new WebSocket(wsUri); //handles requests for image streaming target #1

      binarysock = new binarysocket(0);
      jsonsock = new jsonsocket();


      binarysocket0.binaryType = "arraybuffer";

      //socket connection for image stream #1
      binarysocket0.onopen = binarysock.open;
      binarysocket0.onclose = binarysock.close;
      binarysocket0.onmessage = binarysock.message0; //linked to message0
      binarysocket0.onerror = binarysock.error;

      jsonsocket0.binaryType = "arraybuffer";
      //socket connection for json message requests
      jsonsocket0.onopen = jsonsock.open;
      jsonsocket0.onclose = jsonsock.close;
      jsonsocket0.onmessage = jsonsock.message;
      jsonsocket0.onerror = jsonsock.error;

        //setup tooltips
      //readTextFile("data/tooltip.csv");

      rotationTo = new THREE.Vector3();
      rotationFrom = new THREE.Vector3();
      rotationAxis = new THREE.Vector3();
      panVec = new THREE.Vector3(0,0,-5);

      rotation = new THREE.Quaternion();
      oldRotation = new THREE.Quaternion();
      rotationDelta = new THREE.Quaternion();
      tempold = new THREE.Quaternion();

      modelView = new THREE.Matrix4();
      modelView.setPosition(panVec);

      var streamedImg = document.getElementsByClassName("streamed_img");

      for(var i=0; i<streamedImg.length; i++)
      {
          streamedImg[i].addEventListener("DOMMouseScroll", MouseWheelHandler, false);
          streamedImg[i].addEventListener("mousewheel", MouseWheelHandler, false);
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
        cb.addCommand("EYE", 0.5, 0.408, 2.145);
        cb.addCommand("TARGET", 0.5, 0.408, 0.145);
        cb.addCommand("MAT_DIFFUSE", 1.0, 1.0, 0.0, 1.0);
        cb.addCommand("MAT_SPECULAR", 0.0, 0.0, 0.0, 0.0);
        cb.addCommand("MAT_EMISSIVE", 0.0, 0.0, 0.0, 0.0);
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
    //todo: currently the differentiation between channel 0 and 1 is solved with 2 separate methods, since for some reason this.channelnum is always undefined
    this.message0 = function (evt) {
        var bytes = new Uint8Array(evt.data),
            binary = "",
            len = bytes.byteLength,
            i;

        for (i=0; i<len; i++)
            binary += String.fromCharCode(bytes[i]);

        //console.log("msg0 received");
        imgreceived = true;
        screenImage.set(binary, 0);


        // var reader = new FileReader();
        // reader.onload = function(e) {
        //   image.src = e.target.result;
        // };
        // reader.readAsDataURL(evt.data);

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
