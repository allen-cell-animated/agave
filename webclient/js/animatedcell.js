
  var wsUri = "ws://localhost:1234";
//  var wsUri = "ws://dev-aics-dtp-001:1234";

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

  /**
   * switches the supplied element to (in)visible
   * @param element
   * @param visible
   */
  function toggleDivVisibility(element, visible) {
      element.style.visibility = (visible ? "visible" : "hidden");
  }
  //msgtype: not used yet
  //msgcontent: modelview matrix
  //(structure_)visibility: modeled channel on/off (currently 6 channels)
  //observed_visibility: observed channel on/0ff (currently 6 channels)
  //visibility_mask: used for temporally storing the visibility selection
  //sliderset: percentage of visibility for observed data [0,1], for 6 channels currently
  //var messageobj = {msgtype: 0, msgcontent:null, visibility:null, sliderset:null, observed:null};// , modeled:null, observed:null};

  /**
   * object for storing the information for the server request-
   *
   */
  function messageformat (){
      this.msgtype = 0;            // 0 = for image requests, 1 = for file structure requests
      this.mode = 1;            //switches between render modes on the server: 0 = animated cell, 1 = render 1 cell, 2 = render 2 cells
      this.msgcontent = null;  //stores the modelview matrix
      this.visibility = null; // used for the visibility of channels in the animated cell data set (working but not actually used currently)
      this.sliderset = null;  //stores slider values (not used anymore but in theory still functional
      this.observed = [1,0,0,0,0,0,0]; //booleans for observed channels
      this.modeled = [0,0,0,0,0,0]; //booleans for modeled channels
      this.datatype = ["Interphase_5cells", "Mitotic_2cells"]; //storing the data type. the second element is used when 2 datasets should be morphed or overlaid
      this.datachannel = ["20161216_C02_005_6", "20160705_S03_058_7"];  //storing the channel type. the second element is used when 2 datasets should be morphed or overlaid
      this.animationstate = 1; //value between 0 and 1 that represents the crossfade state in tab2, and the current state of an animation, e.g., in tab3
  };
  //if true requests one image for the dataset[0] and one image for [1]
  var splitscreen = true;
  var messageobj;  //used for sending & receiving image requests in socket connections #1
  //var messageobj2; //used for sending & receiving image requests in socket connections #2
  var jsonmessage; //this is the message object that we use for sending & receiving all non-binary requests

  var structure_visibility = {0:true, 1:true, 2:true, 3:true, 4:false, 5:false};
  var observed_visibility = {0:false, 1:false, 2:false, 3:false, 4:false, 5:false};
  var visibility_mask = {0:false, 1:false, 2:false, 3:false, 4:false, 5:false};
  var slider_settings = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0};
  var current_slider = 0;
  var slider;
  var slider_drag = false;

  var jsonfilestruct = {};
  var cellselection = 0;

  var resetChannelSelectors = false;

var binarysock, jsonsock;

  /**
   *
   */
  function init()
  {
    binarysocket0 = new WebSocket(wsUri); //handles requests for image streaming target #1
    binarysocket1 = new WebSocket(wsUri); //handles requests for image streaming target #1
    jsonsocket0 = new WebSocket(wsUri); //handles requests for image streaming target #1

      messageobj = new messageformat();
      //messageobj2 = new messageformat();

      jsonmessage = new messageformat();
      binarysock = new binarysocket(0);
      jsonsock = new jsonsocket();


      binarysocket0.binaryType = "arraybuffer";

      //socket connection for image stream #1
      binarysocket0.onopen = binarysock.open;
      binarysocket0.onclose = binarysock.close;
      binarysocket0.onmessage = binarysock.message0; //linked to message0
      binarysocket0.onerror = binarysock.error;

      binarysocket1.binaryType = "arraybuffer";
      //socket connection for image stream #2
      binarysocket1.onopen = binarysock.open;
      binarysocket1.onclose = binarysock.close;
      binarysocket1.onmessage = binarysock.message1; //linked to message1
      binarysocket1.onerror = binarysock.error;

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
      var streamimg2 = document.getElementById("imageB");

      toggleDivVisibility(streamimg2, false);
      toggleDivVisibility(streamimg1, true);

  }



  /**
   * socket that exclusively receives binary data for streaming jpg images
   * @param channelnumber = 0 or 1 for left or right image => currently message0 or message1 are used since channelnumber cannot always be set via the constructor for some reason
   */
  function binarysocket(channelnumber = 0) {
    this.channelnum = channelnumber;
    this.open = function (evt) {
        //send the initial camera & data query upon opening the connection
        messageobj.msgcontent = modelView;
        messageobj.visibility = structure_visibility;
        messageobj.sliderset = slider_settings;
        triggerUpdate(messageobj);
        //binarysocket0.send("req_image");
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

    };
    this.message1 = function (evt) {
        var bytes = new Uint8Array(evt.data),
            binary = "",
            len = bytes.byteLength,
            i;

        for (i=0; i<len; i++)
            binary += String.fromCharCode(bytes[i]);

        //console.log("msg1 received");
        screenImage.set(binary, 1);

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
        jsonmessage.msgtype = 1; //tells the server that we want the file structure
        triggerUpdate(jsonmessage, 2);

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

  /**
   * requests an image (or other information) from the server (with the currently specified parametrization)
   * two messages are sent when splitscreen mode is selected
   * @param msgobj contains the json message object that will be parsed by the server
   * @param connectionslot: 0 for image requests, 2 for other json commands
   */
  function triggerUpdate(msgobj, connectionslot = 0)
    {
        //lastmsg=msgobj;
        //console.log(lastmsg);

        var message0 = JSON.stringify(msgobj);
        switch (connectionslot)
        {
            case 0:
                //requesting only one image stream
                if(!splitscreen)
                {
                    binarysocket0.send(message0);
                    break;
                }
                //requesting two image streams
                else
                {
                    //splitting the message into one for each data set
                    binarysocket0.send(message0);

                    if (binarysocket1.readyState === 1) {
                      //deep copy of message object
                      var newmsg = JSON.parse(message0);
                      //switch first & second entry in data type & channel
                      newmsg.datatype[0] = msgobj.datatype[1];
                      newmsg.datatype[1] = msgobj.datatype[0];
                      newmsg.datachannel[0] = msgobj.datachannel[1];
                      newmsg.datachannel[1] = msgobj.datachannel[0];

                      var message1 = JSON.stringify(newmsg);
                      binarysocket1.send(message1);
                    }
                    break;
                }
            //case 1:

            case 2:
                jsonsocket0.send(message0);
                break;
        }
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
