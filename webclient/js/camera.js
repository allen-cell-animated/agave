/**
 * Created by jsorger on 25.07.2017.
 */


var lastClientX, lastClientY;


document.documentElement.addEventListener('mouseup', function(e){
  messageobj.mouseDeltaRotate = {x:0, y:0};

    dragFlag = 0;

    triggerUpdate(messageobj);
    triggerUpdate(messageobj);
    triggerUpdate(messageobj);
    triggerUpdate(messageobj);
    triggerUpdate(messageobj);
    triggerUpdate(messageobj);
    triggerUpdate(messageobj);
    triggerUpdate(messageobj);
    triggerUpdate(messageobj);
    triggerUpdate(messageobj);
});

/**
 * updates the camera when the mouse is moved while dragging is true.
 * a new model view matrix is calculated and sent to the server.
 *
 */
document.documentElement.addEventListener('mousemove', function(e){

    //console.log("movin!");
    if (this.mouseMoveTimer) return;

    this.mouseMoveTimer = setTimeout(function () {

        if(slider_drag)
        {
            var value = $( "#slider" ).slider( "value" );
            slider_settings[current_slider] = value / 100.0;

            messageobj.msgcontent = modelView;
            messageobj.visibility = structure_visibility;
            messageobj.sliderset = slider_settings;
            triggerUpdate(messageobj);
        }

        if(dragFlag)
        {
            var relativeX = e.clientX - initialMouseX;
            var relativeY = e.clientY - initialMouseY;

            var outX = (relativeX / img_width) * 0.2;  //1280
            var outY = (-relativeY / img_height) * 0.2;  //720

            rotationTo.x = outX;
            rotationTo.y = outY;
            rotationTo.z = 0.1;
            rotationTo.normalize();

            rotationAxis.crossVectors(rotationFrom, rotationTo);
            var angle = rotationFrom.dot(rotationTo);
            rotationDelta.x = rotationAxis.x;
            rotationDelta.y = rotationAxis.y;
            rotationDelta.z = rotationAxis.z;
            rotationDelta.w = angle;

            ///////////////
            messageobj.mouseDeltaRotate = {x:e.clientX - lastClientX, y:e.clientY - lastClientY};
            lastClientX = e.clientX;
            lastClientY = e.clientY;
            messageobj.deltaRotate = {x:rotationAxis.x, y:rotationAxis.y, z:rotationAxis.z, angle:angle};
            ///////////////

            rotation = multiplyQuat(oldRotation, rotationDelta);//.multiplyQuaternions(oldRotation, rotationDelta);

            rotation.z = -rotation.z;
            modelView.makeRotationFromQuaternion(rotation);
            modelView.setPosition(panVec);

            var matrixmode = true;

            messageobj.msgcontent = modelView;
            messageobj.visibility = structure_visibility;
            messageobj.sliderset = slider_settings;

            var message;
            if(matrixmode)
            {
                triggerUpdate(messageobj);
            }
            else
            {
                message = JSON.stringify(rotation);
                binarysocket0.send(message);
                console.warn("rotation mode is deprecated.")
            }

            rotation.z = -rotation.z;
        }
        this.mouseMoveTimer = null;

    }.bind(this), 50);
});

/**
 * this object holds the image that is received from the server
 * @type {{mouseMoveTimer: null, dragstart: screenImage.dragstart, mousedown: screenImage.mousedown, set: screenImage.set}}
 */
var screenImage = {
    mouseMoveTimer : null,
    dragstart : function (e) {
        return false;
    },
    mousedown : function (e) {

        dragFlag = 1;

        initialMouseX = e.x | e.clientX;
        initialMouseY = e.y | e.clientY;
        lastClientX = initialMouseX;
        lastClientY = initialMouseY;
        rotationFrom.x = 0;
        rotationFrom.y = 0;
        rotationFrom.z = 0.1;
        rotationFrom.normalize();
        oldRotation = rotation;
    },


    /**
     * sets the image and the events. called from the websocket "message" signal
     * @param binary
     * @param channelnumber
     */
    set : function (binary, channelnumber) {

        //get all the divs with the streamed_img tag and set the received binary data to the image's source
        var tabs;
        tabs = document.getElementsByClassName("streamed_img"+ " img" +channelnumber);

        if(tabs.length > 0)
        {
            for(var i=0; i<tabs.length; i++)
            {
                tabs[i] = document.createElement("img");
                tabs[i].ondragstart = this.dragstart;
                tabs[i].onmousedown = this.mousedown;
                tabs[i].onmouseup = this.mouseup;
                tabs[i].onmousemove = this.mousemove;
                tabs[i].onmousewheel = this.mousewheel;
                tabs[i].src = "data:image/png;base64,"+window.btoa( binary );

                img_width = tabs[i].width;
                img_height = tabs[i].height;
            }
        }
        else
        {
            //document.body.appendChild(img);
            console.warn("div 'streamed_img' not found :(");
        }
    }
};

function DragStartHandler(e)
{
    //console.log("thou shall not drag!");
    return false;
}

function MouseDownHandler(e)
{
    dragFlag = 1;

    initialMouseX = e.x | e.clientX;
    initialMouseY = e.y | e.clientY;
    lastClientX = initialMouseX;
    lastClientY = initialMouseY;

    rotationFrom.x = 0;
    rotationFrom.y = 0;
    rotationFrom.z = 0.1;
    rotationFrom.normalize();
    oldRotation = rotation;
}

function MouseWheelHandler(e)
{
    // cross-browser wheel delta
    var e = window.event || e; // old IE support
    var delta = Math.max(-1, Math.min(1, (e.wheelDelta || -e.detail)));

    //console.log(delta);
    panVec.z += delta*0.3;

    if (panVec.z < -5.0)
        panVec.z = -5.0;

    if (panVec.z > 5.0)
        panVec.z = 5.0;

    modelView.setPosition(panVec);

    messageobj.msgcontent = modelView;
    messageobj.visibility = structure_visibility;
    messageobj.sliderset = slider_settings;
    triggerUpdate(messageobj);
}

function multiplyQuat(q1, q2)
{
    var result = new THREE.Quaternion();
    result.x =  q1.x * q2.w + q1.y * q2.z - q1.z * q2.y + q1.w * q2.x;
    result.y = -q1.x * q2.z + q1.y * q2.w + q1.z * q2.x + q1.w * q2.y;
    result.z =  q1.x * q2.y - q1.y * q2.x + q1.z * q2.w + q1.w * q2.z;
    result.w = -q1.x * q2.x - q1.y * q2.y - q1.z * q2.z + q1.w * q2.w;

    return result;
}


function setModelView(mv)
{
    messageobj.msgcontent = mv;
}

function setVisibility(visi)
{
    messageobj.visibility = visi;
}

function setSliders(slset)
{
    messageobj.sliderset = slset;
}
