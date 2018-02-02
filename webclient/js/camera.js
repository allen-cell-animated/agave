/**
 * Created by jsorger on 25.07.2017.
 */


var lastClientX, lastClientY;

document.documentElement.addEventListener('mouseup', function(e){
    if (dragFlag) {
        dragFlag = 0;

        let cb = new commandBuffer();
        cb.addCommand("STREAM_MODE", _stream_mode ? 1 : 0);
        cb.addCommand("REDRAW");
        flushCommandBuffer(cb);
    }
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

            messageobj.msgcontent = null;
            messageobj.visibility = structure_visibility;
            messageobj.sliderset = slider_settings;
            triggerUpdate(messageobj);
        }

        if(dragFlag)
        {

            cb = new commandBuffer();
            cb.addCommand("STREAM_MODE", 0);
            cb.addCommand("EYE", gCamera.position.x, gCamera.position.y, gCamera.position.z);
            cb.addCommand("TARGET", gControls.target.x, gControls.target.y, gControls.target.z);
            cb.addCommand("UP", gCamera.up.x, gCamera.up.y, gCamera.up.z);
            cb.addCommand("REDRAW");
            flushCommandBuffer(cb);

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
                //tabs[i] = document.createElement("img");
                tabs[i].ondragstart = this.dragstart;
                tabs[i].onmousedown = this.mousedown;
                tabs[i].onmouseup = this.mouseup;
                tabs[i].onmousemove = this.mousemove;
                tabs[i].onmousewheel = this.mousewheel;
//                tabs[i].src = "data:image/png;base64,"+window.btoa( binary );
                tabs[i].src = binary;

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
}

function MouseWheelHandler(e)
{
    cb = new commandBuffer();
    cb.addCommand("EYE", gCamera.position.x, gCamera.position.y, gCamera.position.z);
    cb.addCommand("TARGET", gControls.target.x, gControls.target.y, gControls.target.z);
    cb.addCommand("UP", gCamera.up.x, gCamera.up.y, gCamera.up.z);
    cb.addCommand("REDRAW");
    flushCommandBuffer(cb);
}

function setVisibility(visi)
{
    messageobj.visibility = visi;
}

function setSliders(slset)
{
    messageobj.sliderset = slset;
}
