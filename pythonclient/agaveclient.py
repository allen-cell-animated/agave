from ws4py.client.threadedclient import WebSocketClient
import io
import json
import logging
import math
import numpy
from PIL import Image
from commandbuffer import CommandBuffer
from collections import deque
import matplotlib
matplotlib.use('TkAgg')


def lerp(startframe, endframe, startval, endval):
    x = numpy.linspace(startframe, endframe, num=endframe-startframe+1,
                       endpoint=True)
    y = startval + (endval-startval)*(x-startframe)/(endframe-startframe)
    print(y)


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = numpy.asarray(axis)
    axis = axis/math.sqrt(numpy.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return numpy.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                        [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                        [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def rotate_vec(v, axis, angle):
    return numpy.dot(rotation_matrix(axis, angle), v)


# assumptions: every commandbuffer send should result in one image.
# also, they arrive in the order the buffers were sent.
class AgaveClient(WebSocketClient):
    def __init__(self, *args, **kwargs):
        super(AgaveClient, self).__init__(*args, **kwargs)
        self.onOpened = None
        self.onClose = None
        self.requested_frame = 0
        self.queue = deque([])

    def render_frame(self, command_list, number=0, output_name="frame", callback=None):
        cb = CommandBuffer(command_list)
        self.push_request(cb, output_name+'_'+str(number).zfill(4)+".png", callback=callback)

    def render_sequence(self, sequence):
        # wait for each image in the sequence to be returned before
        # sending the next request
        self.sequence = sequence
        self.connect()
        self.run_forever()

    def get_info(self, filepath, callback):
        print("Get info: " + filepath)
        cb = CommandBuffer()
        cb.add_command("LOAD_OME_TIF", filepath)
        self.push_request(cb, "info", callback=callback)

    def _handleInfoText(self, m, req):
        print("Received info text: " + str(self.requested_frame))
        self.imgdata = json.loads(m.data)
        # do something special using the imgdata
        if req[2]:
            req[2](self.imgdata)

    def push_request(self, cb, data, callback=None):
        self.requested_frame = self.requested_frame + 1
        buf = cb.make_buffer()
        self.send(buf, True)
        print("Request : " + data)
        self.queue.append((self.requested_frame, data, callback))
        print("pushed request, queue: " + str(len(self.queue)))

    def opened(self):
        self.requested_frame = 0
        self.queue = deque([])
        print("opened up")
        if self.onOpened:
            self.onOpened()

    def closed(self, code, reason=None):
        print("Closed down", code, reason)
        if self.onClose:
            self.onClose()

    def received_message(self, m):
        print("Received message, queue: " + str(len(self.queue)))
        # req is (frame, data, callback)
        req = self.queue.popleft()
        print("popped request, queue: " + str(len(self.queue)))
        if m.is_binary:
            # print(req)
            # number = req[0]
            name = req[1]
            im = Image.open(io.BytesIO(m.data))
            im.save(name)
            print("Saved frame " + str(req[0]) + " : " + name)
        else:
            # print(m)
            if len(m) == 175:
                self.close(reason='Bye bye')
            else:
                # should be json data coming from an "info" request
                self._handleInfoText(m, req)
                # return request back on queue for the actual image
                self.queue.appendleft(req)


# imgplot = plt.imshow(numpy.zeros((1024, 768)))
if __name__ == '__main__':

    per_frame_commands = [
        ("LOAD_OME_TIF", "D:\\data\\april2019\\aligned_100s\\Interphase\\ACTB_36972_seg.ome.tif"),
        ("SET_RESOLUTION", 1024, 1024),
        ("SET_VOXEL_SCALE", 0.8, -0.8, 2.0),
        ("RENDER_ITERATIONS", 512),
        ("SET_CLIP_REGION", 0, 1, 0, 1, 0, 1),
        ("EYE", 0.5, -0.5, 1.39614),
        ("TARGET", 0.5, -0.5, 0.0),
        ("UP", 0.0, 1.0, 0.0),
        ("FOV_Y", 55),
        ("EXPOSURE", 0.8714),
        ("DENSITY", 100),
        ("APERTURE", 0),
        ("FOCALDIST", 0.75),
        ("ENABLE_CHANNEL", 0, 1),
        ("MAT_DIFFUSE", 0, 1, 0, 1, 1.0),
        ("MAT_SPECULAR", 0, 0, 0, 0, 0.0),
        ("MAT_EMISSIVE", 0, 0, 0, 0, 0.0),
        ("MAT_GLOSSINESS", 0, 0),
        ("SET_WINDOW_LEVEL", 0, 1, 0.758),
        ("ENABLE_CHANNEL", 1, 1),
        ("MAT_DIFFUSE", 1, 1, 1, 1, 1.0),
        ("MAT_SPECULAR", 1, 0, 0, 0, 0.0),
        ("MAT_EMISSIVE", 1, 0, 0, 0, 0.0),
        ("MAT_GLOSSINESS", 1, 0),
        # ("SET_WINDOW_LEVEL", 1, 1, 0.7366),
        ("SET_WINDOW_LEVEL", 1, 1, 0.811),
        ("ENABLE_CHANNEL", 2, 1),
        ("MAT_DIFFUSE", 2, 0, 1, 1, 1.0),
        ("MAT_SPECULAR", 2, 0, 0, 0, 0.0),
        ("MAT_EMISSIVE", 2, 0, 0, 0, 0.0),
        ("MAT_GLOSSINESS", 2, 0),
        ("SET_WINDOW_LEVEL", 2, 0.9922, 0.7704),
        ("SKYLIGHT_TOP_COLOR", 0.5, 0.5, 0.5),
        ("SKYLIGHT_MIDDLE_COLOR", 0.5, 0.5, 0.5),
        ("SKYLIGHT_BOTTOM_COLOR", 0.5, 0.5, 0.5),
        ("LIGHT_POS", 0, 10.1663, 1.1607, 0.5324),
        ("LIGHT_COLOR", 0, 122.926, 122.926, 125.999),
        ("LIGHT_SIZE", 0, 1, 1),
    ]
    per_frame_commands2 = [
        ("LOAD_OME_TIF", "D:\\data\\april2019\\aligned_100s\\Interphase\\TUBA1B_71126_raw.ome.tif"),
        ("SET_RESOLUTION", 1024, 1024),
        ("SET_VOXEL_SCALE", 0.8, -0.8, 2.0),
        ("RENDER_ITERATIONS", 512),
        ("SET_CLIP_REGION", 0, 1, 0, 1, 0, 1),
        ("EYE", 0.5, -0.5, 1.39614),
        ("TARGET", 0.5, -0.5, 0.0),
        ("UP", 0.0, 1.0, 0.0),
        ("FOV_Y", 55),
        ("EXPOSURE", 0.8714),
        ("DENSITY", 100),
        ("APERTURE", 0),
        ("FOCALDIST", 0.75),
        ("ENABLE_CHANNEL", 0, 1),
        ("MAT_DIFFUSE", 0, 1, 0, 1, 1.0),
        ("MAT_SPECULAR", 0, 0, 0, 0, 0.0),
        ("MAT_EMISSIVE", 0, 0, 0, 0, 0.0),
        ("MAT_GLOSSINESS", 0, 0),
        ("SET_WINDOW_LEVEL", 0, 1, 0.758),
        ("ENABLE_CHANNEL", 1, 1),
        ("MAT_DIFFUSE", 1, 1, 1, 1, 1.0),
        ("MAT_SPECULAR", 1, 0, 0, 0, 0.0),
        ("MAT_EMISSIVE", 1, 0, 0, 0, 0.0),
        ("MAT_GLOSSINESS", 1, 0),
        # ("SET_WINDOW_LEVEL", 1, 1, 0.7366),
        ("SET_WINDOW_LEVEL", 1, 1, 0.811),
        ("ENABLE_CHANNEL", 2, 1),
        ("MAT_DIFFUSE", 2, 0, 1, 1, 1.0),
        ("MAT_SPECULAR", 2, 0, 0, 0, 0.0),
        ("MAT_EMISSIVE", 2, 0, 0, 0, 0.0),
        ("MAT_GLOSSINESS", 2, 0),
        ("SET_WINDOW_LEVEL", 2, 0.9922, 0.7704),
        ("SKYLIGHT_TOP_COLOR", 0.5, 0.5, 0.5),
        ("SKYLIGHT_MIDDLE_COLOR", 0.5, 0.5, 0.5),
        ("SKYLIGHT_BOTTOM_COLOR", 0.5, 0.5, 0.5),
        ("LIGHT_POS", 0, 10.1663, 1.1607, 0.5324),
        ("LIGHT_COLOR", 0, 122.926, 122.926, 125.999),
        ("LIGHT_SIZE", 0, 1, 1),
    ]
    try:
        # convert_combined()
        # convertFiles()
        # ws = DummyClient('ws://dev-aics-dtp-001:1235/', protocols=['http-only', 'chat'])
        ws = AgaveClient('ws://localhost:1235/', protocols=['http-only', 'chat'])
        print("created client")

        def onGetInfo(jsondict):
            print(jsondict)

        def onOpen():
            ws.get_info("D:\\data\\april2019\\aligned_100s\\Interphase\\ACTB_36972_seg.ome.tif", onGetInfo)
            ws.render_frame(per_frame_commands, 1, "one")
            ws.render_frame(per_frame_commands2, 2, "two")

        ws.onOpened = onOpen
        ws.connect()
        ws.run_forever()

        # ws.render_sequence(sequence)
    except KeyboardInterrupt:
        print("keyboard")
        ws.close()

