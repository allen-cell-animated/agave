# require pillow, numpy, ws4py
from ws4py.client.threadedclient import WebSocketClient
import io
import json
import math
import numpy
from PIL import Image
from commandbuffer import CommandBuffer
from collections import deque


def lerp(startframe, endframe, startval, endval):
    x = numpy.linspace(
        startframe, endframe, num=endframe - startframe + 1, endpoint=True
    )
    y = startval + (endval - startval) * (x - startframe) / (endframe - startframe)
    print(y)


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = numpy.asarray(axis)
    axis = axis / math.sqrt(numpy.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return numpy.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def rotate_vec(v, axis, angle):
    return numpy.dot(rotation_matrix(axis, angle), v)


def vec_sub(v1, v2):
    return [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]]


def vec_add(v1, v2):
    return [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]]


def vec_normalize(v):
    vmag = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return [v[0] / vmag, v[1] / vmag, v[2] / vmag]


def vec_cross(v1, v2):
    c = [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ]
    return c


def get_vertical_axis(lookdir, up):
    eyeDirection = vec_normalize(lookdir)
    objectUpDirection = vec_normalize(up)
    objectSidewaysDirection = vec_normalize(vec_cross(objectUpDirection, eyeDirection))
    axis = vec_normalize(vec_cross(objectSidewaysDirection, lookdir))
    return axis


# assumptions: every commandbuffer send should result in one image.
# also, they arrive in the order the buffers were sent.
class AgaveClient(WebSocketClient):
    def __init__(self, *args, **kwargs):
        super(AgaveClient, self).__init__(*args, **kwargs)
        self.onOpened = None
        self.onClose = None
        self.requested_frame = 0
        self.queue = deque([])

    def load_image(self, image_path, onLoaded=None):
        self.get_info(image_path, callback=onLoaded)

    def render_frame(
        self, command_list, number=None, output_name="frame", callback=None
    ):
        cb = CommandBuffer(command_list)
        if number is not None:
            out = output_name + "_" + str(int(number)).zfill(4) + ".png"
        else:
            out = output_name + ".png"
        self.push_request(cb, out, callback=callback)

    def render_sequence(
        self, sequence, output_name="frame", first_frame=0, callback=None
    ):
        # sequence is a list of lists of commands.
        # each list describes one frame
        for i, cmds in enumerate(sequence):
            self.render_frame(
                command_list=cmds,
                number=i + first_frame,
                output_name=output_name,
                callback=callback,
            )

    def render_turntable(
        self,
        command_list,
        number_of_frames=90,
        direction=1,
        output_name="frame",
        first_frame=0,
        callback=None,
    ):
        # direction must be +/-1
        if direction != 1 and direction != -1:
            return

        # issue the first command buffer.
        self.render_frame(
            command_list, number=first_frame, output_name=output_name, callback=callback
        )

        # then orbit the camera parametrically
        for i in range(1, number_of_frames):
            self.render_frame(
                [("ORBIT_CAMERA", 0.0, direction * (360.0 / float(number_of_frames)))],
                number=i + first_frame,
                output_name=output_name,
                callback=callback,
            )

    def render_rocker(
        self,
        command_list,
        number_of_frames=90,
        angle=30,
        direction=1,
        output_name="frame",
        first_frame=0,
        callback=None,
    ):
        # direction must be +/-1
        if direction != 1 and direction != -1:
            return

        # issue the first command buffer.
        self.render_frame(
            command_list, number=first_frame, output_name=output_name, callback=callback
        )
        # then orbit the camera parametrically
        angledelta = 4.0 * float(angle) / float(number_of_frames)
        for i in range(1, number_of_frames):
            quadrant = (i * 4) // number_of_frames
            quadrantdirection = 1 if quadrant == 0 or quadrant == 3 else -1
            self.render_frame(
                [("ORBIT_CAMERA", 0.0, angledelta * direction * quadrantdirection)],
                number=i + first_frame,
                output_name=output_name,
                callback=callback,
            )

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
            if name != "info":
                im = Image.open(io.BytesIO(m.data))
                im.save(name)
                print("Saved frame " + str(req[0]) + " : " + name)
        else:
            # print(m)
            if len(m) == 175:
                self.close(reason="Bye bye")
            else:
                # should be json data coming from an "info" request
                self._handleInfoText(m, req)
                # return request back on queue for the actual image
                self.queue.appendleft(req)


def agaveclient(port=1235, renderfunc=None):
    if renderfunc is None:
        return
    try:
        ws = AgaveClient(
            "ws://localhost:" + str(port) + "/", protocols=["http-only", "chat"]
        )
        print("created client")

        def onOpen():
            renderfunc(ws)

        ws.onOpened = onOpen
        ws.connect()
        ws.run_forever()
    except KeyboardInterrupt:
        print("keyboard")
        ws.close()


# imgplot = plt.imshow(numpy.zeros((1024, 768)))
if __name__ == "__main__":
    try:
        ws = AgaveClient("ws://localhost:1235/", protocols=["http-only", "chat"])
        print("created client")

        def onOpen():
            print("opened connection")

        ws.onOpened = onOpen
        ws.connect()
        ws.run_forever()
    except KeyboardInterrupt:
        print("keyboard")
        ws.close()

