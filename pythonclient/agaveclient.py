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

matplotlib.use("TkAgg")


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
        self.push_request(
            cb, output_name + "_" + str(number).zfill(4) + ".png", callback=callback
        )

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
                self.close(reason="Bye bye")
            else:
                # should be json data coming from an "info" request
                self._handleInfoText(m, req)
                # return request back on queue for the actual image
                self.queue.appendleft(req)


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

