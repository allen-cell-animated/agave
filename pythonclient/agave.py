# require pillow, numpy, ws4py
from ws4py.client.threadedclient import WebSocketClient
import copy
import io
import json
import math
import numpy
import queue
from PIL import Image
from commandbuffer import CommandBuffer
from collections import deque
from typing import List


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
        self.messages = queue.Queue()

    def load_image(self, image_path, onLoaded=None):
        self.get_info(image_path, callback=onLoaded)

    def opened(self):
        print("opened up")
        if self.onOpened:
            self.onOpened()

    def closed(self, code, reason=None):
        print("Closed down", code, reason)
        if self.onClose:
            self.onClose()

    def wait_for_image(self):
        while True:
            m = self.receive()
            if m is not None:
                if m.is_binary:
                    return io.BytesIO(m.data)
                else:
                    print("Non binary ws message returned")
            else:
                break
        return None

    def wait_for_json(self):
        while True:
            m = self.receive()
            if m is not None:
                if not m.is_binary:
                    return json.loads(m.data)
            print("binary ws message returned")
            break
        return None

    def received_message(self, m):
        self.messages.put(copy.deepcopy(m))

    def closed(self, code, reason=None):
        """
        Puts a :exc:`StopIteration` as a message into the
        `messages` queue.
        """
        # When the connection is closed, put a StopIteration
        # on the message queue to signal there's nothing left
        # to wait for
        self.messages.put(StopIteration)

    def receive(self, block=True):
        """
        Returns messages that were stored into the
        `messages` queue and returns `None` when the
        websocket is terminated or closed.
        `block` is passed though the gevent queue `.get()` method, which if
        True will block until an item in the queue is available. Set this to
        False if you just want to check the queue, which will raise an
        Empty exception you need to handle if there is no message to return.
        """
        # If the websocket was terminated and there are no messages
        # left in the queue, return None immediately otherwise the client
        # will block forever
        if self.terminated and self.messages.empty():
            return None
        message = self.messages.get(block=block)
        if message is StopIteration:
            return None
        return message


class AgaveRenderer:
    def __init__(self) -> None:
        self.cb = CommandBuffer()
        self.session_name = ""
        self.ws = AgaveClient("ws://localhost:1235/", protocols=["http-only", "chat"])
        # self.ws.onOpened = self.onOpen
        self.ws.connect()
        # self.ws.run_forever()
        # except KeyboardInterrupt:
        #     print("keyboard")
        #     ws.close()

    def session(self, name: str):
        # 0
        self.cb.add_command("SESSION", name)
        self.session_name = name

    def asset_path(self, name: str):
        # 1
        self.cb.add_command("ASSET_PATH", name)

    def load_ome_tif(self, name: str):
        # 2
        self.cb.add_command("LOAD_OME_TIF", name)

    def eye(self, x: float, y: float, z: float):
        # 3
        self.cb.add_command("EYE", x, y, z)

    def target(self, x: float, y: float, z: float):
        # 4
        self.cb.add_command("TARGET", x, y, z)

    def up(self, x: float, y: float, z: float):
        # 5
        self.cb.add_command("UP", x, y, z)

    def aperture(self, x: float):
        # 6
        self.cb.add_command("APERTURE", x)

    def camera_projection(self, projection_type: int, x: float):
        # 7
        self.cb.add_command("CAMERA_PROJECTION", projection_type, x)

    def focaldist(self, x: float):
        # 8
        self.cb.add_command("FOCALDIST", x)

    def exposure(self, x: float):
        # 9
        self.cb.add_command("EXPOSURE", x)

    def mat_diffuse(self, channel: int, r: float, g: float, b: float, a: float):
        # 10
        self.cb.add_command("MAT_DIFFUSE", channel, r, g, b, a)

    def mat_specular(self, channel: int, r: float, g: float, b: float, a: float):
        # 11
        self.cb.add_command("MAT_SPECULAR", channel, r, g, b, a)

    def mat_emissive(self, channel: int, r: float, g: float, b: float, a: float):
        # 12
        self.cb.add_command("MAT_EMISSIVE", channel, r, g, b, a)

    def render_iterations(self, x: int):
        # 13
        self.cb.add_command("RENDER_ITERATIONS", x)

    def stream_mode(self, x: int):
        # 14
        self.cb.add_command("STREAM_MODE", x)

    def redraw(self):
        # 15
        # issue command buffer
        self.cb.add_command("REDRAW")
        buf = self.cb.make_buffer()
        # TODO ENSURE CONNECTED
        self.ws.send(buf, True)
        #  and then WAIT for render to be completed
        binarydata = self.ws.wait_for_image()
        # and save image
        im = Image.open(binarydata)
        print(self.session_name)
        im.save(self.session_name)
        # ready for next frame
        self.session_name = ""
        self.cb = CommandBuffer()

    def set_resolution(self, x: int, y: int):
        # 16
        self.cb.add_command("SET_RESOLUTION", x, y)

    def density(self, x: float):
        # 17
        self.cb.add_command("DENSITY", x)

    def frame_scene(self):
        # 18
        self.cb.add_command("FRAME_SCENE")

    def mat_glossiness(self, channel: int, glossiness: float):
        # 19
        self.cb.add_command("MAT_GLOSSINESS", channel, glossiness)

    def enable_channel(self, channel: int, enabled: int):
        # 20
        self.cb.add_command("ENABLE_CHANNEL", channel, enabled)

    def set_window_level(self, channel: int, window: float, level: float):
        # 21
        self.cb.add_command("SET_WINDOW_LEVEL", channel, window, level)

    def orbit_camera(self, theta: float, phi: float):
        # 22
        self.cb.add_command("ORBIT_CAMERA", theta, phi)

    def skylight_top_color(self, r: float, g: float, b: float):
        # 23
        self.cb.add_command("SKYLIGHT_TOP_COLOR", r, g, b)

    def skylight_middle_color(self, r: float, g: float, b: float):
        # 24
        self.cb.add_command("SKYLIGHT_MIDDLE_COLOR", r, g, b)

    def skylight_bottom_color(self, r: float, g: float, b: float):
        # 25
        self.cb.add_command("SKYLIGHT_BOTTOM_COLOR", r, g, b)

    def light_pos(self, index: int, r: float, theta: float, phi: float):
        # 26
        self.cb.add_command("LIGHT_POS", index, r, theta, phi)

    def light_color(self, index: int, r: float, g: float, b: float):
        # 27
        self.cb.add_command("LIGHT_COLOR", index, r, g, b)

    def light_size(self, index: int, x: float, y: float):
        # 28
        self.cb.add_command("LIGHT_SIZE", index, x, y)

    def set_clip_region(
        self,
        minx: float,
        maxx: float,
        miny: float,
        maxy: float,
        minz: float,
        maxz: float,
    ):
        # 29
        self.cb.add_command("SET_CLIP_REGION", minx, maxx, miny, maxy, minz, maxz)

    def set_voxel_scale(self, x: float, y: float, z: float):
        # 30
        self.cb.add_command("SET_VOXEL_SCALE", x, y, z)

    def auto_threshold(self, channel: int, method: int):
        # 31
        self.cb.add_command("AUTO_THRESHOLD", channel, method)

    def set_percentile_threshold(self, channel: int, pct_low: float, pct_high: float):
        # 32
        self.cb.add_command("SET_PERCENTILE_THRESHOLD", channel, pct_low, pct_high)

    def mat_opacity(self, channel: int, opacity: float):
        # 33
        self.cb.add_command("MAT_OPACITY", channel, opacity)

    def set_primary_ray_step_size(self, step_size: float):
        # 34
        self.cb.add_command("SET_PRIMARY_RAY_STEP_SIZE", step_size)

    def set_secondary_ray_step_size(self, step_size: float):
        # 35
        self.cb.add_command("SET_SECONDARY_RAY_STEP_SIZE", step_size)

    def background_color(self, r: float, g: float, b: float):
        # 36
        self.cb.add_command("BACKGROUND_COLOR", r, g, b)

    def set_isovalue_threshold(self, channel: int, isovalue: float, isorange: float):
        # 37
        self.cb.add_command("SET_ISOVALUE_THRESHOLD", channel, isovalue, isorange)

    def set_control_points(self, channel: int, data: List[float]):
        # 38
        self.cb.add_command("SET_CONTROL_POINTS", channel, data)

    def load_volume_from_file(self, path: str, scene: int, time: int):
        # 39
        self.cb.add_command("LOAD_VOLUME_FROM_FILE", path, scene, time)

    def set_time(self, time: int):
        # 40
        self.cb.add_command("SET_TIME", time)

    def batch_render_turntable(
        self, number_of_frames=90, direction=1, output_name="frame", first_frame=0
    ):
        # direction must be +/-1
        if direction != 1 and direction != -1:
            return

        # then orbit the camera parametrically
        for i in range(0, number_of_frames):
            self.session(f"{output_name}_{i+first_frame}.png")
            self.redraw()
            # first frame gets zero orbit, then onward:
            self.orbit_camera(0.0, direction * (360.0 / float(number_of_frames)))

    def batch_render_rocker(
        self,
        number_of_frames=90,
        angle=30,
        direction=1,
        output_name="frame",
        first_frame=0,
    ):
        # direction must be +/-1
        if direction != 1 and direction != -1:
            return

        # orbit the camera parametrically
        angledelta = 4.0 * float(angle) / float(number_of_frames)
        for i in range(0, number_of_frames):
            quadrant = (i * 4) // number_of_frames
            quadrantdirection = 1 if quadrant == 0 or quadrant == 3 else -1
            self.session(f"{output_name}_{i+first_frame}.png")
            self.redraw()
            # first frame gets zero orbit, then onward:
            self.orbit_camera(0.0, angledelta * direction * quadrantdirection)
