"""Websocket client used by :class:`agave_pyclient.agave.AgaveRenderer`.

This module isolates the low-level ws4py-based websocket subclass from
the higher-level renderer API.
"""

import copy
import io
import json
import queue

from ws4py.client.threadedclient import WebSocketClient

__all__ = ["AgaveClient"]


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

    def received_message(self, message):
        self.messages.put(copy.deepcopy(message))

    def closed(self, code, reason=None):
        """
        Puts a :exc:`StopIteration` as a message into the
        `messages` queue.
        """
        # When the connection is closed, put a StopIteration
        # on the message queue to signal there's nothing left
        # to wait for
        self.messages.put(StopIteration)
        print("Closed down", code, reason)
        if self.onClose:
            self.onClose()

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
