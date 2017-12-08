import ws4py
from ws4py.client.threadedclient import WebSocketClient
import io
import math
from PIL import Image

from commandbuffer import CommandBuffer

N = 0
class DummyClient(WebSocketClient):
    def opened(self):
        print("opened up")
        rot = 180
        for i in range(0, rot):
            cb = CommandBuffer()
            cb.add_command("CHANNEL", 1)
            cb.add_command("APERTURE", 0.01)
            cb.add_command("EXPOSURE", 0.45)
            cb.add_command("DENSITY", 37.7)
            cb.add_command("SET_RESOLUTION", 1024, 1024)
            cb.add_command("RENDER_ITERATIONS", 64)
            cb.add_command("MAT_DIFFUSE", (float(i))/float(rot-1), (float(rot-1)-float(i))/float(rot-1), 0.0, 1.0)
            cb.add_command("MAT_SPECULAR", 3.0, 3.0, 3.0, 0.0)
            cb.add_command("MAT_EMISSIVE", 0.0, 0.0, 0.0, 0.0)
            cb.add_command("EYE", 0.5 + 1.0*math.sin(2.0*float(i)*3.14159265/180.0), 0.408, 0.145 + 1.0*math.cos(2.0*float(i)*3.14159265/180.0))
            cb.add_command("TARGET", 0.5, 0.408, 0.145)
            buf = cb.make_buffer()
            self.send(buf, True)
            # cb2 = CommandBuffer()
            # cb2.add_command("CHANNEL", 1)
            # buf = cb2.make_buffer()
            # self.send(buf, True)

    def closed(self, code, reason=None):
        print("Closed down", code, reason)

    def received_message(self, m):
        if m.is_binary:
            global N
            i = Image.open(io.BytesIO(m.data))
            i.save("//allen/aics/animated-cell/Dan/output/test2/TEST_1_"+str(N)+".png")
            N=N+1

        else:
            print(m)
            if len(m) == 175:
                self.close(reason='Bye bye')

if __name__ == '__main__':
    try:
        ws = DummyClient('ws://localhost:1234/', protocols=['http-only', 'chat'])
        ws.connect()
        ws.run_forever()


    except KeyboardInterrupt:
        ws.close()
