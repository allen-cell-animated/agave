import ws4py
from ws4py.client.threadedclient import WebSocketClient
import io
import math
from PIL import Image
from commandbuffer import CommandBuffer

import numpy
from aicsimage.io.tifReader import TifReader
from aicsimage.io.omeTifWriter import OmeTifWriter
from aicsimage.io.omexml import OMEXML
from aicsimage.io.omexml import qn

import numpy
from aicsimage.io.tifReader import TifReader
from aicsimage.io.omeTifWriter import OmeTifWriter

OUTROOT = '\\\\allen\\aics\\animated-cell\\Dan\\2018-02-14_dan_vday_mitosis\\timelapse_wt2_s2\\'
CHNAMES = [
    'dna',
    'fibrillarin',
    'lamin_b1',
    'tom20'
]
INFILES = [
    'prediction_dna.tiff',
    'prediction_fibrillarin.tiff',
    'prediction_lamin_b1.tiff',
    'prediction_tom20.tiff'
]


def convert_tiff_to_ome_tiff_1ch(filepathin, filepathout):
    image = TifReader(filepathin).load()
    image = image.transpose([1,0,2,3])
    # normalizes data in range 0 - uint16max
    image = image.clip(min=0.0)
    image = image / image.max()
    image = 65535 * image
    # convert float to uint16
    image = image.astype(numpy.uint16)
    with OmeTifWriter(file_path=filepathout, overwrite_file=True) as writer:
        writer.save(image, channel_names=['dna'], pixels_physical_size=[0.290, 0.290, 0.290])


def convert_combined():
    inroot = '\\\\allen\\aics\\modeling\\cheko\\for_others\\2018-02-14_dan_vday_mitosis\\timelapse_wt2_s2\\'
    for j in range(0, 20):
        finalimage = None
        for i in range(0, len(INFILES)):
            infilepath = inroot + str(j).zfill(2) + '\\' + INFILES[i]
            image = TifReader(infilepath).load()
            image = image.transpose([1,0,2,3])
            # normalizes data in range 0 - uint16max
            image = image.clip(min=0.0)
            image = image / image.max()
            image = 65535 * image
            # convert float to uint16
            image = image.astype(numpy.uint16)

            # axis=2 is the C axis
            if finalimage is None:
                finalimage = [image]
            else:
                finalimage = numpy.append(finalimage, [image], axis=2)



        with OmeTifWriter(file_path=OUTROOT + 'combined_frame_' + str(j).zfill(2) + '.ome.tiff', overwrite_file=True) as writer:
            writer.save(finalimage, channel_names=CHNAMES, pixels_physical_size=[0.290, 0.290, 0.290])


def convertFiles():
    inroot = '\\\\allen\\aics\\modeling\\cheko\\for_others\\2018-02-14_dan_vday_mitosis\\timelapse_wt2_s2\\'
    # 00 .. 19
    for i in range(0, len(INFILES)):
        for j in range(0, 20):
            infilepath = inroot + str(j).zfill(2) + '\\' + INFILES[i]
            outfilepath = OUTROOT + CHNAMES[i] + '_frame_' + str(j).zfill(2) + '.ome.tiff'
            convert_tiff_to_ome_tiff_1ch(infilepath, outfilepath)

N = 0
class DummyClient(WebSocketClient):
    def opened(self):
        print("opened up")

        # 00 .. 19
        # for j in range(0, len(CHNAMES)):
        #     for i in range(0, 20):
        #         outfilepath = OUTROOT + CHNAMES[j] + '_frame_' + str(i).zfill(2) + '.ome.tiff'
        #
        #         cb = CommandBuffer()
        #         cb.add_command("LOAD_OME_TIF", outfilepath)
        #
        #         # render
        #         cb.add_command("SET_RESOLUTION", 1024, 1024)
        #         cb.add_command("RENDER_ITERATIONS", 64)
        #         cb.add_command("FRAME_SCENE");
        #         cb.add_command("MAT_DIFFUSE", 0, 1.0, 0.0, 1.0, 1.0);
        #         cb.add_command("MAT_SPECULAR", 0, 1.0, 1.0, 1.0, 0.0);
        #         cb.add_command("MAT_EMISSIVE", 0, 0.0, 0.0, 0.0, 0.0);
        #         cb.add_command("MAT_GLOSSINESS", 0, 30.0);
        #         cb.add_command("APERTURE", 0.0);
        #         cb.add_command("EXPOSURE", 0.75);
        #         cb.add_command("DENSITY", 100.0)
        #         buf = cb.make_buffer()
        #         self.send(buf, True)

        for i in range(0, 20):
            outfilepath = OUTROOT + 'combined_frame_' + str(i).zfill(2) + '.ome.tiff'

            cb = CommandBuffer()
            cb.add_command("LOAD_OME_TIF", outfilepath)
            cb.add_command("SET_RESOLUTION", 1024, 768)
            cb.add_command("RENDER_ITERATIONS", 256)

            # shiny
            # cb.add_command("EXPOSURE", 0.75)
            # cb.add_command("DENSITY", 17.858)
            # cb.add_command("APERTURE", 0.0)
            # cb.add_command("EYE", 0.696401, -0.271581, 0.556592)
            # cb.add_command("TARGET", 0.573675, 0.328722, 0.0643632)
            # cb.add_command("UP", 0.184778, 0.645456, 0.741107)
            # cb.add_command("FOV_Y", 55)
            # cb.add_command("ENABLE_CHANNEL", 0, 1)
            # cb.add_command("MAT_DIFFUSE", 0, 1, 0.227451, 0.537255, 1.0)
            # cb.add_command("MAT_SPECULAR", 0, 0.996078, 0.996078, 0.996078, 0.0)
            # cb.add_command("MAT_EMISSIVE", 0, 0, 0, 0, 0.0)
            # cb.add_command("MAT_GLOSSINESS", 0, 91.6667)
            # cb.add_command("SET_WINDOW_LEVEL", 0, 0.4784, 0.2964)
            # cb.add_command("ENABLE_CHANNEL", 1, 1)
            # cb.add_command("MAT_DIFFUSE", 1, 0, 0.291844, 1, 1.0)
            # cb.add_command("MAT_SPECULAR", 1, 0.552941, 0.980392, 0.933333, 0.0)
            # cb.add_command("MAT_EMISSIVE", 1, 0, 0, 0, 0.0)
            # cb.add_command("MAT_GLOSSINESS", 1, 65.1961)
            # cb.add_command("SET_WINDOW_LEVEL", 1, 0.6856, 0.3588)
            # cb.add_command("ENABLE_CHANNEL", 2, 1)
            # cb.add_command("MAT_DIFFUSE", 2, 0.462745, 0.568627, 0.356863, 1.0)
            # cb.add_command("MAT_SPECULAR", 2, 0, 0, 0, 0.0)
            # cb.add_command("MAT_EMISSIVE", 2, 0, 0, 0, 0.0)
            # cb.add_command("MAT_GLOSSINESS", 2, 0)
            # cb.add_command("SET_WINDOW_LEVEL", 2, 0.643137, 0.329412)
            # cb.add_command("ENABLE_CHANNEL", 3, 0)
            # cb.add_command("MAT_DIFFUSE", 3, 1, 0, 0.875334, 1.0)
            # cb.add_command("MAT_SPECULAR", 3, 0, 0, 0, 0.0)
            # cb.add_command("MAT_EMISSIVE", 3, 0, 0, 0, 0.0)
            # cb.add_command("MAT_GLOSSINESS", 3, 0)
            # cb.add_command("SET_WINDOW_LEVEL", 3, 0.588235, 0.301961)

            # test clipping
            cb.add_command("SET_CLIP_REGION", 0, 1, 0, 1, 0, 0.64)
            cb.add_command("EYE", 0.52469, -0.338348, 0.779557)
            cb.add_command("TARGET", 0.485269, 0.325739, 0.0273235)
            cb.add_command("UP", 0.0346096, 0.750108, 0.660407)
            cb.add_command("FOV_Y", 55)
            cb.add_command("EXPOSURE", 0.7655)
            cb.add_command("DENSITY", 12.8009)
            cb.add_command("APERTURE", 0)
            cb.add_command("ENABLE_CHANNEL", 0, 1)
            cb.add_command("MAT_DIFFUSE", 0, 1, 1, 1, 1.0)
            cb.add_command("MAT_SPECULAR", 0, 1, 1, 1, 0.0)
            cb.add_command("MAT_EMISSIVE", 0, 0, 0, 0, 0.0)
            cb.add_command("MAT_GLOSSINESS", 0, 100)
            cb.add_command("SET_WINDOW_LEVEL", 0, 0.4156, 0.2355)
            cb.add_command("ENABLE_CHANNEL", 1, 1)
            cb.add_command("MAT_DIFFUSE", 1, 1, 1, 0, 1.0)
            cb.add_command("MAT_SPECULAR", 1, 1, 1, 0, 0.0)
            cb.add_command("MAT_EMISSIVE", 1, 0, 0, 0, 0.0)
            cb.add_command("MAT_GLOSSINESS", 1, 100)
            cb.add_command("SET_WINDOW_LEVEL", 1, 0.694118, 0.358824)
            cb.add_command("ENABLE_CHANNEL", 2, 1)
            cb.add_command("MAT_DIFFUSE", 2, 0.333333, 1, 1, 1.0)
            cb.add_command("MAT_SPECULAR", 2, 0, 0, 0, 0.0)
            cb.add_command("MAT_EMISSIVE", 2, 0, 0, 0, 0.0)
            cb.add_command("MAT_GLOSSINESS", 2, 0)
            cb.add_command("SET_WINDOW_LEVEL", 2, 0.6784, 0.3398)
            cb.add_command("ENABLE_CHANNEL", 3, 1)
            cb.add_command("MAT_DIFFUSE", 3, 0.45098, 0.364706, 0.4, 1.0)
            cb.add_command("MAT_SPECULAR", 3, 0, 0, 0, 0.0)
            cb.add_command("MAT_EMISSIVE", 3, 0, 0, 0, 0.0)
            cb.add_command("MAT_GLOSSINESS", 3, 0)
            cb.add_command("SET_WINDOW_LEVEL", 3, 0.5725, 0.2833)
            cb.add_command("SKYLIGHT_TOP_COLOR", 1, 0, 0)
            cb.add_command("SKYLIGHT_MIDDLE_COLOR", 1, 1, 1)
            cb.add_command("SKYLIGHT_BOTTOM_COLOR", 0, 0, 1)
            cb.add_command("LIGHT_POS", 0, 1, 0, 0)
            cb.add_command("LIGHT_COLOR", 0, 10, 10, 10)
            cb.add_command("LIGHT_SIZE", 0, 1, 1)

            # orig
            # cb.add_command("EYE", 0.975529, -0.803002, 0.953116)
            # cb.add_command("TARGET", 0.5, 0.333333, 0.0952381)
            # cb.add_command("UP", 0.366998, 0.65342, 0.662083)
            # cb.add_command("FOV_Y", 18.2623)
            # cb.add_command("EXPOSURE", 0.6552)
            # cb.add_command("DENSITY", 17.6008)
            # cb.add_command("APERTURE", 0.05)
            # cb.add_command("ENABLE_CHANNEL", 0, 1)
            # cb.add_command("MAT_DIFFUSE", 0, 1, 0.345098, 0.937255, 1.0)
            # cb.add_command("MAT_SPECULAR", 0, 0, 0, 0, 0.0)
            # cb.add_command("MAT_EMISSIVE", 0, 0, 0, 0, 0.0)
            # cb.add_command("MAT_GLOSSINESS", 0, 0)
            # cb.add_command("SET_WINDOW_LEVEL", 0, 0.4156, 0.266)
            # cb.add_command("ENABLE_CHANNEL", 1, 1)
            # cb.add_command("MAT_DIFFUSE", 1, 0, 0.291844, 1, 1.0)
            # cb.add_command("MAT_SPECULAR", 1, 0, 0, 0, 0.0)
            # cb.add_command("MAT_EMISSIVE", 1, 0, 0, 0, 0.0)
            # cb.add_command("MAT_GLOSSINESS", 1, 0)
            # cb.add_command("SET_WINDOW_LEVEL", 1, 0.694118, 0.358824)
            # cb.add_command("ENABLE_CHANNEL", 2, 1)
            # cb.add_command("MAT_DIFFUSE", 2, 0.356863, 0.470588, 0.341176, 1.0)
            # cb.add_command("MAT_SPECULAR", 2, 0, 0, 0, 0.0)
            # cb.add_command("MAT_EMISSIVE", 2, 0, 0, 0, 0.0)
            # cb.add_command("MAT_GLOSSINESS", 2, 0)
            # cb.add_command("SET_WINDOW_LEVEL", 2, 0.678431, 0.343137)
            # cb.add_command("ENABLE_CHANNEL", 3, 0)
            # cb.add_command("MAT_DIFFUSE", 3, 1, 0, 0.875334, 1.0)
            # cb.add_command("MAT_SPECULAR", 3, 0, 0, 0, 0.0)
            # cb.add_command("MAT_EMISSIVE", 3, 0, 0, 0, 0.0)
            # cb.add_command("MAT_GLOSSINESS", 3, 0)
            # cb.add_command("SET_WINDOW_LEVEL", 3, 0.572549, 0.294118)
            buf = cb.make_buffer()
            self.send(buf, True)

    def closed(self, code, reason=None):
        print("Closed down", code, reason)

    def received_message(self, m):
        if m.is_binary:
            global N
            im = Image.open(io.BytesIO(m.data))
            im.save(OUTROOT + "Test000" + "_" + str(N%20).zfill(2) + ".png")
            N=N+1
        else:
            print(m)
            if len(m) == 175:
                self.close(reason='Bye bye')

if __name__ == '__main__':
    try:
        # convert_combined()
        # convertFiles()
        ws = DummyClient('ws://localhost:1235/', protocols=['http-only', 'chat'])
        ws.connect()
        ws.run_forever()
    except KeyboardInterrupt:
        print("keyboard")
        ws.close()

