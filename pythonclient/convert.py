import csv
import numpy
from aicsimage.io.tifReader import TifReader
from aicsimage.io.omeTifWriter import OmeTifWriter
from aicsimage.processing import AICSImage

OUTROOT = '//allen/aics/animated-cell/Dan/2018-02-14_dan_vday_mitosis/timelapse_wt2_s2/'
CHNAMES = [
    'dna',
    'fibrillarin',
    'lamin_b1',
    'tom20',
    'brightfield'
]
INFILES = [
    'prediction_dna.tiff',
    'prediction_fibrillarin.tiff',
    'prediction_lamin_b1.tiff',
    'prediction_tom20.tiff',
    'signal.tiff'
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

def combineFiles(files, out, channel_names=None):
    finalimage = None
    for f in files:
        ai = AICSImage(f)
        # ai.data is 5d.
        image = ai.data
        if finalimage is None:
            finalimage = [image[0]]
        else:
            finalimage = numpy.append(finalimage, [image[0]], axis=1)
    print(finalimage.shape)
    finalimage = finalimage.transpose([0, 2, 1, 3, 4])
    with OmeTifWriter(file_path=out, overwrite_file=True) as writer:
        writer.save(finalimage, channel_names=channel_names, pixels_physical_size=[0.108, 0.108, 0.290])

# for derek
with open('\\\\allen\\aics\\microscopy\\UserFolders\\Derek\\2018-03-12_fake_fluorescence_images_v0\\resize_predictions.csv', newline='') as csvfile:
    spamreader = csv.DictReader(csvfile)
    i = 0
    for row in spamreader:
        # skip first?
        if i > 0:
            segfolder = '\\\\allen\\aics\\microscopy\\UserFolders\\Derek\\2018-03-12_fake_fluorescence_images_v0\\Watershed4xDownsampleOutput\\%02d\\' % i
            predfolder = '\\\\allen\\aics\\microscopy\\UserFolders\\Derek\\2018-03-12_fake_fluorescence_images_v0\\%02d\\' % i
            outfolder = '\\\\allen\\aics\\animated-cell\\Dan\\2018-03-12_fake_fluorescence_images_v0\\w4xd\\'
            channel_names = [str(j) for j in range(0, 11)]
            channel_names[7] = 'pred_memb'
            channel_names[8] = 'pred_dna'
            channel_names[9] = 'seg_memb'
            channel_names[10] = 'seg_dna'
            combineFiles([
                row['path_czi'],
                predfolder + 'prediction_63x_bf_membrane_caax_resized.tiff',
                predfolder + 'prediction_dna_extended_resized.tiff',
                segfolder + 'prediction_63x_bf_membrane_caax_resized_WatershedCellSeg.tiff',
                segfolder + 'prediction_dna_extended_resized_WatershedNucSeg.tiff'
            ],
                channel_names = channel_names,
                out=outfolder + ('big%02d.ome.tif' % i))
        i = i + 1
