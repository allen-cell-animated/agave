import numpy
from aicsimageio.readers import TiffReader

from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter

from pathlib import Path

from prefect import task, Flow

OUTROOT = "//allen/aics/animated-cell/Dan/2018-02-14_dan_vday_mitosis/timelapse_wt2_s2/"
CHNAMES = ["dna", "fibrillarin", "lamin_b1", "tom20", "brightfield"]
INFILES = [
    "prediction_dna.tiff",
    "prediction_fibrillarin.tiff",
    "prediction_lamin_b1.tiff",
    "prediction_tom20.tiff",
    "signal.tiff",
]


def convert_tiff_to_ome_tiff_1ch(filepathin, filepathout):
    image = TiffReader(filepathin).load()
    image = image.transpose([1, 0, 2, 3])
    # normalizes data in range 0 - uint16max
    image = image.clip(min=0.0)
    image = image / image.max()
    image = 65535 * image
    # convert float to uint16
    image = image.astype(numpy.uint16)
    with OmeTiffWriter(file_path=filepathout, overwrite_file=True) as writer:
        writer.save(
            image, channel_names=["dna"], pixels_physical_size=[0.290, 0.290, 0.290]
        )


def convert_combined():
    inroot = "\\\\allen\\aics\\modeling\\cheko\\for_others\\2018-02-14_dan_vday_mitosis\\timelapse_wt2_s2\\"
    for j in range(0, 20):
        finalimage = None
        for i in range(0, len(INFILES)):
            infilepath = inroot + str(j).zfill(2) + "\\" + INFILES[i]
            image = TifReader(infilepath).load()
            image = image.transpose([1, 0, 2, 3])
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

        with OmeTiffWriter(
            file_path=OUTROOT + "combined_frame_" + str(j).zfill(2) + ".ome.tiff",
            overwrite_file=True,
        ) as writer:
            writer.save(
                finalimage,
                channel_names=CHNAMES,
                pixels_physical_size=[0.290, 0.290, 0.290],
            )


def convertFiles():
    inroot = "\\\\allen\\aics\\modeling\\cheko\\for_others\\2018-02-14_dan_vday_mitosis\\timelapse_wt2_s2\\"
    # 00 .. 19
    for i in range(0, len(INFILES)):
        for j in range(0, 20):
            infilepath = inroot + str(j).zfill(2) + "\\" + INFILES[i]
            outfilepath = (
                OUTROOT + CHNAMES[i] + "_frame_" + str(j).zfill(2) + ".ome.tiff"
            )
            convert_tiff_to_ome_tiff_1ch(infilepath, outfilepath)


def combineFiles(files, out, channel_names=None):
    finalimage = None
    for f in files:
        ai = AICSImage(f)
        # ai.data is 5d.
        image = ai.data
        if image.dtype == numpy.float32:
            # normalizes data in range 0 - uint16max
            image = image.clip(min=0.0)
            image = image / image.max()
            image = 65535 * image
            # convert float to uint16
            image = image.astype(numpy.uint16)
        if finalimage is None:
            finalimage = [image[0]]
        else:
            finalimage = numpy.append(finalimage, [image[0]], axis=1)
    print(finalimage.shape)
    finalimage = finalimage.transpose([0, 2, 1, 3, 4])
    with OmeTiffWriter(file_path=out, overwrite_file=True) as writer:
        writer.save(
            finalimage,
            channel_names=channel_names,
            pixels_physical_size=[0.108, 0.108, 0.290],
        )


# for derek
# with open('\\\\allen\\aics\\microscopy\\UserFolders\\Derek\\2018-03-12_fake_fluorescence_images_v0\\resize_predictions.csv', newline='') as csvfile:
#     spamreader = csv.DictReader(csvfile)
#     i = 0
#     for row in spamreader:
#         # skip first?
#         if i > 0:
#             segfolder = '\\\\allen\\aics\\microscopy\\UserFolders\\Derek\\2018-03-12_fake_fluorescence_images_v0\\Watershed4xDownsampleOutput\\%02d\\' % i
#             predfolder = '\\\\allen\\aics\\microscopy\\UserFolders\\Derek\\2018-03-12_fake_fluorescence_images_v0\\%02d\\' % i
#             outfolder = '\\\\allen\\aics\\animated-cell\\Dan\\2018-03-12_fake_fluorescence_images_v0\\w4xd\\'
#             channel_names = [str(j) for j in range(0, 11)]
#             channel_names[7] = 'pred_memb'
#             channel_names[8] = 'pred_dna'
#             channel_names[9] = 'seg_memb'
#             channel_names[10] = 'seg_dna'
#             combineFiles([
#                 row['path_czi'],
#                 predfolder + 'prediction_63x_bf_membrane_caax_resized.tiff',
#                 predfolder + 'prediction_dna_extended_resized.tiff',
#                 segfolder + 'prediction_63x_bf_membrane_caax_resized_WatershedCellSeg.tiff',
#                 segfolder + 'prediction_dna_extended_resized_WatershedNucSeg.tiff'
#             ],
#                 channel_names = channel_names,
#                 out=outfolder + ('big%02d.ome.tif' % i))
#         i = i + 1


def convert_labefreetestdata():
    indir = "\\\\allen\\aics\\modeling\\cheko\\projects\\for_others\\2017-11-29_for_ac\\3500000766_100X_20170328_D04_P04.czi\\"
    imgs = [
        "img_chan_brightfield",
        "img_chan_dna",
        "img_chan_membrane",
        "img_chan_structure",
        "img_prediction_alpha_tubulin",
        "img_prediction_beta_actin",
        "img_prediction_dna",
        "img_prediction_fibrillarin",
        "img_prediction_lamin_b1",
        "img_prediction_membrane",
        "img_prediction_sec61_beta",
        "img_prediction_tom20",
        "img_segmentation",
    ]
    outfolder = "\\\\allen\\aics\\animated-cell\\Dan\\labelfree\\"
    channel_names = [j[4:] for j in imgs]
    combineFiles(
        [indir + j + ".tif" for j in imgs],
        channel_names=channel_names,
        out=outfolder + "881.ome.tif",
    )


@task
def save_timepoint_as_tiff(dask_array, idx):
    # write your saving code here
    writer = OmeTiffWriter(
        f"//allen/aics/animated-cell/Dan/LLS2/T{idx}.ome.tif", overwrite_file=True
    )
    imgdata = dask_array.compute()
    writer.save(imgdata, dimension_order="CZYX")


@task
def generate_timepoints_array(img):
    timepoints = []
    for i in range(img.size_t):
        timepoints.append(img.dask_data[0, i, :])
    # timepoints.append(img.dask_data[0, 0, :])
    return timepoints


def czi_to_tiffs(img_path):
    from prefect.engine.executors import DaskExecutor

    executor = DaskExecutor()
    img = AICSImage(img_path)
    num_t = img.size_t
    try:
        physical_pixel_size = img.reader.get_physical_pixel_size()
    except AttributeError:
        physical_pixel_size = (1.0, 1.0, 1.0)

    with Flow("convertCziToTimeOmeTiffs") as flow:
        timepoints = generate_timepoints_array(img)
        save_timepoint_as_tiff.map(timepoints, list(range(num_t)))
    flow.run(executor=executor)


img_path = Path(
    "//allen/aics/microscopy/Jie/Zeiss visit Dec 2019/2019-12-09/Lamin stem cells plate of 33deg lid 40deg-Deskew-24.czi"
)
czi_to_tiffs(img_path)
