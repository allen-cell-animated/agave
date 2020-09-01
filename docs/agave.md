# AGAVE
## Advanced GPU Accelerated Volume Explorer

AGAVE is a desktop application for viewing multichannel volume data.

AGAVE’s core viewing engine uses a "progressive path tracer". During interactive use, your image will appear grainy at first, but will refine over time as the rendering system builds up more and more render passes. The speed of refinement depends on your hardware, the size and complexity of the image, and the AGAVE parameters you set, i.e., the faster your GPU, the quicker a rendering will resolve. GPU memory dictates the maximum size of the files AGAVE can load. As soon as you change any viewing parameter, including your camera angle, the rendering will start over.

## Contents

- [User Interface Overview](#user-interface-overview)
- [Loading Volume Data](#loading-volume-data)
- [Adjusting the 3D Camera View](#adjusting-the-camera-view)
- [Appearance Panel](#appearance-panel)
- [Volume Channel Settings](#volume-channel-settings)
- [Output: Saving Results](#output-saving-results)
- [Camera Panel](#camera-panel)
- [Time Panel](#time-panel)
- [Troubleshooting](#troubleshooting)

## User Interface Overview

AGAVE is a windowed desktop application.

The application is divided into several panels. All of the panels except for the 3D view can be undocked by dragging their title bar with the mouse. Undocked panels can float independently or the interface can be reconfigured by re-docking them in a different position. Note: currently, restarting AGAVE will reset the interface to the default configuration.

All panels can be resized by dragging from their edges. Your cursor should change to indicate when you are hovering over an edge that can be resized.

Tip: On some screens, the panels may be too narrow to have enough range to move the horizontal sliders. In that case, it can help to un-dock the panel and stretch it wider.

If you close any panels, you can always reopen them from the View menu.

## Loading volume data

AGAVE can load multi-scene, multi-time, and multi-channel files. It can present up to 4 channels concurrently and offers a time slider when time channels are detected. 

Metadata embedded in the file is important for AGAVE to display the volume properly. AGAVE can decode OME metadata (OME-TIFF files), ImageJ metadata (TIFF files exported from ImageJ/FIJI), and Zeiss CZI metadata.

We recommend preparing or converting your volumetric files to OME-TIFF (see Preparing an AGAVE Compatible ome-tiff file with Fiji <link to Preparing an AGAVE Compatible ome-tiff file with Fiji at bottom of this document> to ensure the expected data structure, channel order, etc.

AGAVE currently supports ome.tiff, tiff, and .czi image file formats containing either 8-bit or 16-bit unsigned pixel intensities. CZI is the native file format produced by Zeiss microscopes. See https://www.zeiss.com/microscopy/us/products/microscope-software/zen/czi.html for more information.  TIFF (Tagged Image File Format) is an open image file format commonly used to store microscopy data. See https://docs.openmicroscopy.org/ome-model/latest/ for more information.

### Open Volume

File-->Open Volume or \[Open Volume\] toolbar button

\[Open Volume\] will pop open a file browser dialog in which you can navigate to the volume file of choice.

If the file you select cannot be loaded, an error dialog will pop open to notify you. There are several possible reasons why this operation may fail and if AGAVE can determine the reason, it will be displayed in the AGAVE log (see description of the AGAVE log under [Troubleshooting](#troubleshooting)). If the file is multi-scene, a dialog will pop up to let you select the scene you wish to view. To view a different scene, just re-open the same file using \[Open Volume\]. For files with many time samples, there is a Time panel in the user interface with a slider. Nothing will be loaded while dragging the slider; AGAVE will load the new time sample when the slider is released or the numeric input is incremented.

### Open JSON

File-->Open Json or \[Open Json\] toolbar button

\[Open JSON\] can be used when you have previously saved an AGAVE session. This will load the volume file associated with that session, along with all of the AGAVE settings as they were when you saved (if you encounter problems, see [Save JSON] below for details about proper file and filepath structure).

### Recent

File-->Recent...

This menu will present a list of the most recently opened volume files for quick selection if AGAVE was previously used on your computer.

## Adjusting the camera view

The 3D viewport in Agave supports direct manipulation by *zoom*, *pan*, and *rotate*.

### Rotate

To rotate the volume, left-click and drag the mouse.

### Zoom

To zoom in or out, right-click and drag up or down. On Mac trackpad, Command-click and drag up or down, or drag up or down with two fingers as if scrolling.

### Pan

To slide the camera parallel to the screen, middle-click and drag in the view. On Mac, Option-click and drag in the view.

### Reset

The \[Reset\] button in the toolbar will return your camera to a default view position that should frame the volume data in the window.

### Perspective/Orthographic

The \[Persp/Ortho\] toolbar button will toggle between a perspective projection and an orthographic one. In a perspective projection, parallel lines will meet in the distance and 3D objects will be foreshortened. This is considered a "realistic" view. In orthographic projection, parallel lines receding into the distance will remain parallel no matter which direction they go. There is no foreshortening or tapering of the volume.

## Appearance Panel

The appearance panel has the majority of the controls you will use to adjust the image. In particular, you will adjust volume channel intensities and colors, control how transparent or opaque the volume appears, and control lights and shadows.

This panel is split into the following subsections, each detailed below: 
[*Global Rendering* controls], [*Volume Scale* controls], [*Region of Interest Clipping*], [*Lighting per-channel*], [*volume data*] controls (only appear after a volume has been loaded).

### Global Rendering Settings

#### Renderer

Select \[Path Traced\] for the standard high quality rendering system. Select \[Ray March Blending\] for a faster performing but simplified renderer. The Ray Marching renderer will not do advanced lighting and shadowing, but can still provide a useful view of your volume data. It does not behave in a progressive fashion, instead giving you the finished image immediately with no waiting.

#### Scattering Density

*Scattering density* controls how dense (opaque) or sparse (translucent) the volume appears. Higher density can be helpful for objects with well defined edges and less noisy data. It also tends to bring out the lighting in a more pronounced way. Lower density can be useful for multichannel viewing in which the data has a lot of overlap.

#### Shading Type

There are two shading methods: \[BRDF\] and \[Phase\]. The \[Mixed\] setting combines the two and is the default. The BRDF (Bidirectional Reflectance Distribution Function) is more sensitive to lighting angle and can produce a shiny reflective appearance, whereas the Phase function does not produce glossy highlights.

#### Shading Type Mixture

In Mixed shading mode, this slider controls the relative contribution of Phase and BRDF.

#### Primary Ray Step Size

The *primary ray step size* controls the distance rays can travel into the volume before hitting something. Larger values will render faster but also result in some rays bypassing important parts of the volume. This can be used for quicker preview rendering. Smaller values will be more precise and ensure that you are capturing every detail in the volume data.

#### Secondary Ray Step Size

The *secondary ray step size* controls the distance rays will travel after they have scattered within the volume and are bouncing out toward the light sources. Higher values will brighten the image and reduce shadows because more rays will penetrate through the volume and make it out to the lights. Smaller values will ensure that some rays are stopped by volume data, which will increase the accuracy of cast shadows.

#### Background Color

Clicking on the color square next to Background Color allows you to change the image background color from black (default) to any other color.

### Volume Scale

These X, Y, and Z values describe the physical dimensions of the volume data relative to the number of pixels. Often microscopes do not have the same physical dimensions in Z that they do in X and Y. Usually these values are read from the volume file's metadata. If they could not be found in the metadata, they will often appear here as X=1, Y=1, Z=1. They can be modified here.

### Region Of Interest (ROI)

Three sliders presented here let you clip the volume along each of its three axes. These sliders have two handles each, which let you clip each dimension from either side. For example, to see only the bottom Z half of your volume (or display the cross section middle slice), move the rightmost Z handle about halfway to the left.

### Lighting

There are two types of light illuminating your volume. One is an “Area Light”, represented by an imaginary square-shaped light source that can be moved anywhere around the volume. The second is a “Sky Sphere”, which can illuminate the volume from all directions.

Tip: it can be useful to turn one light off while tuning the settings for the other.

#### Area Light Theta, Phi, and Distance

These three coordinates let you position the light anywhere on a sphere around the volume. Theta and Phi are in radians (where 3.14159 radians is half a circle).

#### Area Light Size

The size of the light controls the spread of its illumination over the volume. A smaller light closer to the volume will appear very dramatic with exaggerated shadows, due to its rays spreading over a wide angle. A larger light will give a more even illumination.

#### Area Light Intensity

You may select a RGB color for the area light, and modify it with a scalar intensity value to brighten or darken it. Note that you can turn the light off by setting its color to black or its intensity to 0.

#### SkyLight Top, Middle, and Bottom

The Sky Light is described by a sphere completely surrounding the volume. You can set a color and intensity for the "north pole" (Top) of the sphere, the "equator" (Middle) and the "south pole" (Bottom). These values will be interpolated to compute the light at any point in between. The Sky Light can be turned off by setting its intensities to 0 or its colors to black.

## Volume Channel Settings

Each volume channel contains adjustable settings. Expand the channel menus to access the following parameters.

### Transfer Function Editor

The transfer function editor lets you transform the intensity values in your volume data to clarify and fine-tune your visual analysis. You can select particular intensity ranges to view, to pick out particular details in the volume.

The editor displays a graph at the top. The background of the graph contains a histogram in black, showing where the volume intensity is distributed (Y axis) along the intensity range (X axis). The white line shows how volume intensities X are remapped to new intensities Y.

The editor has 4 mutually exclusive modes. You can switch between any of the modes and each mode's settings will be remembered.

#### Window / Level

Window/Level lets you remap the data range to a narrower range and clip data above and below the selected range. AGAVE provides two controls: one to define how wide the range is (the window), and another to control where the window lies in the raw intensity range (the level).

#### Isovalue

Isovalue lets you select a range of intensity values and clips all other values to 0. You may select a middle intensity value and a range of values above and below it. A thinner range will let you isolate one particular intensity.

#### Histogram Percentile

Percentile mode is similar to Window/Level as it results in the same linear remapping, but the choice of start and end is based on a percentage of the total pixels in the image. The default is to clip the bottom 50% of pixels to zero, and clip the upper 2% of pixels to maximum.

#### Custom

In Custom mode, you are free to edit the graph yourself. You will create your own piecewise linear transfer function. You start by default with a 1-1 intensity remapping, with one point in the bottom left corner and another in the upper right. Click in the graph anywhere to create a new vertex. It will be represented by a white circle. Click the middle of a circle and drag to move it.

### Color settings

#### Diffuse Color

This should be thought of as the main color for this channel.

#### Specular Color

This is the color for reflective highlights. It is additive on top of the diffuse color. Leave it at black to have no shiny highlights at all. This color should be tuned in conjunction with the Glossiness slider.

#### Emissive Color

This color is not truly light-emitting, but can not be darkened by the effects of shadowing from other lights. It should be used sparingly, if at all.

#### Glossiness

The glossiness value controls how sharp the reflected Specular highlights are. It defaults to a low value which makes them seem more diffuse. Higher values will appear shinier or glossier.

## Output: Saving Results

### Save Image

File-->Save image or the \[Save image\] toolbar button

Save the current viewport window as a PNG, or JPG file.

### Save JSON

File-->Save to JSON or \[Save to JSON\] toolbar button

Save to JSON will save the current AGAVE session into a small file that records every setting so you can pick up work where you left off. The JSON file is a text file, which you can (carefully) hand-edit if you need to. The file name of the currently loaded volume file is embedded in the JSON, so if you copy the file around you should bring the volume data file with it. It is best to keep them in the same directory if possible.

## Camera Panel

The camera panel controls will let you affect the image's exposure amount, and control the focus blurring.

### Film Exposure

The exposure value will brighten or darken the overall image.

### Exposure Time

This setting should normally be kept at 1, but if you have a sufficiently powerful GPU, increasing it will render more paths before refreshing the view, and make the image resolve faster. Only change this if your image already resolves very quickly.

### Noise Reduction

Noise reduction applies a filter to the image to reduce the graininess of early render passes. After the image has resolved beyond a certain level, the denoiser will shut off and have no effect. The image will continue to accumulate samples and resolve via brute force computation.

### Aperture Size

Aperture size affects the depth of focus, or how much of the image is in focus. A small aperture size will keep the entire image in focus at all times. A large aperture size will let you only focus on a thin plane a specific distance from the camera.

### Projection Field of View

The field of view is an angle in degrees describing how narrow or wide an angle your camera can cover. A smaller field of view will span a very small section of your volume and will give the impression of zooming in while at the same time reducing the perspective foreshortening. A large field of view will have increased perspective distortion and give the impression of zooming out as the camera angle can show more and more of the scene being displayed.

### Focal Distance

Focal distance describes the distance from the camera lens that is the center of focus. For aperture size 0, this has no effect, since the entire image will remain in focus (effectively an infinite focus depth range).

## Time Panel

If your volume contains multiple time steps in the file, move the time slider or change the numeric input to load a new time sample. Beware that this is loading a whole new volume and can take some time. If your volume only has a single time, then the slider will have no effect.

## Troubleshooting

### AGAVE Log

The AGAVE log is a plain text stream of informational output from AGAVE.  It can be found in the following locations: 
* Windows:  C:\Users\username\AppData\Local\AllenInstitute\agave\logfile.log
* Mac OS:  ~/Library/Logs/AllenInstitute/agave/logfile.log
* Linux:  ~/.agave/logfile.log
For troubleshooting, it can be useful to refer to this file or send it with any communication about issues in AGAVE.
