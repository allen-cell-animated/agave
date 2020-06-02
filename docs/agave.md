# AGAVE : Advanced GPU Accelerated Volume Explorer

Agave is a desktop application for viewing multichannel volume data.

The viewing engine is at its heart a "progressive path tracer". In interactive use, your image will appear noisy or grainy at first, but will refine over time as the rendering system builds up more and more render passes. As soon as you change any viewing parameter, including your camera angle, the image will reset and start over. The faster your GPU, the quicker the image will resolve itself.

## Contents

- [Overview](#user-interface-overview)
- [Loading Volume Data](#loading-volume-data)
- [Adjusting the 3D Camera](#adjusting-the-camera-view)
- [Appearance Panel](#appearance-panel)
- [Volume Channel Settings](#volume-channel-settings)
- [Output: Saving Results](#output-saving-results)
- [Camera Panel](#camera-panel)
- [Time Panel](#time-panel)

## User Interface Overview

Agave is a windowed desktop application.

The application is divided into several panels. All of the panels except for the 3D view can be undocked by dragging their title bar with the mouse. They can then be reconfigured by re-docking them in a different position.

All the panels can be resized by dragging from their edges. Your cursor should change to indicate when you are hovering over an edge that can be resized.

Tip: On some screens the panels may be too narrow to have enough range to move the horizontal sliders. In that case, it can help to un-dock the panel and stretch it wider.

If you close any of the panels, you can always reopen it from the View menu.

## Loading volume data

Agave supports .tiff and .czi files containing either 8-bit or 16-bit unsigned pixel intensities.
Agave can load multi-scene, multi-time, and multi-channel files. It can present up to 4 channels concurrently.

Metadata embedded in the file is important for Agave to display the volume properly. Agave can decode OME metadata (OME-TIFF files), ImageJ metadata (TIFF files exported from ImageJ/FIJI), and Zeiss CZI metadata.

### Open Volume

File-->Open Volume or "Open Volume" toolbar button

"Open Volume" will pop open a file browser dialog in which you can navigate to the volume file of choice.

If the file you select can not be loaded, an error dialog will pop open to notify you. There are several possible reasons why this may fail and if Agave can determine the reason, it will be displayed in the Agave log.
If the file is multi-scene, a dialog will pop up to let you select the scene you wish to view. To view a different scene, just re-open the same file using "Open Volume".
For files with many time samples, there is a Time panel in the user interface with a slider. Nothing will be loaded while dragging the slider; Agave will load the new time sample when the slider is released or the numeric input is incremented.

### Open JSON

File-->Open Json or "Open Json" toolbar button

"Open JSON" is to be used when you have previously saved an Agave session. This will load the volume file associated with that session, along with all of the Agave settings as they were when you saved.

### Recent

File-->Recent...

The bottom of the File menu will present a list of the most recently opened volume files for quick selection.

## Adjusting the camera view

The 3D viewport in Agave supports direct manipulation by zoom, pan, and rotate.

### Rotate

To rotate the volume, left-click and drag the mouse.

### Zoom

To zoom in or out, right-click and drag up or down. On Mac trackpad, Command-click and drag up or down, or double tap and drag up or down as if scrolling.

### Pan

To move the camera transverse to the view direction, middle-click and drag in the view. On Mac, Option-click and drag in the view.

### Reset

The Reset button in the toolbar will return your camera to a default view position that should frame the volume data in the window.

### Perspective/Orthographic

The Persp/Ortho toolbar button will toggle between a perspective projection and an orthographic one. In a perspective projection, parallel lines will meet in the distance and 3 dimensional objects will be foreshortened. This is considered a "realistic" view. In orthographic projection, parallel lines receding into the distacnce will remain parallel no matter which direction they go. There is no foreshortening or tapering of the volume.

## Appearance Panel

The appearance panel has the majority of the controls you will use to adjust the image. In particular, you will adjust volume channel intensities and colors, control how transparent or opaque the volume appears, and control lights and shadows.

This panel is split into subsections:
Global rendering controls
Volume Scale controls
Region of Interest Clipping
Lighting
per-channel volume data controls (only appear after a volume has been loaded)

### Global Rendering Settings

#### Renderer

Select "Path Traced" for the standard high quality rendering system. Select "Ray March Blending" for a faster performing but simplified renderer. The Ray Marching renderer will not do advanced lighting and shadowing but can still provide a useful view of your volume data. It does not behave in a progressive fashion, instead giving you the finished image immediately with no waiting.

#### Scattering Density

Scattering density controls how dense (opaque) or sparse (translucent) the volume appears. Higher density can be helpful for objects with well defined edges and less noisy data. It also tends to bring out the lighting in a more pronounced way. Lower density can be useful for multichannel viewing in which the data has a lot of overlap.

#### Shading Type

There are two shading methods: BRDF and Phase. The "Mixed" setting combines the two and is the default. The BRDF is more sensitive to lighting angle and can give shiny reflective appearance, whereas the Phase function does not produce glossy highlights.

#### Shading Type Mixture

In Mixed shading mode, this slider controls the relative contribution of Phase and BRDF.

#### Primary Ray Step Size

The Primary ray step size controls the distance rays can travel into the volume before hitting something. Larger values will render faster but also result in some rays bypassing important parts of the volume. This can be used for quicker preview rendering. Smaller values will be more precise and ensure that you are capturing every detail in the volume data.

#### Secondary Ray Step Size

The secondary ray step size controls the distance rays will travel after they have scattered within the volume and are bouncing out toward the light sources. Higher values will brighten the image and reduce shadows because more rays will penetrate through the volume and make it out to the lights. Smaller values will ensure that some rays are stopped by volume data and increase the accuracy of cast shadows.

#### Background Color

This button is very mysterious and has no understood effect. It is rumored that a wise man once knew its secret, but he only passed it on to his eldest son, whose whereabouts are unknown.

### Volume Scale

These X, Y, and Z values describe the physical dimensions of the volume data relative to the number of pixels. Often microscopes do not have the same physical dimensions in Z that they do in X and Y. Usually these values are read from the volume file's metadata. If they could not be found in the metadata, they will often appear here as X=1, Y=1, Z=1. They can be modified here.

### Region Of Interest (ROI)

The three sliders presented here let you clip the volume along each of its three axes. These sliders have two handles each, which let you clip each dimension from either side. For example, to see only the bottom Z half of your volume (or display the cross section middle slice), move the rightmost Z handle about halfway to the left.

### Lighting

There are two types of light illuminating your volume. One is an Area Light, represented by an imaginary square-shaped light source that can be moved anywhere around the volume. The second is a Sky Sphere, which can illuminate the volume from all directions.

Tip: it can be useful to turn one light off while tuning the settings for the other.

#### Area Light Theta, Phi, and Distance

These three coordinates let you position the light anywhere on a sphere around the volume. Theta and Phi are in radians (where 3.14159 radians is half a circle) (INSERT ILLUSTRATION HERE DESCRIBING THE SPHERICAL COORDINATES)

#### Area Light Size

The size of the light controls the spread of its illumination over the volume. A smaller light closer to the volume will appear very dramatic with exaggerated shadows, due to its rays spread over a wide angle. A larger light will give more even illumination.

#### Area Light Intensity

You may select a RGB color for the area light, and modify it with a scalar intensity value to brighten or darken it. Note that you can turn the light off by setting its color to black or its intensity to 0.

#### SkyLight Top, Middle, and Bottom

The Sky Light is described by a sphere completely surrounding the volume. You can set a color and intensity for the "north pole" (Top) of the sphere, the "equator" (Middle) and the "south pole" (Bottom). These values will be interpolated to compute the light at any point in between. The Sky Light can be turned off by setting its intensities to 0 or its colors to black.

## Volume Channel Settings

### Transfer Function Editor

The transfer function editor lets you transform the intensity values in your volume data so they are more amenable to display. You can select particular intensity ranges to view, to pick out particular details in the volume.

The editor displays a graph at the top. The background of the graph contains a histogram in black, showing where the volume intensity is distributed (Y axis) along the intensity range (X axis). The white line shows how volume intensities X are remapped to new intensities Y.

The editor has 4 mutually exclusive modes. You can switch between any of the modes and each mode's settings will be remembered.

#### Window / Level

Window/Level lets you remap the data range to a narrower range and clip data above and below the selected range. You get two controls: one to define how wide the range is (the window), and another to control where the window lies in the raw intensity range (the level).

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

File-->Save image or "Save image" toolbar button

Save the current viewport window as a PNG, or JPG file.

### Save Json

File-->Save to Json or "Save to json" toolbar button

Save to json will save the current Agave session into a small file that records every setting so you can pick up work where you left off. The json file is a text file, which you can (carefully) hand-edit if you need to. The file name of the currenly loaded volume file is embedded in the json, so if you copy the file around you should bring the volume data file with it. It is best to keep them in the same directory if possible.

### Save Python script (EXPERIMENTAL)

File-->Save to python script or "Save to python" toolbar button

This feature will save a python script file that contains all of the commands necessary to render the current viewport from python. Agave on windows can be run in "script mode" from the command line by running `agave --script "path_to_saved_python_script"`
Note this feature is currently an experimental preview and may not work on your system.

## Camera Panel

The camera panel controls will let you affect the image's exposure amount, and control the focus blurring.

### Film Exposure

The exposure value will brighten or darken the overall image.

### Exposure Time

This setting should normally be kept at 1, but if you have a sufficiently powerful GPU, increasing it will render more paths before refreshing the view, and make the image seem to resolve faster. Only change this if your image already resolves very quickly.

### Noise Reduction

Noise reduction applies a filter to the image to reduce the graininess of early render passes. After the image has resolved beyond a certain level, the denoising will have no effect.

### Aperture Size

Aperture size affects the depth of focus, or how much of the image is in focus. A small aperture size will keep the entire image in focus at all times. A large aperture size will let you only focus on a thin plane a specific distance from the camera.

### Projection Field of View

The field of view is an angle in degrees describing how narrow or wide an angle your camera can cover. A smaller field of view will span a very small section of your volume and will give the impression of zoooming in while at the same time reducing the perspective foreshortening. A large field of view will have increased perspective distortion and give the impression of zooming out as the camera angle can show more and more of the scene being displayed.

### Focal Distance

Focal distance describes the distance from the camera lens that is the center of focus. For aperture size 0, this has no effect, since the entire image will remain in focus (effectively an infinite focus depth range).

## Time Panel

If your volume has multiple times in the file, move the time slider or change the numeric input to load a new time sample. Beware that this is loading a whole new volume and can take some time. If your volume only has a single time, then the slider will have no effect.
