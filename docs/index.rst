.. AGAVE documentation master file, created by
   sphinx-quickstart on Fri Sep  4 12:54:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====
AGAVE
=====
Advanced GPU Accelerated Volume Explorer
========================================

AGAVE is a desktop application for viewing multichannel volume data.

AGAVE's core viewing engine uses a "progressive path tracer". During interactive use, your image will appear grainy at first, but will refine over time as the rendering system builds up more and more render passes. The speed of refinement depends on your hardware, the size and complexity of the image, and the AGAVE parameters you set, i.e., the faster your GPU, the quicker a rendering will resolve. GPU memory dictates the maximum size of the files AGAVE can load. As soon as you change any viewing parameter, including your camera angle, the rendering will start over.

.. toctree::
   :includehidden:
   :maxdepth: 2
   :caption: Contents

   AGAVE <agave>

