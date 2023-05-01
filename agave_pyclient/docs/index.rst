Welcome to agave_pyclient's documentation!
==========================================

Quick start
===========

.. code-block:: console

   pip install agave_pyclient

You must have Agave installed separately. On command line, run:

.. code-block:: console

   agave --server &

And then in your Python code, import and use the :ref:`agave-renderer-label` class.

.. code-block:: python

   from agave_pyclient import AgaveRenderer

   # 1. connect to the agave server
   r = agave_pyclient.AgaveRenderer()
   # 2. tell it what data to load
   r.load_data("my_favorite.ome.tiff", 0, 0, 0, [], [])
   # 3. set some render settings (abbreviated list here)
   r.set_resolution(681, 612)
   r.background_color(0, 0, 0)
   r.render_iterations(128)
   r.set_primary_ray_step_size(4)
   r.set_secondary_ray_step_size(4)
   r.set_voxel_scale(0.270833, 0.270833, 0.53)
   r.exposure(0.75)
   r.density(28.7678)
   # 4. give the output a name
   r.session("output.png")
   # 5. wait for render and then save output
   r.redraw()

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Contents:

   Overview <self>
   agave_renderer
