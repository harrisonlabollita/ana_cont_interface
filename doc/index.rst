.. _welcome:

triqs_ana_cont_interface
************************

.. sidebar:: triqs_ana_cont_interface |PROJECT_VERSION|

   This is the homepage of triqs_ana_cont_interface |PROJECT_VERSION|.
   For changes see the :doc:`changelog page <ChangeLog>`.

      .. image:: _static/logo_github.png
         :width: 75%
         :align: center
         :target: https://github.com/harrisonlabollita/ana_cont_interface


A :ref:`TRIQS <triqslibs:welcome>` front-end for `ana_cont
<https://github.com/josefkaufmann/ana_cont>`_. Give it a ``Gf`` or ``BlockGf`` on an
imaginary mesh, get the continued object back on a real-frequency mesh, with the physics
conventions applied once and the diagnostics needed to judge the result attached to the
output.

.. code-block:: python

   from triqs_ana_cont_interface import gf_problem, solve, linear_grid, validate

   prob = gf_problem(g_iw, grid=linear_grid(-10, 10, 501), error=1e-4, n_iw=60)
   res  = solve(prob, preblur=0.5)

   res.g_w                  # Gf / BlockGf on MeshReFreq, complex retarded
   res.a_w                  # the spectral matrix A = (i/2pi)(g_w - g_w^dag)
   print(validate(res))     # per-element table of fit quality and spectral weight

Learn how to use triqs_ana_cont_interface in the :ref:`documentation`.


.. toctree::
   :maxdepth: 2
   :hidden:

   install
   documentation
   issues
   ChangeLog
   about
