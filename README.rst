
.. image:: logo.svg
  :width: 800
  :alt: Logo

======
Simon Lab Optics Package in PYthon
======

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lksplm/sloppy/HEAD?urlpath=%2Fdoc%2Ftree%2Findex.ipynb)

The Simon Lab Optics Package in PYthon is a collection of computational methods for the design of optical systems, especially cavities.

It strides to provide a unified interface to perform
- ABCD matrix formalism
- Raytracing
- Perturbation theory on top of paraxial eigenmodes


Description
===========

So far, the package includes Raytracing of Mirrors and Lenses, 


Installation
============

It's recommended use a conda environment to install the required packages::
    conda create -n sloppy 
    
If you want to use this environment as a kernel in your Jupyter Lab, add the kernel as described in this link `<https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084>`_.
    
Apart from the standard packages (numpy, matplotlib, ...), the only package that is a little tricky is `K3D-jupyter <https://github.com/K3D-tools/K3D-jupyter>`_
that is used to eable 3D visualisation in Jupyter Lab. It can be installed via::
    conda install -c conda-forge k3d

To enable the extension for Jupyter Lab run::
    jupyter labextension install @jupyter-widgets/jupyterlab-manager
    jupyter labextension install k3d
    
After cloning using git, install the package in the development version (only one supported now) via::
    python setup.py develop

You can then generate the documentation (managed via sphinx) as::
    python setup.py docs
    
Note
====

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
