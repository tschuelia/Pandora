
.. _Examples:

Pandora Python Library Examples
===============================
In addition to the command line, all of Pandora's functionality can be used as Python library with an additional interface
for any kind of `numpy` data matrix. The idea is that many (computational) biologists use `R` or `Python` to perform their PCA and MDS
analyses instead of using `smartpca`. This enables the user to perform any kind of preprocessing before doing the PCA/MDS analyses.

In the following, I will provide examples on how to use the numpy-based interface for PCA and MDS analyses. Since Pandora
of course also has the Eigenfiles-based interface available in the Python library, I will also demonstrate how to use this interface.


.. toctree::
   :maxdepth: 4
   :caption: Examples

   NumPy-based interface <numpy_interface.rst>
   EIGEN-based interface <eigen_interface.rst>
   Plotting and further insights <inspections.ipynb>
   Working with R exports <r_import_export.rst>
