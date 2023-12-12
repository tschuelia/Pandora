Usage
=====
You have two options to use Pandora. You can either use the Command Line Interface or you can use Pandora as Python library.

Command Line Interface
----------------------
When using Pandora's command line interface you need your data available in one of the typical genotype dataformats,
meaning the data is separated into three files, a genotype file, a SNP file, and an individual file.

You can configure your Pandora analysis using a configuration file in ``yaml`` format. You can find a detailed description
of all available settings here: :ref:`Command Line Interface`.

Suppose you have a configuration file called ``config.yaml``, then you can run the Pandora analyses like this:

.. code-block:: bash

    pandora -c config.yaml


See :ref:`Getting Started` for an example on how to use Pandora with the command line interface.


Python Library
--------------
In addition to the command line, all of Pandora's functionality can be used as Python library with an additional interface
for any kind of `numpy` data matrix. The idea is that many (computational) biologists use `R` or `Python` to perform their PCA and MDS
analyses instead of using `smartpca`. This enables the user to perform any kind of preprocessing before doing the PCA/MDS analyses.

See :ref:`Examples` for a few examples on how to use Pandora as Python library.
Since many biologists work with `R` instead of `Python`, you can also find an example on how to export your `R` data
and import it in `Python` such that you can reuse your existing `R` preprocessing scripts :-)


Additional Notes
----------------

PCA and MDS implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Command line interface, as well as when using the Eigen-based Pandora Python library, Pandora will perform PCA analyses using smartpca.
Smartpca is a powerful PCA tool that implements a lot of genotype-data specific routines and optimizations and provides a lot of useful options for meaningful PCA analyses such as outlier detection.
Pandora supports all custom configuration settings of smartpca. See the Section :ref:`SmartPCA` for more information. For MDS analyses, Pandora will use
smartpca to generate the Fst-distance matrix as input for MDS. Note that this distance matrix computes the distances between population and not between samples.
The subsequent MDS analysis is performed using the scikit-learn MDS implementation.
If you have genotype data in Eigenfiles but want to be able to do a more flexible analysis, consider using the alternative NumPy interface. Pandora provides a method
to load your genotype data in EIGENSTRAT format as numpy array.

If you are using the NumPy-based Pandora interface, PCA and MDS is performed using the scikit-learn implementations. For both analyses, Pandora supports different types of data imputation, see the API documentation for more information.
Per default, Pandora will apply SNP-wise mean imputation. The default distance metric for MDS analysis is the pairwise euclidean distance between all samples in your data. However, Pandora provides alternative distance metrics
and allows you to define your own distance metric as well. Again, see the API documentation for further information.

Note that since the two distinct interfaces implement PCA and MDS analyses a bit different, the results for the same input data may differ.
