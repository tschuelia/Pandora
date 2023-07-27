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


See (TODO: GETTING STARTED) for an example on how to use Pandora with the command line interface.


Python Library
--------------
In addition to the command line, all of Pandora's functionality can be used as Python library with an additional interface
for any kind of `numpy` data matrix. The idea is that many (computational) biologists use `R` or `Python` to perform their PCA and MDS
analyses instead of using `smartpca`. This enables the user to perform any kind of preprocessing before doing the PCA/MDS analyses.

See (TODO EXAMPLES) for a few examples on how to use Pandora as Python library.
Since many biologists work with `R` instead of `Python`, you can also find an example on how to export your `R` data
and import it in `Python` such that you can reuse your existing `R` preprocessing scripts :-)
(TODO R Export/Import Example)
