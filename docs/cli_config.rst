.. _Command Line Interface:
Pandora Command Line Interface Configuration
============================================

Configuration options:
----------------------
- ``dataset_prefix``, File path prefix pointing to the dataset to use for the Pandora analyses. Pandora will look for files called .* so make sure all files have the same prefix.
- ``result_dir``, Directory where to store all (intermediate) results to.
- ``file_format``, default = ``EIGENSTRAT``, Name of the file format your dataset is in. Supported formats are ``ANCESTRYMAP``, ``EIGENSTRAT``, ``PED``, ``PACKEDPED``, ``PACKEDANCESTRYMAP``. For more information see Section `Input data`_ below.
- ``convertf``, default = ``convertf``, File path pointing to an executable of Eigensoft's ``convertf`` tool. ``convertf`` is used if the provided dataset is not in ``EIGENSTRAT`` format. Default is ``convertf``. This will only work if ``convertf`` is installed systemwide.
- ``n_bootstraps``, default = 100, Number of bootstrap replicates to compute
- ``keep_bootstraps``, default = ``false``, Whether to store all bootstrap datasets files (``.geno``, ``.snp``, ``.ind``). Note that this will result in a substantial storage consumption. Note that the bootstrapped indicies are stored as checkpoints for full reproducibility in any case.
- ``n_components``, default = 10, Number of components to compute and compare for PCA or MDS analyses. We recommend 10 for PCA analyses and 2 for MDS analyses. The default is 10 since the default for ``embedding_algorithm`` is ``PCA``.
- ``embedding_algorithm``, default = ``PCA``, Dimensionality reduction technique you want to use. Allowed options are ``PCA`` and ``MDS``.
- ``smartpca``, default = ``smartpca``, File path pointing to an executable of Eigensoft's ``smartpca`` tool. ``smartpca`` is used for PCA analyses on the provided dataset. Default is ``smartpca``. This will only work if ``smartpca`` is installed systemwide.
- ``smartpca_optional_settings``, default = not set, Optional additional settings to use when performing PCA with ``smartpca``. See `SmartPCA`_ section below for more details.
- ``embedding_populations``, default = not set, File containing a new-line separated list of population names. Only these populations will be used for the dimensionality reduction. In case of PCA analyses, all remaining samples in the dataset will be projected onto the PCA results.
- ``support_value_rogue_cutoff``, default = 0.5, When plotting the support values, only samples with a support value lower than the ``support_value_rogue_cutoff`` will be annotated with their sample IDs. Note that all samples in the respective plot are color-coded according to their support value in any case.
- ``kmeans_k``, default = not set, Number of clusters k to use for K-Means clustering of the dimensionality reduction embeddings. If not set, the optimal number of clusters will be automatically determined according to the Bayesian Information Criterion (BIC).
- ``do_bootstrapping``, default = ``True``, Whether to do the stability analysis using bootstrapping.
- ``redo``, default = ``False``, Whether to rerun all analyses in case the results files from a previous run are already present. Careful: this will overwrite existing results!
- ``seed``, default = current unix timestamp, Seed to initialize the random number generator. This setting is recommended for reproducible analyses.
- ``threads``, default = number of system threads, Number of threads to use for the analysis.
- ``result_decimals``, default = 2, Number of decimals to round the stability scores and support values in the output.
- ``verbosity``, default = 1, Verbosity of the output logging of Pandora.

    - 0 = quiet, prints only errors and the results (loglevel = ``ERROR``)
    - 1 = verbose, prints all intermediate infos (loglevel = ``INFO``)
    - 2 = debug, prints intermediate infos and debug messages (loglevel = ``DEBUG``)

- ``plot_results``, default = ``False``, Whether to plot all dimensionality reduction results and sample support values.
- ``plot_dim_x``, default = 0, Dimension to plot on the x-axis. Note that the dimensions are zero-indexed. To plot the first dimension set ``plot_dim_x = 0``.
- ``plot_dim_y``, default = 1, Dimension to plot on the y-axis. Note that the dimensions are zero-indexed. To plot the second dimension set ``plot_dim_y = 1``.


.. _SmartPCA:
SmartPCA optional settings
^^^^^^^^^^^^^^^^^^^^^^^^^^

Pandora supports all smartPCA commands, for a list of possible settings
see the `SmartPCA
documentation <https://github.com/DReichLab/EIG/blob/master/POPGEN/README>`__.
Not allowed are the following options: ``genotypename``, ``snpname``,
``indivname``, ``evecoutname``, ``evaloutname``, ``numoutevec``,
``maxpops``. Use the following schema to set the options:

.. code:: yaml

   smartpca_optional_settings:
     shrinkmode: YES
     numoutlieriter: 1

.. _Input data:
Input data
----------

Pandora supports a variety of different input formats. Basically, we
support all file formats than can be converted to Eigensoft's Eigenstrat
format using the ``convertf`` program. Pandora expects the three input
files (SNP, GENO, IND files) to have the same prefix and the file
endings should follow the convention according to the table below.

================= =============================
File Format       Expected file endings
================= =============================
Ancestrymap       ``.geno``, ``.ind``, ``.snp``
Eigenstrat        ``.geno``, ``.ind``, ``.snp``
PED               ``.ped``, ``.fam``, ``.map``
PackedPED         ``.bed``, ``.fam``, ``.bim``
PackedAncestrymap ``.geno``, ``.ind``, ``.snp``
================= =============================

Pandora performs its bootstrapping file-based and makes use of the
Eigenstrat format. Thus, all other file formats are automatically
converted to Eigenstrat prior to the analyses using the ``convertf``
tool. Make sure to correctly set the ``convertf`` option in your config
file before running Pandora.
