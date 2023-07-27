Welcome to Pandora's documentation!
=======================================
Pandora is a command line tool that estimates the stability of your PCA or MDS analyses under bootstrapping.
As the name suggests: Pandora is the box of all evil for PCA and MDS analyses as there might be results surfacing that you don't like :-)

Pandora is specifically designed for population genetics and ancient DNA data.

Boostrap Stability Analysis
---------------------------
Pandora estimates the stability of your dimensionality reduction based on bootstrapping the SNPs in your data. Bootstrapping is a standard analysis technique from statistics to estimate confidence intervals. Pandora bootstraps the input dataset by resampling the SNPs with replacement. For each bootstrap replicate dataset we use the selected dimensionality reduction technique (PCA or MDS) and compute the respective embedding.

We finally quantify the stability on three levels:

**Pandora stability:** The Pandora stability score is a score from 0 to 1, with 0 being the worst possible score and 1 the best.
If your dataset exhibits a Pandora stability of 0, that means that all bootstrap replicates are entirely different,
and we recommend you to not base any conclusions on the embedding.
In contrast, a Pandora stability of 1 indicates that your dataset appears to be stable.

**Sample support values:** For each sample in the input dataset, we estimate how far apart their embedding is in euclidean
space when comparing the bootstrap replicates pairwise. The lower the support value, the more the respective sample
"jumps" around due to the bootstrapping. We recommend you to check the sample support values for all your samples before
drawing any conclusions and advice you to be especially careful with samples with low support.

**Pandora cluster stability:** Typically, a PCA analysis is followed by a clustering (usually using K-Means clustering)
to determine e.g. population structures in the data. Pandora estimates the (in)stability of the clustering using the
bootstrapped embeddings and the Fowlkes-Mallows index. This index compares cluster label assignments for all samples
when comparing two bootstrap Again, the stability is measured on a scale of 0 to 1, with 0 being the worst possible score and 1 the best.


Sliding Window Analysis
-----------------------
In addition to the above boostrapping stability, Pandora offers a sliding window stability estimation.
This execution mode is especially useful for whole genome analyses. The idea is to quantify the (in)stability of dimensionality
reduction when applied to different parts of the genome. In a sliding-window procedure, Pandora subsequently applies
dimensionality reduction (PCA or MDS) to a section of the whole genome data and compares them pairwise to compute the overall
stability as average similarity between pairs of sections.


Support
-------
If you encounter any trouble using Pandora, have a question, or you find a bug, please feel free to open an issue here.


Publication
-----------
The paper explaining the details of Pandora is soon available as preprint on bioRxiv. Stay tuned!



.. toctree::
   :maxdepth: 4
   :caption: Getting Started

   Installation <install.rst>
   Getting Started <getting_started.rst>

.. toctree::
   :maxdepth: 4
   :caption: Usage

   Usage <usage.rst>
   Command Line Interface <cli_config.rst>
   Examples <examples/example.rst>
   Pandora API <api/modules>



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
