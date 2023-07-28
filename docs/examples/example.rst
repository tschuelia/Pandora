Pandora Python Library Examples
===============================
In addition to the command line, all of Pandora's functionality can be used as Python library with an additional interface
for any kind of `numpy` data matrix. The idea is that many (computational) biologists use `R` or `Python` to perform their PCA and MDS
analyses instead of using `smartpca`. This enables the user to perform any kind of preprocessing before doing the PCA/MDS analyses.

In the following, I will provide examples on how to use the numpy-based interface for PCA and MDS analyses. Since Pandora
of course also has the Eigenfiles-based interface available in the Python library, I will also demonstrate how to use this interface.


The NumPy-based Interface
-------------------------

PCA Analysis
^^^^^^^^^^^^
The following example demonstrates you how to do the PCA bootstrap stability analysis using the alternative ``numpy`` interface of Pandora.
Instead of providing the file path to EIGEN files, you directly provide Pandora with a ``numpy`` data matrix.

.. code-block:: python

    import pandas as pd
    import numpy as np

    from pandora.custom_types import EmbeddingAlgorithm
    from pandora.dataset import NumpyDataset, bootstrap_and_embed_multiple_numpy
    from pandora.embedding_comparison import BatchEmbeddingComparison

    # this is the same geno type data as we used above, but already typed out as numpy matrix
    geno_data = np.asarray([
        [1, 0, 2, 0, 2, 0, 2, 0, 1, 2],
        [1, 1, 1, 0, 1, 0, 2, 0, 2, 2],
        [1, 2, 1, 1, 1, 1, 1, 1, 2, 2],
        [0, 1, 0, 2, 0, 1, 1, 1, 0, 0],
        [0, 2, 1, 2, 0, 1, 0, 2, 1, 1]]
    )
    # create a new pandas Series for the respective sample IDs and populations
    sample_ids = pd.Series(["SAMPLE0", "SAMPLE1", "SAMPLE2", "SAMPLE3", "SAMPLE4"])
    populations = pd.Series(["pop0", "pop1", "pop2", "pop3", "pop4"])

    # finally initialize the dataset using the geno data and the sample metadata
    dataset = NumpyDataset(geno_data, sample_ids, populations)

    # create 10 bootstrap replicates based on the dataset and compute the PCA embedding for each bootstrap replicate
    bootstrap_replicates = bootstrap_and_embed_multiple_numpy(
        dataset=dataset,
        n_bootstraps=10,
        embedding=EmbeddingAlgorithm.PCA,  # tell Pandora to compute PCA for each of the bootstrap replicates
        n_components=2,  # here we only compute 2 PCs
        seed=42, # set the seed for full reproducibility of the bootstraps
        threads=2  # compute the bootstraps in parallel using 2 threads
    )

    # finally, using all bootstrap PCA objects, we create a container for comparing all replicates and getting the overall PS score
    batch_comparison = BatchEmbeddingComparison([b.pca for b in bootstrap_replicates])
    pandora_stability = batch_comparison.compare()

    print("Pandora Stability (PS): ", round(pandora_stability, 2))


This will print something like ``Pandora Stability (PS):  0.96``.


MDS Analysis
^^^^^^^^^^^^
This example shows how to perform an MDS analysis using the NumPy-based interface. You can select the distance metric you wish to compute as input for MDS analysis.
The first example uses one of the pre-implemented distance metrics, afterwards I will demonstrate how you can define your own custom distance metric.

**Important:** while the above examples all also work in a Jupyter notebook, the following example will only run if you paste it into a python
file and run it from command line. The reason for this is the custom distance metric we will pass for MDS
(which is a python ``Callable`` and ``bootstrap_and_embed_multiple_numpy`` uses multiprocessing which causes some trouble when not wrapped in the ``if name == "main":``)

.. code-block:: python

    import pandas as pd
    import numpy as np

    from pandora.custom_types import EmbeddingAlgorithm
    from pandora.dataset import NumpyDataset, bootstrap_and_embed_multiple_numpy
    from pandora.distance_metrics import manhattan_population_distance
    from pandora.embedding_comparison import BatchEmbeddingComparison


    if __name__ == "__main__":
        # this is identical to the example with PCA above: define the data and init the dataset
        geno_data = np.asarray([
            [1, 0, 2, 0, 2, 0, 2, 0, 1, 2],
            [1, 1, 1, 0, 1, 0, 2, 0, 2, 2],
            [1, 2, 1, 1, 1, 1, 1, 1, 2, 2],
            [0, 1, 0, 2, 0, 1, 1, 1, 0, 0],
            [0, 2, 1, 2, 0, 1, 0, 2, 1, 1]]
        )
        sample_ids = pd.Series(["SAMPLE0", "SAMPLE1", "SAMPLE2", "SAMPLE3", "SAMPLE4"])
        populations = pd.Series(["pop0", "pop1", "pop2", "pop3", "pop4"])

        dataset = NumpyDataset(geno_data, sample_ids, populations)

        # instead of PCA, this time we pass MDS as embedding method
        # in this case we also need to pass a Callable, we use the above euclidean function in this example
        bootstrap_replicates = bootstrap_and_embed_multiple_numpy(
            dataset=dataset,
            n_bootstraps=10,  # again compute 10 bootstrap datasets
            embedding=EmbeddingAlgorithm.MDS,  # and perform MDS analysis for each bootstrap
            distance_metric=manhattan_population_distance,  # use the Manhattan distance between populations for MDS computation
            n_components=2,
            seed=42,
            threads=2
        )

        batch_comparison = BatchEmbeddingComparison([b.mds for b in bootstrap_replicates])
        pandora_stability = batch_comparison.compare()

        print("Pandora Stability (PS): ", round(pandora_stability, 2))


Again we will se an output like ``Pandora Stability (PS):  0.91.``

**Custom distance metric**
If you want to use a distance metric that is not implemented in Pandora, you can define one very easily as I will show you with the following
example in which we will use the ``scikit-learn`` pairwise ``cosine_distances`` function. You can define a per-sample and a per-population metric like this:

.. code-block:: python

    from sklearn.metrics.pairwise import cosine_distances

    from pandora.distance_metrics import *


    def cosine_sample_distance(input_data: npt.NDArray, populations: pd.Series) -> Tuple[npt.NDArray, pd.Series]:
        return cosine_distances(input_data, input_data), populations

    def cosine_population_distance(input_data: npt.NDArray, populations: pd.Series) -> Tuple[npt.NDArray, pd.Series]:
        return population_distance(input_data, populations, cosine_distances)


For the per-population metric, we make use of Pandora's ``population_distance`` function. Provided a numpy data array and the respective populations,
as well as the desired pairwise distance metric, ``population_distance`` will take care of the population grouping.
Of course you can implement an arbitrarily complex distance metric suited for your needs.


The EIGEN-based interface
-------------------------

You can do the same analyses as you would do using the command line by using the python EIGEN interface.
This interface is specifically targeted for analyses with genotype data provided in EIGENSTRAT format.

.. code-block:: python

    import pathlib
    import tempfile

    from pandora.custom_types import EmbeddingAlgorithm
    from pandora.dataset import EigenDataset, bootstrap_and_embed_multiple
    from pandora.embedding_comparison import BatchEmbeddingComparison


    # set up the variables for the dataset you want to analyze and provide a path to a smartpca executable
    eigen_example_prefix = pathlib.Path("example/example")
    smartpca = "path/to/smartpca"

    dataset = EigenDataset(eigen_example_prefix)

    with tempfile.TemporaryDirectory() as tmpdir:
        # for this toy example we won't store the actual bootstrap results and smartpca logs, so we do this computation in a TemporaryDirectory
        result_dir = pathlib.Path(tmpdir)

        # create 10 bootstrap replicates based on the dataset and compute the PCA embedding for each bootstrap replicate
        bootstrap_replicates = bootstrap_and_embed_multiple(
            dataset=dataset,
            n_bootstraps=10,
            result_dir=result_dir,
            smartpca=smartpca,
            embedding=EmbeddingAlgorithm.PCA,  # tell Pandora to compute PCA for each of the bootstrap replicates
            n_components=2,  # here we only compute 2 PCs
            seed=42,  # set the seed for full reproducibility of the bootstraps
            threads=2,  # compute the bootstraps in parallel using 2 threads
            smartpca_optional_settings=dict(numoutlieriters=0)  # set the number of outlier detection iterations to 0 for smartpca
        )


    # finally, using all bootstrap PCA objects, we create a container for comparing all replicates and getting the overall PS score
    batch_comparison = BatchEmbeddingComparison([b.pca for b in bootstrap_replicates])
    pandora_stability = batch_comparison.compare()

    print("Pandora Stability (PS): ", round(pandora_stability, 2))

This will print something like ``Pandora Stability (PS):  0.92.`` Note that due to the different implementations of PCA in
``smartpca`` versus ``scikit-learn``, the PS is slightly different for the numpy-based interface versus this EIGEN-based interface.
