
The NumPy-based Interface
=========================
The following code snippets provide examples on how to use the numpy-based Pandora interface.

PCA Analysis
------------
The following example demonstrates you how to do the PCA bootstrap stability analysis using the alternative ``numpy`` interface of Pandora.
Instead of providing the file path to EIGEN files, you directly provide Pandora with a ``numpy`` data matrix.

.. code-block:: python

    import pandas as pd
    import numpy as np

    from pandora.bootstrap import bootstrap_and_embed_multiple_numpy
    from pandora.custom_types import EmbeddingAlgorithm
    from pandora.dataset import NumpyDataset
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
------------
This example shows how to perform an MDS analysis using the NumPy-based interface. You can select the distance metric you wish to compute as input for MDS analysis.
The first example uses one of the pre-implemented distance metrics, afterwards I will demonstrate how you can define your own custom distance metric.

**Important:** while the above examples all also work in a Jupyter notebook, the following example will only run if you paste it into a python
file and run it from command line. The reason for this is the custom distance metric we will pass for MDS
(which is a python ``Callable`` and ``bootstrap_and_embed_multiple_numpy`` uses multiprocessing which causes some trouble when not wrapped in the ``if name == "main":``)

.. code-block:: python

    import pandas as pd
    import numpy as np

    from pandora.bootstrap import bootstrap_and_embed_multiple_numpy
    from pandora.custom_types import EmbeddingAlgorithm
    from pandora.dataset import NumpyDataset
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


Again we will se an output like ``Pandora Stability (PS):  0.92.``

Custom distance metric
^^^^^^^^^^^^^^^^^^^^^^
If you want to use a distance metric that is not implemented in Pandora, you can define one very easily as I will show you with the following
example in which we will use the ``scikit-learn`` pairwise ``cosine_distances`` function. You can define a per-sample and a per-population metric like this:

.. code-block:: python

    from sklearn.metrics.pairwise import cosine_distances

    from pandora.distance_metrics import population_distance
    from pandora.imputation import impute_data


    def cosine_sample_distance(input_data: npt.NDArray, populations: pd.Series, imputation: Optional[str]) -> Tuple[npt.NDArray, pd.Series]:
        # first we impute the data, note that depending on the distance metric you are using not all imputations make sense
        # you can of course also implement your own custom imputation method and not use the provided ``impute_data`` functionality.
        input_data = impute_data(input_data, imputation)
        return cosine_distances(input_data, input_data), populations

    def cosine_population_distance(input_data: npt.NDArray, populations: pd.Series, imputation: Optional[str]) -> Tuple[npt.NDArray, pd.Series]:
        # first we impute the data, note that depending on the distance metric you are using not all imputations make sense
        # you can of course also implement your own custom imputation method and not use the provided ``impute_data`` functionality.
        input_data = impute_data(input_data, imputation)
        return population_distance(input_data, populations, cosine_distances)


For the per-population metric, we make use of Pandora's ``population_distance`` function. Provided a numpy data array and the respective populations,
as well as the desired pairwise distance metric, ``population_distance`` will take care of the population grouping.
You can then use one of your custom distance metrics by passing it as ``distance_metric`` to the respective function calls in Pandora
(e.g. the above example of ``bootstrap_and_embed_multiple_numpy``).



Sliding-Window Analysis
-----------------------
The above examples show the usage of the numpy-based interface for the Pandora bootstrap analysis. Pandora additionally provides methods to estimate the stability of dimensionality reduction along a genome.
The code for these analyses is basically the same as above, but instead of ``bootstrap_and_embed_multiple_numpy``, we will use the ``sliding_window_embedding_numpy`` method.
The following example demonstrates a sliding-window PCA analysis.

.. code-block:: python

    import pandas as pd
    import numpy as np

    from pandora.custom_types import EmbeddingAlgorithm
    from pandora.dataset import NumpyDataset
    from pandora.embedding_comparison import BatchEmbeddingComparison
    from pandora.sliding_window import sliding_window_embedding_numpy

    # for the sliding window analysis we use a larger array as example
    geno_data = np.asarray([
        [1, 0, 2, 0, 2, 0, 2, 0, 1, 2, 1, 0, 2, 0, 2, 0, 2, 0, 1, 2],
        [1, 1, 1, 0, 1, 0, 2, 0, 2, 2, 1, 1, 1, 0, 1, 0, 2, 0, 2, 2],
        [1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2],
        [0, 1, 0, 2, 0, 1, 1, 1, 0, 0, 0, 1, 0, 2, 0, 1, 1, 1, 0, 0],
        [0, 2, 1, 2, 0, 1, 0, 2, 1, 1, 0, 2, 1, 2, 0, 1, 0, 2, 1, 1]]
    )
    # create a new pandas Series for the respective sample IDs and populations
    sample_ids = pd.Series(["SAMPLE0", "SAMPLE1", "SAMPLE2", "SAMPLE3", "SAMPLE4"])
    populations = pd.Series(["pop0", "pop1", "pop2", "pop3", "pop4"])

    # finally initialize the dataset using the geno data and the sample metadata
    dataset = NumpyDataset(geno_data, sample_ids, populations)

    # for this example, we separate the dataset into 5 windows and compute the PCA embedding for each of the windows
    sliding_windows = sliding_window_embedding_numpy(
        dataset=dataset,
        n_windows=5,
        embedding=EmbeddingAlgorithm.PCA,  # tell Pandora to compute PCA for each of the windows
        n_components=2,  # here we only compute 2 PCs
        threads=2,  # compute the bootstraps in parallel using 2 threads
    )

    # finally, using all windowed PCA objects, we create a container for comparing all replicates and getting the overall PS score
    batch_comparison = BatchEmbeddingComparison([w.pca for w in sliding_windows])
    pandora_stability = batch_comparison.compare()

    print("Pandora Stability (PS): ", round(pandora_stability, 2))


This will print something like ``Pandora Stability (PS):  0.93``.


Loading Eigen-files as NumpyDataset
-----------------------------------
Instead of defining your data as a numpy dataset manually, you can also load genotype dataset files in EIGENSTRAT format.
You can simply use the ``pandora.dataset.numpy_dataset_from_eigenfiles`` method to do so:

.. code-block:: python

    import pathlib

    from pandora.dataset import numpy_dataset_from_eigenfiles

    # set the prefix to the .geno, .ind, and .snp files
    # if your dataset is not in EIGENSTRAT format, check out Pandora's conversion module
    dataset_prefix = pathlib.Path("path/to/eigenfiles")
    # numpy_dataset_from_eigenfiles expects three files:
    # - path/to/eigenfiles.snp
    # - path/to/eigenfiles.geno
    # - path/to/eigenfiles.ind
    dataset = numpy_dataset_from_eigenfiles(dataset_prefix)
