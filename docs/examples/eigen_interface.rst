
The EIGEN-based interface
=========================

You can do the same analyses as you would do using the command line by using the python EIGEN interface.
This interface is specifically targeted for analyses with genotype data provided in EIGENSTRAT format.

.. code-block:: python

    import pathlib
    import tempfile

    from pandora.bootstrap import bootstrap_and_embed_multiple
    from pandora.custom_types import EmbeddingAlgorithm
    from pandora.dataset import EigenDataset
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

To do the analyses for MDS, just replace ``embedding=EmbeddingAlgorithm.PCA`` with ``embedding=EmbeddingAlgorithm.MDS`` and remove the  ``smartpca_optional_settings=dict(numoutlieriters=0)`` line as this will have no effect for MDS analyses anyway.
