from __future__ import annotations

import pathlib
import random
from multiprocessing import Pool
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from numpy import typing as npt

from pandora.custom_errors import PandoraException
from pandora.custom_types import EmbeddingAlgorithm, Executable
from pandora.dataset import EigenDataset, NumpyDataset, _smartpca_finished
from pandora.distance_metrics import euclidean_sample_distance


def _bootstrap_and_embed(args):
    """Draws a bootstrap EigenDataset and performs dimensionality reduction using the provided arguments."""
    (
        dataset,
        bootstrap_prefix,
        smartpca,
        seed,
        embedding,
        n_components,
        redo,
        keep_bootstraps,
        smartpca_optional_settings,
    ) = args
    if embedding == EmbeddingAlgorithm.PCA:
        if _smartpca_finished(n_components, bootstrap_prefix) and not redo:
            bootstrap = EigenDataset(
                bootstrap_prefix,
                dataset._embedding_populations_file,
                dataset.sample_ids,
                dataset.populations,
                dataset.n_snps,
            )
        else:
            bootstrap = dataset.bootstrap(bootstrap_prefix, seed, redo)
        bootstrap.run_pca(
            smartpca=smartpca,
            n_components=n_components,
            redo=redo,
            smartpca_optional_settings=smartpca_optional_settings,
        )
    elif embedding == EmbeddingAlgorithm.MDS:
        bootstrap = dataset.bootstrap(bootstrap_prefix, seed, redo)
        bootstrap.run_mds(smartpca=smartpca, n_components=n_components, redo=redo)
    else:
        raise PandoraException(
            f"Unrecognized embedding option {embedding}. Supported are 'pca' and 'mds'."
        )
    if not keep_bootstraps:
        bootstrap.remove_input_files()
    return bootstrap


def bootstrap_and_embed_multiple(
    dataset: EigenDataset,
    n_bootstraps: int,
    result_dir: pathlib.Path,
    smartpca: Executable,
    embedding: EmbeddingAlgorithm,
    n_components: int,
    seed: Optional[int] = None,
    threads: Optional[int] = None,
    redo: bool = False,
    keep_bootstraps: bool = False,
    smartpca_optional_settings: Optional[Dict] = None,
) -> List[EigenDataset]:
    """Draws n_replicates bootstrap datasets of the provided EigenDataset and performs PCA/MDS analysis (as specified by
    embedding) for each bootstrap.

    Note that unless threads=1, the computation is performed in parallel.

    Parameters
    ----------
    dataset : EigenDataset
        Dataset object to base the bootstrap replicates on.
    n_bootstraps : int
        Number of bootstrap replicates to draw.
    result_dir : pathlib.Path
        Directory where to store all result files.
    smartpca : Executable
        Path pointing to an executable of the EIGENSOFT smartpca tool.
    embedding : EmbeddingAlgorithm
        Dimensionality reduction technique to apply. Allowed options are
        EmbeddingAlgorithm.PCA for PCA analysis and EmbeddingAlgorithm.MDS for MDS analysis.
    n_components : int
        Number of dimensions to reduce the data to.
        The recommended number is 10 for PCA and 2 for MDS.
    seed : int, default=None
        Seed to initialize the random number generator with.
    threads : int, default=None
        Number of threads to use for parallel bootstrap generation.
        Default is to use all system threads.
    redo : bool, default=False
        Whether to rerun analyses in case the result files are already present.
    keep_bootstraps : bool, default=False
        Whether to store all intermediate bootstrap ind, geno, and snp files.
        Note that setting this to True might require a notable amount of disk space.
    smartpca_optional_settings : Dict, default=None
        Additional smartpca settings.
        Not allowed are the following options: genotypename, snpname, indivname,
        evecoutname, evaloutname, numoutevec, maxpops. Note that this option is only used when embedding == "PCA".

    Returns
    -------
    bootstraps : List[EigenDataset]
        List of `n_replicates` boostrap replicates as EigenDataset objects. Each of the resulting
        datasets will have either bootstrap.pca != None or bootstrap.mds != None depending on the selected
        embedding option.
    """
    result_dir.mkdir(exist_ok=True, parents=True)

    if seed is not None:
        random.seed(seed)

    args = [
        (
            dataset,
            result_dir / f"bootstrap_{i}",
            smartpca,
            random.randint(0, 1_000_000),
            embedding,
            n_components,
            redo,
            keep_bootstraps,
            smartpca_optional_settings,
        )
        for i in range(n_bootstraps)
    ]
    with Pool(threads) as p:
        bootstraps = list(p.map(_bootstrap_and_embed, args))

    return bootstraps


def _bootstrap_and_embed_numpy(args):
    (
        dataset,
        seed,
        embedding,
        n_components,
        distance_metric,
        imputation,
    ) = args
    bootstrap = dataset.bootstrap(seed)
    if embedding == EmbeddingAlgorithm.PCA:
        bootstrap.run_pca(n_components, imputation)
    elif embedding == EmbeddingAlgorithm.MDS:
        bootstrap.run_mds(
            n_components,
            distance_metric,
            imputation,
        )
    else:
        raise PandoraException(
            f"Unrecognized embedding option {embedding}. Supported are 'pca' and 'mds'."
        )

    return bootstrap


def bootstrap_and_embed_multiple_numpy(
    dataset: NumpyDataset,
    n_bootstraps: int,
    embedding: EmbeddingAlgorithm,
    n_components: int,
    seed: Optional[int] = None,
    threads: Optional[int] = None,
    distance_metric: Callable[
        [npt.NDArray, pd.Series], Tuple[npt.NDArray, pd.Series]
    ] = euclidean_sample_distance,
    imputation: Optional[str] = "mean",
) -> List[NumpyDataset]:
    """Draws n_replicates bootstrap datasets of the provided NumpyDataset and performs PCA/MDS analysis (as specified by
    embedding) for each bootstrap.

    Note that unless threads=1, the computation is performed in parallel.

    Parameters
    ----------
    dataset : NumpyDataset
        Dataset object to base the bootstrap replicates on.
    n_bootstraps : int
        Number of bootstrap replicates to draw.
    embedding : EmbeddingAlgorithm
        Dimensionality reduction technique to apply. Allowed options are
        EmbeddingAlgorithm.PCA for PCA analysis and EmbeddingAlgorithm.MDS for MDS analysis.
    n_components : int
        Number of dimensions to reduce the data to.
        The recommended number is 10 for PCA and 2 for MDS.
    seed : int, default=None
        Seed to initialize the random number generator with.
    threads : int, default=None
        Number of threads to use for parallel bootstrap generation.
        Default is to use all system threads.
    distance_metric : Callable[[npt.NDArray, pd.Series, str], Tuple[npt.NDArray, pd.Series]], default=eculidean_sample_distance
        Distance metric to use for computing the distance matrix input for MDS. This is expected to be a
        function that receives the numpy array of sequences, the population for each sequencem and the imputation method
        as input and should output the distance matrix and the respective populations for each row.
        The resulting distance matrix is of size (n, m) and the resulting populations is expected to be
        of size (n, 1).
        Default is distance_metrics::eculidean_sample_distance (the pairwise Euclidean distance of all samples)
    imputation : Optional[str], default="mean"
        Imputation method to use. Available options are:\n
        - mean: Imputes missing values with the average of the respective SNP\n
        - remove: Removes all SNPs with at least one missing value.
        - None: Does not impute missing data.
        Note that depending on the distance_metric, not all imputation methods are supported. See the respective
        documentations in the distance_metrics module.

    Returns
    -------
    bootstraps : List[NumpyDataset]
        List of `n_replicates` boostrap replicates as NumpyDataset objects. Each of the resulting
        datasets will have either bootstrap.pca != None or bootstrap.mds != None depending on the selected
        embedding option.
    """
    if seed is not None:
        random.seed(seed)

    args = [
        (
            dataset,
            random.randint(0, 1_000_000),
            embedding,
            n_components,
            distance_metric,
            imputation,
        )
        for _ in range(n_bootstraps)
    ]
    with Pool(threads) as p:
        bootstraps = list(p.map(_bootstrap_and_embed_numpy, args))

    return bootstraps
