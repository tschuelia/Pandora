from __future__ import annotations

import pathlib
from multiprocessing import Pool
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from numpy import typing as npt

from pandora.custom_errors import PandoraException
from pandora.custom_types import EmbeddingAlgorithm, Executable
from pandora.dataset import EigenDataset, NumpyDataset
from pandora.distance_metrics import euclidean_sample_distance


def _embed_window(args):
    (
        window,
        window_prefix,
        smartpca,
        embedding,
        n_components,
        redo,
        keep_windows,
        smartpca_optional_settings,
    ) = args

    if embedding == EmbeddingAlgorithm.PCA:
        window.run_pca(
            smartpca=smartpca,
            n_components=n_components,
            redo=redo,
            smartpca_optional_settings=smartpca_optional_settings,
        )

    elif embedding == EmbeddingAlgorithm.MDS:
        window.run_mds(smartpca=smartpca, n_components=n_components, redo=redo)

    else:
        raise PandoraException(
            f"Unrecognized embedding option {embedding}. Supported are 'pca' and 'mds'."
        )

    if not keep_windows:
        window.remove_input_files()

    return window


def sliding_window_embedding(
    dataset: EigenDataset,
    n_windows: int,
    result_dir: pathlib.Path,
    smartpca: Executable,
    embedding: EmbeddingAlgorithm,
    n_components: int,
    threads: Optional[int] = None,
    redo: bool = False,
    keep_windows: bool = False,
    smartpca_optional_settings: Optional[Dict] = None,
) -> List[EigenDataset]:
    """Separates the given EigenDataset into n_windows sliding-window datasets and performs PCA/MDS analysis (as
    specified by ``embedding``) for each window.

    Note that unless ``threads=1``, the computation is performed in parallel.

    Parameters
    ----------
    dataset : EigenDataset
        Dataset object separate into windows.
    n_windows : int
        Number of sliding-windows to separate the dataset into.
    result_dir : pathlib.Path
        Directory where to store all result files.
    smartpca : Executable
        Path pointing to an executable of the EIGENSOFT smartpca tool.
    embedding : EmbeddingAlgorithm
        Dimensionality reduction technique to apply. Allowed options are
        ``EmbeddingAlgorithm.PCA`` for PCA analysis and ``EmbeddingAlgorithm.MDS`` for MDS analysis.
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
    keep_windows : bool, default=False
        Whether to store all intermediate window-dataset ``ind``, ``geno``, and ``snp`` files.
        Note that setting this to True might require a substantial amount of disk space.
    smartpca_optional_settings : Dict, default=None
        Additional smartpca settings.
        Not allowed are the following options: ``genotypename``, ``snpname``, ``indivname``,
        ``evecoutname``, ``evaloutname``, ``numoutevec``, ``maxpops``. N
        ote that this option is only used when ``embedding == EmbeddingAlgorithm.PCA``.

    Returns
    -------
    windows : List[EigenDataset]
        List of ``n_windows`` subsets as EigenDataset objects. Each of the resulting window
        datasets will have either ``window.pca != None`` or ``window.mds != None`` depending on the selected
        ``embedding`` option.
    """
    result_dir.mkdir(exist_ok=True, parents=True)

    sliding_windows = dataset.get_windows(result_dir, n_windows)

    args = [
        (
            window,
            result_dir / f"window_{i}",
            smartpca,
            embedding,
            n_components,
            redo,
            keep_windows,
            smartpca_optional_settings,
        )
        for i, window in enumerate(sliding_windows)
    ]

    with Pool(threads) as p:
        sliding_windows = list(p.map(_embed_window, args))

    return sliding_windows


def _embed_window_numpy(args):
    (
        window,
        embedding,
        n_components,
        distance_metric,
        imputation,
    ) = args

    if embedding == EmbeddingAlgorithm.PCA:
        window.run_pca(n_components, imputation)
    elif embedding == EmbeddingAlgorithm.MDS:
        window.run_mds(n_components, distance_metric, imputation)

    else:
        raise PandoraException(
            f"Unrecognized embedding option {embedding}. Supported are 'pca' and 'mds'."
        )

    return window


def sliding_window_embedding_numpy(
    dataset: NumpyDataset,
    n_windows: int,
    embedding: EmbeddingAlgorithm,
    n_components: int,
    threads: Optional[int] = None,
    distance_metric: Callable[
        [npt.NDArray, pd.Series], Tuple[npt.NDArray, pd.Series]
    ] = euclidean_sample_distance,
    imputation: Optional[str] = "mean",
) -> List[NumpyDataset]:
    """Separates the given ``NumpyDataset`` into ``n_windows`` sliding-window datasets and performs PCA/MDS analysis (as
    specified by ``embedding``) for each window.

    Note that unless ``threads=1``, the computation is performed in parallel.

    Parameters
    ----------
    dataset : NumpyDataset
        Dataset object separate into windows.
    n_windows : int
        Number of sliding-windows to separate the dataset into.
    embedding : EmbeddingAlgorithm
        Dimensionality reduction technique to apply. Allowed options are
        ``EmbeddingAlgorithm.PCA`` for PCA analysis and ``EmbeddingAlgorithm.MDS`` for MDS analysis.
    n_components : int
        Number of dimensions to reduce the data to.
        The recommended number is 10 for PCA and 2 for MDS.
    threads : int, default=None
        Number of threads to use for parallel window embedding.
        Default is to use all system threads.
    distance_metric : Callable[[npt.NDArray, pd.Series, str], Tuple[npt.NDArray, pd.Series]], default=eculidean_sample_distance
        Distance metric to use for computing the distance matrix input for MDS. This is expected to be a
        function that receives the numpy array of sequences, the population for each sequence and the imputation method
        as input and should output the distance matrix and the respective populations for each row.
        The resulting distance matrix is of size ``(n, m)`` and the resulting populations is expected to be
        of size ``(n, 1)``.
        Default is ``distance_metrics::eculidean_sample_distance`` (the pairwise Euclidean distance of all samples)
    imputation : Optional[str], default="mean"
        Imputation method to use. Available options are:\n
        - ``"mean"``: Imputes missing values with the average of the respective SNP\n
        - ``"remove"``: Removes all SNPs with at least one missing value.
        - ``None``: Does not impute missing data.
        Note that depending on the distance_metric, not all imputation methods are supported. See the respective
        documentations in the ``distance_metrics`` module.

    Returns
    -------
    windows : List[NumpyDataset]
        List of ``n_windows`` subsets as ``NumpyDataset`` objects. Each of the resulting window
        datasets will have either ``window.pca != None`` or ``window.mds != None`` depending on the selected
        embedding option.
    """

    args = [
        (
            window,
            embedding,
            n_components,
            distance_metric,
            imputation,
        )
        for window in dataset.get_windows(n_windows)
    ]

    with Pool(threads) as p:
        sliding_windows = list(p.map(_embed_window_numpy, args))

    return sliding_windows
