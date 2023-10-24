from __future__ import annotations

import concurrent.futures
import functools
import multiprocessing
import os
import pathlib
import random
import signal
import time
from multiprocessing import Event, Process, Queue
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy import typing as npt

from pandora.custom_errors import PandoraException
from pandora.custom_types import EmbeddingAlgorithm, Executable
from pandora.dataset import EigenDataset, NumpyDataset, _smartpca_finished
from pandora.distance_metrics import euclidean_sample_distance
from pandora.embedding import Embedding
from pandora.embedding_comparison import BatchEmbeddingComparison

# Event to terminate waiting and running bootstrap processes in case convergence was detected
STOP_BOOTSTRAP = Event()
# Event to pause waiting and running bootstrap processes during convergence check
# Using this event allows us to utilize all cores for the convergence check
PAUSE_BOOTSTRAP = Event()


def _wrapped_func(func, result_queue, args):
    """Runs the given function ``func`` with the given arguments ``args`` and stores the result in result_queue.

    If the function call raises an exception, the Exception is also stored in the
    result_queue.
    """
    try:
        result_queue.put(func(*args))
    except Exception as e:
        result_queue.put(e)


def _run_function_in_process(func, args):
    """Runs the given function ``func`` with the provided arguments ``args`` in a multiprocessing.Process.

    We periodically check for a stop signal (`STOP_BOOTSTRAP`) that signals bootstrap convergence.
    If this signal is set, we terminate the running process.
    Returns the result of ``func(*args)`` in case the process terminates without an error.
    If the underlying ``func`` call raises an exception, the exception is passed through to the caller
    of this function.
    """
    if STOP_BOOTSTRAP.is_set():
        # instead of starting and immediately stopping the process stop here
        return
    # since multiprocessing.Process provides no interface to the computed result,
    # we need a Queue to store the bootstrap results in
    # -> we pass this Queue as additional argument to the _wrapped_func
    result_queue = Queue()

    try:
        # open a new Process using _wrapped_func
        # _wrapped_func simply calls the specified function ``func`` with the provided arguments ``args``
        # and stores the result (or Exception) in the result_queue
        process = Process(
            target=functools.partial(_wrapped_func, func, result_queue, args),
            daemon=True,
        )
        process.start()

        process_paused = False

        while 1:
            if not process.is_alive():
                # Bootstrapping finished, stop the loop
                break
            if STOP_BOOTSTRAP.is_set():
                # Bootstrapping convergence detected, terminate the running Process
                process.terminate()
                break
            if PAUSE_BOOTSTRAP.is_set():
                # Bootrap convergence check running requiring all provided resources, pause process
                if not process_paused:
                    os.kill(process.pid, signal.SIGTSTP)
                    process_paused = True
                time.sleep(0.01)
                continue
            if process_paused:
                # resume paused process
                os.kill(process.pid, signal.SIGCONT)
                process_paused = False
            time.sleep(0.01)
        process.join()
        process.close()
        if not STOP_BOOTSTRAP.is_set():
            # we can only get the result from the result_queue if the process was not terminated due to the stop signal
            result = result_queue.get(timeout=1)
            if isinstance(result, Exception):
                # the result_queue also stores the Exception if the underlying func call raises one
                # in this case we simply re-raise the Exception to be able to properly handle it
                raise result
            return result
    finally:
        result_queue.close()


def _bootstrap_convergence_check(
    bootstraps: List[Union[NumpyDataset, EigenDataset]],
    embedding: EmbeddingAlgorithm,
    threads: int,
):
    if embedding == EmbeddingAlgorithm.PCA:
        embeddings = [b.pca for b in bootstraps]
    elif embedding == EmbeddingAlgorithm.MDS:
        embeddings = [b.mds for b in bootstraps]
    else:
        raise PandoraException(
            f"Unrecognized embedding option {embedding}. Supported are 'pca' and 'mds'."
        )
    # interrupt other running bootstrap processes
    PAUSE_BOOTSTRAP.set()
    converged = _bootstrap_converged(embeddings, threads)
    # resume remaining bootstrap processes
    PAUSE_BOOTSTRAP.clear()
    return converged


def _bootstrap_converged(bootstraps: List[Embedding], threads: int):
    """Checks for convergence by comparing the Pandora Stabilities for 10 subsets of the given list of bootstraps."""
    n_bootstraps = len(bootstraps)

    subset_size = int(n_bootstraps / 2)

    subset_stabilities = []
    for _ in range(10):
        subset = random.sample(bootstraps, k=subset_size)
        comparison = BatchEmbeddingComparison(subset)
        # TODO: maybe pause all other bootstrap comparisons and then use all cores here
        # either in compare or do the loop in parallel
        subset_stabilities.append(comparison.compare(threads=threads))

    subset_stabilities = np.asarray(subset_stabilities)
    subset_stabilities_reshaped = subset_stabilities.reshape(-1, 1)
    pairwise_relative_differences = (
        np.abs(subset_stabilities - subset_stabilities_reshaped) / subset_stabilities
    )
    return np.all(pairwise_relative_differences <= 0.05)


def _bootstrap_and_embed(
    bootstrap_index,
    dataset,
    result_dir,
    smartpca,
    seed,
    embedding,
    n_components,
    redo,
    keep_bootstraps,
    smartpca_optional_settings,
):
    """Draws a bootstrap EigenDataset and performs dimensionality reduction using the provided arguments."""
    bootstrap_prefix = result_dir / f"bootstrap_{bootstrap_index}"

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

    return bootstrap, bootstrap_index


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
    bootstrap_convergence_check: bool = True,
    smartpca_optional_settings: Optional[Dict] = None,
) -> List[EigenDataset]:
    """Draws ``n_replicates`` bootstrap datasets of the provided EigenDataset and performs PCA/MDS analysis (as
    specified by embedding) for each bootstrap.

    If ``bootstrap_convergence_check`` is set, this method will draw *at most* ``n_replicates`` bootstraps. See
    Notes below for further details.
    Note that unless ``threads=1``, the computation is performed in parallel.

    Parameters
    ----------
    dataset : EigenDataset
        Dataset object to base the bootstrap replicates on.
    n_bootstraps : int
        Number of bootstrap replicates to draw.
        In case ``bootstrap_convergence_check`` is set, this is the upper limit of number of replicates.
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
    keep_bootstraps : bool, default=False
        Whether to store all intermediate bootstrap ind, geno, and snp files.
        Note that setting this to ``True`` might require a substantial amount of disk space depending on the size of your
        dataset.
    bootstrap_convergence_check : bool, default=True
        Whether to automatically determine bootstrap convergence. If ``True``, will only compute as many replicates as
        required for convergence according to our heuristic (see Notes below).
    smartpca_optional_settings : Dict, default=None
        Additional smartpca settings.
        Not allowed are the following options: ``genotypename``, ``snpname``, ``indivname``,
        ``evecoutname``, ``evaloutname``, ``numoutevec``, ``maxpops``. Note that this option is only used for PCA analyses.

    Returns
    -------
    bootstraps : List[EigenDataset]
        List of ``n_replicates`` boostrap replicates as EigenDataset objects. Each of the resulting
        datasets will have either ``bootstrap.pca != None`` or ``bootstrap.mds != None`` depending on the selected
        embedding option.

    Notes
    -----
    Bootstrap Convergence ("Bootstopping"): While more bootstraps yield more reliable stability analyses
    results, computing a vast amount of replicates is very compute heavy for typical genotype datasets.
    We thus suggest a trade-off between the accuracy of the stability and the ressource usage.
    To this end, we implement a bootstopping procedure intended to determine convergence of the bootstrapping procedure.
    Once every ``max(10, threads)``(*) replicates, we perform the following heuristic convergence check:
    Let :math:`N` be the number of replicate computed when performing the convergence check.
    We first create 10 random subsets of size :math:`int(N/2)` by sampling from all :math:`N` replicates.
    We then compute the Pandora Stability (PS) for each of the 10 subsets and compute the relative difference of PS
    values between all possible pairs of subsets :math:`(PS_1, PS_2)` by computing :math:`\\frac{\\left|PS_1 - PS_2\\right|}{PS_2}`.
    We assume convergence if all pairwise relative differences are below 5%.
    If we determine that the bootstrap has converged, all remaining bootstrap computations are cancelled.

    (*) The reasoning for checking every ``max(10, threads)`` is the following: if Pandora runs on a machine with e.g. 48
    provided threads, 48 bootstraps will be computed in parallel and will terminate at approximately the same time.
    If we check convergence every 10 replicates, we will have to perform 4 checks, three of which are unnecessary (since
    the 48 replicates are already computed anyway, might as well use them instead of throwing away 30 in case 10 would
    have been sufficient).
    """
    # before starting the bootstrap computation, make sure the convergence and pause signals are cleared
    STOP_BOOTSTRAP.clear()
    PAUSE_BOOTSTRAP.clear()

    if threads is None:
        threads = multiprocessing.cpu_count()

    result_dir.mkdir(exist_ok=True, parents=True)

    if seed is not None:
        random.seed(seed)

    bootstrap_args = [
        (
            bootstrap_index,
            dataset,
            result_dir,
            smartpca,
            random.randint(0, 1_000_000),
            embedding,
            n_components,
            redo,
            keep_bootstraps,
            smartpca_optional_settings,
        )
        for bootstrap_index in range(n_bootstraps)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        tasks = [
            pool.submit(_run_function_in_process, _bootstrap_and_embed, args)
            for args in bootstrap_args
        ]
        bootstraps = []
        finished_indices = []

        for finished_task in concurrent.futures.as_completed(tasks):
            try:
                bootstrap_dataset, bootstrap_index = finished_task.result()
            except Exception as e:
                STOP_BOOTSTRAP.set()
                time.sleep(0.1)
                pool.shutdown()
                raise PandoraException(
                    "Something went wrong during the bootstrap computation."
                ) from e

            # collect the finished bootstrap dataset
            bootstraps.append(bootstrap_dataset)
            # and also collect the index of the finished bootstrap:
            # we need to keep track of this for file cleanup later on
            finished_indices.append(bootstrap_index)

            # perform a convergence check:
            # If the user wants to do convergence check (`bootstrap_convergence_check`)
            # AND if not all n_bootstraps already computed anyway
            # AND only every max(10, threads) replicates
            if (
                bootstrap_convergence_check
                and len(bootstraps) < n_bootstraps
                and len(bootstraps) % max(10, threads) == 0
            ):
                if _bootstrap_convergence_check(bootstraps, embedding, threads):
                    # in case convergence is detected, we set the event that interrupts all running smartpca runs
                    STOP_BOOTSTRAP.set()
                    break

    # reset the convergence flag
    STOP_BOOTSTRAP.clear()

    # we also need to remove all files associated with the interrupted bootstrap calls
    for bootstrap_index in range(n_bootstraps):
        if bootstrap_index in finished_indices:
            # this is a finished bootstrap replicate that we are going to keep
            continue
        # remove the .geno, .snp, .ind, and .ckp file
        bootstrap_prefix = result_dir / f"bootstrap_{bootstrap_index}"
        for file_suffix in [".geno", ".snp", ".ind", ".ckp"]:
            # we allow files to be missing as the process might have just started and no files were created yet
            pathlib.Path(f"{bootstrap_prefix}{file_suffix}").unlink(missing_ok=True)

    return bootstraps


def _bootstrap_and_embed_numpy(
    dataset, seed, embedding, n_components, distance_metric, imputation
):
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
    bootstrap_convergence_check: bool = True,
) -> List[NumpyDataset]:
    """Draws ``n_replicates`` bootstrap datasets of the provided NumpyDataset and performs PCA/MDS analysis (as
    specified by ``embedding``) for each bootstrap.

    If ``bootstrap_convergence_check`` is set, this method will draw *at most* ``n_replicates`` bootstraps. See
    Notes below for further details.
    Note that unless ``threads=1``, the computation is performed in parallel.

    Parameters
    ----------
    dataset : NumpyDataset
        Dataset object to base the bootstrap replicates on.
    n_bootstraps : int
        Number of bootstrap replicates to draw.
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
    distance_metric : Callable[[npt.NDArray, pd.Series, str], Tuple[npt.NDArray, pd.Series]], default=eculidean_sample_distance
        Distance metric to use for computing the distance matrix input for MDS. This is expected to be a
        function that receives the numpy array of sequences, the population for each sequence and the imputation method
        as input and should output the distance matrix and the respective populations for each row.
        The resulting distance matrix is of size :math:`(n, m)`` and the resulting populations is expected to be
        of size :math:`(n, 1)`.
        Default is distance_metrics::eculidean_sample_distance (the pairwise Euclidean distance of all samples)
    imputation : Optional[str], default="mean"
        Imputation method to use. Available options are:\n
        - mean: Imputes missing values with the average of the respective SNP\n
        - remove: Removes all SNPs with at least one missing value.
        - None: Does not impute missing data.
        Note that depending on the distance_metric, not all imputation methods are supported. See the respective
        documentations in the distance_metrics module.
    bootstrap_convergence_check : bool, default=True
        Whether to automatically determine bootstrap convergence. If ``True``, will only compute as many replicates as
        required for convergence according to our heuristic (see Notes below).

    Returns
    -------
    bootstraps : List[NumpyDataset]
        List of ``n_replicates`` boostrap replicates as NumpyDataset objects. Each of the resulting
        datasets will have either ``bootstrap.pca != None`` or ``bootstrap.mds != None`` depending on the selected
        embedding option.

    Notes
    -----
    Bootstrap Convergence ("Bootstopping"): While more bootstraps yield more reliable stability analyses
    results, computing a vast amount of replicates is very compute heavy for typical genotype datasets.
    We thus suggest a trade-off between the accuracy of the stability and the ressource usage.
    To this end, we implement a bootstopping procedure intended to determine convergence of the bootstrapping procedure.
    Once every ``max(10, threads)`` (*) replicates, we perform the following heuristic convergence check:
    Let :math:`N` be the number of replicate computed when performing the convergence check.
    We first create 10 random subsets of size :math:`int(N/2)` by sampling from all :math:`N` replicates.
    We then compute the Pandora Stability (PS) for each of the 10 subsets and compute the relative difference of PS
    values between all possible pairs of subsets :math:`(PS_1, PS_2)` by computing :math:`\\frac{\\left|PS_1 - PS_2\\right|}{PS_2}`.
    We assume convergence if all pairwise relative differences are below 5%.
    If we determine that the bootstrap has converged, all remaining bootstrap computations are cancelled.

    (*) The reasoning for checking every ``max(10, threads)`` is the following: if Pandora runs on a machine with e.g. 48
    provided threads, 48 bootstraps will be computed in parallel and will terminate at approximately the same time.
    If we check convergence every 10 replicates, we will have to perform 4 checks, three of which are unnecessary (since
    the 48 replicates are already computed anyway, might as well use them instead of throwing away 30 in case 10 would
    have been sufficient).
    """
    # before starting the bootstrap computation, make sure the convergence and pause signals are cleared
    STOP_BOOTSTRAP.clear()
    PAUSE_BOOTSTRAP.clear()

    if threads is None:
        threads = multiprocessing.cpu_count()

    if seed is not None:
        random.seed(seed)

    bootstrap_args = [
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        tasks = [
            pool.submit(_run_function_in_process, _bootstrap_and_embed_numpy, args)
            for args in bootstrap_args
        ]
        bootstraps = []

        for finished_task in concurrent.futures.as_completed(tasks):
            try:
                bootstrap = finished_task.result()
                bootstraps.append(bootstrap)
            except Exception as e:
                STOP_BOOTSTRAP.set()
                time.sleep(0.1)
                pool.shutdown()
                raise PandoraException(
                    "Something went wrong during the bootstrap computation."
                ) from e

            # perform a convergence check:
            # If the user wants to do convergence check (`bootstrap_convergence_check`)
            # AND if not all n_bootstraps already computed anyway
            # AND only every max(10, threads) replicates
            if (
                bootstrap_convergence_check
                and len(bootstraps) < n_bootstraps
                and len(bootstraps) % max(10, threads) == 0
            ):
                if _bootstrap_convergence_check(bootstraps, embedding, threads):
                    # in case convergence is detected, we set the event that interrupts all running bootstrap runs
                    STOP_BOOTSTRAP.set()
                    break

    # reset the convergence flag
    STOP_BOOTSTRAP.clear()
    return bootstraps
