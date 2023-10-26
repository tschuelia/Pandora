from __future__ import annotations

import concurrent.futures
import functools
import multiprocessing
import os
import pathlib
import pickle
import random
import signal
import tempfile
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import loguru
import numpy as np
import pandas as pd
from numpy import typing as npt

from pandora.custom_errors import PandoraException
from pandora.custom_types import EmbeddingAlgorithm, Executable
from pandora.dataset import EigenDataset, NumpyDataset, _smartpca_finished
from pandora.distance_metrics import euclidean_sample_distance
from pandora.embedding import Embedding
from pandora.embedding_comparison import BatchEmbeddingComparison
from pandora.logger import fmt_message


def _bootstrap_convergence_check(
    bootstraps: List[Union[NumpyDataset, EigenDataset]],
    embedding: EmbeddingAlgorithm,
    threads: int,
    logger: Optional[loguru.Logger] = None,
):
    if logger is not None:
        logger.debug(fmt_message("Running convergence check."))
    if embedding == EmbeddingAlgorithm.PCA:
        embeddings = [b.pca for b in bootstraps]
    elif embedding == EmbeddingAlgorithm.MDS:
        embeddings = [b.mds for b in bootstraps]
    else:
        raise PandoraException(
            f"Unrecognized embedding option {embedding}. Supported are 'pca' and 'mds'."
        )
    return _bootstrap_converged(embeddings, threads)


def _bootstrap_converged(bootstraps: List[Embedding], threads: int):
    """Checks for convergence by comparing the Pandora Stabilities for 10 subsets of the given list of bootstraps."""
    n_bootstraps = len(bootstraps)

    subset_size = int(n_bootstraps / 2)

    subset_stabilities = []
    for _ in range(10):
        subset = random.sample(bootstraps, k=subset_size)
        comparison = BatchEmbeddingComparison(subset)
        subset_stabilities.append(comparison.compare(threads=threads))

    subset_stabilities = np.asarray(subset_stabilities)
    subset_stabilities_reshaped = subset_stabilities.reshape(-1, 1)
    pairwise_relative_differences = (
        np.abs(subset_stabilities - subset_stabilities_reshaped) / subset_stabilities
    )
    res = np.all(pairwise_relative_differences <= 0.05)
    return res


def _wrapped_func(func, args, tmpfile):
    """Runs the given function ``func`` with the given arguments ``args`` and dumps the results as pickle in the
    tmpfile.

    If the function call raises an exception, the Exception is also dumped in the
    tmpfile.
    """
    try:
        result = func(*args)
    except Exception as e:
        result = e
    tmpfile.write_bytes(pickle.dumps(result))


class ProcessWrapper:
    """
    TODO: Docstring
    """

    def __init__(
        self,
        func: Callable,
        args: Iterable[Any],
        context: multiprocessing.context.BaseContext,
    ):
        self.func = func
        self.args = args
        self.context = context

        self.process = None

        # flags used to send signals to the running process
        self.pause_execution = False
        self.terminate_execution = False
        self.is_paused = False

        # prevent race conditions when handling a signal
        self.lock = self.context.RLock()

    def run(self):
        with self.lock:
            if self.terminate_execution:
                # received terminate signal before starting, nothing to do
                return

        with tempfile.NamedTemporaryFile() as result_tmpfile:
            result_tmpfile = pathlib.Path(result_tmpfile.name)
            with self.lock:
                self.process = self.context.Process(
                    target=functools.partial(
                        _wrapped_func, self.func, self.args, result_tmpfile
                    ),
                    daemon=True,
                )
                self.process.start()

            while 1:
                with self.lock:
                    process_complete = not self.process.is_alive()
                    if self.terminate_execution or process_complete:
                        # Process finished or termination signal sent from outside
                        break
                    elif self.pause_execution:
                        self._pause()
                    else:
                        self._resume()
                time.sleep(0.01)

            # Terminate process and get result
            self._terminate()
            if process_complete:
                # Only if the process was not externally terminated, can get and return the result
                result = pickle.load(result_tmpfile.open("rb"))
                if isinstance(result, Exception):
                    # if the underlying func call raises an Exception, it is also pickled in the result file
                    # in this case we simply re-raise the Exception to be able to properly handle it in the caller
                    raise result
                else:
                    return result

    def terminate(self):
        with self.lock:
            self.terminate_execution = True

    def pause(self):
        with self.lock:
            self.pause_execution = True

    def resume(self):
        with self.lock:
            self.pause_execution = False

    def _pause(self):
        with self.lock:
            if (
                self.process is not None
                and not self.terminate_execution
                and not self.is_paused
            ):
                os.kill(self.process.pid, signal.SIGSTOP)
                self.is_paused = True

    def _resume(self):
        with self.lock:
            if (
                self.process is not None
                # and not self.terminate_execution
                and self.is_paused
            ):
                os.kill(self.process.pid, signal.SIGCONT)
                self.is_paused = False

    def _terminate(self):
        with self.lock:
            if self.process is not None:
                if self.process.is_alive():
                    os.kill(self.process.pid, signal.SIGCONT)
                    self.process.terminate()
                self.process.join()
                self.process.close()
                self.process = None


class ParallelBoostrapProcessManager:
    """
    TODO: Docstring
    """

    def __init__(self, func: Callable, args: Iterable[Any]):
        self.context = multiprocessing.get_context("spawn")
        self.processes = [ProcessWrapper(func, arg, self.context) for arg in args]

    def run(
        self,
        threads: int,
        bootstrap_convergence_check: bool,
        embedding: EmbeddingAlgorithm,
        logger: Optional[loguru.Logger] = None,
    ):
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
            tasks = [pool.submit(process.run) for process in self.processes]

            bootstraps = []
            finished_indices = []

            for finished_task in concurrent.futures.as_completed(tasks):
                try:
                    bootstrap, bootstrap_index = finished_task.result()
                except Exception as e:
                    # terminate all running and waiting processes
                    self.terminate()
                    # cleanup the ThreadPool
                    pool.shutdown()
                    raise PandoraException(
                        "Something went wrong during the bootstrap computation."
                    ) from e

                # collect the finished bootstrap dataset
                bootstraps.append(bootstrap)
                # and also collect the index of the finished bootstrap:
                # we need to keep track of this for cleanup in the caller
                finished_indices.append(bootstrap_index)

                # perform a convergence check:
                # If the user wants to do convergence check (`bootstrap_convergence_check`)
                # AND if not all bootstraps already computed anyway
                # AND only every max(10, threads) replicates
                if (
                    bootstrap_convergence_check
                    and len(bootstraps) < len(self.processes)
                    and len(bootstraps) % max(10, threads) == 0
                ):
                    # Pause all running/waiting processes for the convergence check
                    # so we can use all available threads for the parallel convergence check computation
                    self.pause()
                    converged = _bootstrap_convergence_check(
                        bootstraps, embedding, threads, logger
                    )
                    self.resume()
                    if converged:
                        # in case convergence is detected, we set the event that interrupts all running processes
                        if logger is not None:
                            logger.debug(
                                fmt_message(
                                    "Bootstrap convergence detected. Stopping bootstrapping."
                                )
                            )
                        self.terminate()
                        break
        return bootstraps, finished_indices

    def pause(self):
        for process in self.processes:
            process.pause()

    def resume(self):
        for process in self.processes:
            process.resume()

    def terminate(self):
        for process in self.processes:
            process.terminate()


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
    logger: Optional[loguru.Logger] = None,
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
    logger : loguru.Logger, default=None
        Optional logger instance, used to log debug messages.

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
    if threads is None:
        threads = multiprocessing.cpu_count()

    if logger is not None:
        logger.debug(fmt_message(f"Using {threads} threads for bootstrapping."))

    result_dir.mkdir(exist_ok=True, parents=True)

    if seed is not None:
        random.seed(seed)

        if logger is not None:
            logger.debug(fmt_message(f"Setting the random seed to {seed}."))

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

    if logger is not None and bootstrap_convergence_check:
        logger.debug(
            fmt_message(
                f"Checking for bootstrap convergence every {max(10, threads)} bootstraps."
            )
        )

    parallel_bootstrap_process_manager = ParallelBoostrapProcessManager(
        _bootstrap_and_embed, bootstrap_args
    )
    bootstraps, finished_indices = parallel_bootstrap_process_manager.run(
        threads=threads,
        bootstrap_convergence_check=bootstrap_convergence_check,
        embedding=embedding,
        logger=logger,
    )

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

    return bootstrap, seed


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

    parallel_bootstrap_process_manager = ParallelBoostrapProcessManager(
        _bootstrap_and_embed_numpy, bootstrap_args
    )
    bootstraps, _ = parallel_bootstrap_process_manager.run(
        threads=threads,
        bootstrap_convergence_check=bootstrap_convergence_check,
        embedding=embedding,
    )

    return bootstraps
