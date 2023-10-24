import concurrent.futures
import pathlib
import pickle
import tempfile
import time

import numpy as np
import pandas as pd
import pytest

from pandora.bootstrap import (
    STOP_BOOTSTRAP,
    _bootstrap_converged,
    _bootstrap_convergence_check,
    _run_function_in_process,
    _wrapped_func,
    bootstrap_and_embed_multiple,
    bootstrap_and_embed_multiple_numpy,
)
from pandora.custom_errors import PandoraException
from pandora.custom_types import EmbeddingAlgorithm
from pandora.embedding import MDS, PCA


def _dummy_func(status: int):
    """Dummy function that returns the status if it is > 0 and otherwise raises a ValueError."""
    if status >= 0:
        return status
    raise ValueError("Status < 0")


def _dummy_func_with_wait(status: int):
    """Dummy function that returns the status if it is > 0 and otherwise raises a ValueError.

    Compared to _dummy_func it however waits for 1s before returning/raising.
    """
    time.sleep(1)
    if status >= 0:
        return status
    raise ValueError("Status < 0")


def test_wrapped_func():
    with tempfile.NamedTemporaryFile() as tmpfile:
        tmpfile = pathlib.Path(tmpfile.name)

        # passing 1 should pickle a 1 in tmpfile
        _wrapped_func(_dummy_func, [1], tmpfile)
        status = pickle.load(tmpfile.open("rb"))
        assert status == 1

    # crate a new tempfile
    with tempfile.NamedTemporaryFile() as tmpfile:
        tmpfile = pathlib.Path(tmpfile.name)

        # passing 1 should put ValueError("Status < 0") into tmpfile
        # but not raise the ValueError
        _wrapped_func(_dummy_func, [-1], tmpfile)
        status = pickle.load(tmpfile.open("rb"))
        assert isinstance(status, ValueError)
        assert str(status) == "Status < 0"


def test_run_function_in_progress_stop_signal_set():
    # setting the stop signal before calling _run_function_in_progress should return None
    STOP_BOOTSTRAP.set()
    status = _run_function_in_process(_dummy_func, [1])
    assert status is None


def test_run_function_in_progress_stop_signal_unset():
    # if the stop signal is not set, calling _run_function_in_progress with _dummy_func and [1] should return 1
    STOP_BOOTSTRAP.clear()
    status = _run_function_in_process(_dummy_func, [1])
    assert status == 1


def test_run_function_in_progress_stop_signal_set_during_execution():
    # we create five processes using concurrent futures but set the stop signal once three are completed
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [
            pool.submit(_run_function_in_process, _dummy_func_with_wait, [status])
            for status in range(50)
        ]
        finished_ct = 0
        cancelled_results = 0

        for finished_task in concurrent.futures.as_completed(tasks):
            result = finished_task.result()
            if result is None:
                cancelled_results += 1
                continue
            finished_ct += 1
            assert (
                result >= 0
            )  # every result should be an int >= 0, no error should be raised

            if finished_ct >= 3:
                # send the stop signal to all remaining processes
                STOP_BOOTSTRAP.set()

        # we should only have three results, but
        assert finished_ct < 10
        assert cancelled_results >= 40


def test_run_function_in_progress_exception_during_execution():
    # we create five processes using concurrent futures but one of them raises a ValueError
    # we catch this error and transform it to a Pandora exception to make sure catching the error works as expected
    with pytest.raises(PandoraException):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            tasks = [
                pool.submit(_run_function_in_process, _dummy_func_with_wait, [status])
                for status in range(-1, 4)
            ]
            for finished_task in concurrent.futures.as_completed(tasks):
                try:
                    finished_task.result()
                except ValueError:
                    # we specifically catch only the ValueError raised by the _dummy_func_with_wait
                    # there should be no other errors
                    raise PandoraException()


@pytest.mark.parametrize(
    "embedding_algorithm", [EmbeddingAlgorithm.PCA, EmbeddingAlgorithm.MDS]
)
def test_bootstrap_converged_for_identical_embeddings(
    test_numpy_dataset, embedding_algorithm
):
    # first we need to compute the embedding
    if embedding_algorithm == EmbeddingAlgorithm.PCA:
        test_numpy_dataset.run_pca(n_components=2)
    else:
        test_numpy_dataset.run_mds(n_components=2)
    # if we pass the same embedding multiple times, we expect convergence
    # to make sure, we test different numbers of embeddings we pass to the convergence check
    for n_replicates in range(6, 10):
        # _bootstrap_convergence_check should determine convergence
        assert _bootstrap_convergence_check(
            [test_numpy_dataset] * n_replicates, embedding_algorithm, threads=2
        )


def test_bootstrap_does_not_converge_for_distinct_embeddings(test_numpy_dataset):
    def _random_embedding(embedding):
        new_data = embedding.embedding_matrix
        np.random.shuffle(new_data)
        new_data = pd.DataFrame(data=new_data, columns=[f"D{i}" for i in range(2)])
        new_data["sample_id"] = embedding.sample_ids
        new_data["population"] = embedding.populations
        return PCA(new_data, 2, embedding.explained_variances)

    # first we need to compute the embedding
    test_numpy_dataset.run_pca(n_components=2)
    # create 5 random embeddings (distinct enough from the test_numpy_dataset PCA to not indicate convergence)
    np.random.seed(42)
    random_embeddings = [_random_embedding(test_numpy_dataset.pca) for _ in range(5)]
    bootstraps = random_embeddings + [test_numpy_dataset.pca]
    assert not _bootstrap_converged(bootstraps, threads=2)


@pytest.mark.parametrize("embedding", [EmbeddingAlgorithm.PCA, EmbeddingAlgorithm.MDS])
@pytest.mark.parametrize("keep_files", [True, False])
def test_bootstrap_and_embed_multiple(example_dataset, smartpca, embedding, keep_files):
    n_bootstraps = 20
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        bootstraps = bootstrap_and_embed_multiple(
            dataset=example_dataset,
            n_bootstraps=n_bootstraps,
            result_dir=tmpdir,
            smartpca=smartpca,
            embedding=embedding,
            n_components=2,
            seed=0,
            threads=1,
            keep_bootstraps=keep_files,
            # manually disable the convergence check to check for the correct number of bootstraps
            bootstrap_convergence_check=False,
        )

        assert len(bootstraps) == n_bootstraps

        if embedding == EmbeddingAlgorithm.PCA:
            # each bootstrap should have embedding.pca != None, but embedding.mds == None
            assert all(b.pca is not None for b in bootstraps)
            assert all(isinstance(b.pca, PCA) for b in bootstraps)
            assert all(b.mds is None for b in bootstraps)
        elif embedding == EmbeddingAlgorithm.MDS:
            # each bootstrap should have embedding.mds != None, but embedding.pca == None
            assert all(b.mds is not None for b in bootstraps)
            assert all(isinstance(b.mds, MDS) for b in bootstraps)
            assert all(b.pca is None for b in bootstraps)

        # make sure that all files are present if keep_files == True, otherwise check that they are deleted
        if keep_files:
            assert all(b.files_exist() for b in bootstraps)
        else:
            assert not any(b.files_exist() for b in bootstraps)


@pytest.mark.parametrize("embedding", [EmbeddingAlgorithm.PCA, EmbeddingAlgorithm.MDS])
def test_bootstrap_and_embed_multiple_numpy(test_numpy_dataset, embedding):
    n_bootstraps = 20
    bootstraps = bootstrap_and_embed_multiple_numpy(
        dataset=test_numpy_dataset,
        n_bootstraps=n_bootstraps,
        embedding=embedding,
        n_components=2,
        seed=0,
        threads=2,
        # we don't test for 'remove' here because we have a small dataset and we might end up getting an error
        # because the bootstrapped dataset contains only nan-columns
        imputation="mean",
        # manually disable the convergence check to check for the correct number of bootstraps
        bootstrap_convergence_check=False,
    )

    assert len(bootstraps) == n_bootstraps

    if embedding == EmbeddingAlgorithm.PCA:
        # each bootstrap should have embedding.pca != None, but embedding.mds == None
        assert all(b.pca is not None for b in bootstraps)
        assert all(isinstance(b.pca, PCA) for b in bootstraps)
        assert all(b.mds is None for b in bootstraps)
    elif embedding == EmbeddingAlgorithm.MDS:
        # each bootstrap should have embedding.mds != None, but embedding.pca == None
        assert all(b.mds is not None for b in bootstraps)
        assert all(isinstance(b.mds, MDS) for b in bootstraps)
        assert all(b.pca is None for b in bootstraps)


def test_bootstrap_and_embed_multiple_with_convergence_check_pca(
    example_dataset, smartpca
):
    # example_dataset actually does not require all 100 bootstraps to converge
    # so if we run bootstrap_and_embed_multiple with the convergence check enabled
    # we should get less than 100 bootstraps as result

    n_bootstraps = 100
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        bootstraps = bootstrap_and_embed_multiple(
            dataset=example_dataset,
            n_bootstraps=n_bootstraps,
            result_dir=tmpdir,
            smartpca=smartpca,
            embedding=EmbeddingAlgorithm.PCA,
            n_components=2,
            seed=0,
            keep_bootstraps=False,
            bootstrap_convergence_check=True,
        )

        assert len(bootstraps) < n_bootstraps


@pytest.mark.parametrize("embedding", [EmbeddingAlgorithm.PCA, EmbeddingAlgorithm.MDS])
def test_bootstrap_and_embed_multiple_stops_with_pandora_exception_if_subprocess_fails(
    example_dataset, smartpca, embedding
):
    # we trigger a failure in the bootstrapping computation by demanding for way too many n_components
    n_bootstraps = 100
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)

        with pytest.raises(
            PandoraException,
            match="Something went wrong during the bootstrap computation.",
        ):
            bootstrap_and_embed_multiple(
                dataset=example_dataset,
                n_bootstraps=n_bootstraps,
                result_dir=tmpdir,
                smartpca=smartpca,
                embedding=embedding,
                # example dataset does not have that many SNPs
                n_components=100,
                seed=0,
                keep_bootstraps=False,
                bootstrap_convergence_check=True,
            )


@pytest.mark.parametrize("embedding", [EmbeddingAlgorithm.PCA, EmbeddingAlgorithm.MDS])
def test_bootstrap_and_embed_multiple_numpy_with_convergence_check_pca(
    test_numpy_dataset, embedding
):
    # test_numpy_dataset actually does not require all 100 bootstraps to converge (neither for PCA nor for MDS)
    # so if we run bootstrap_and_embed_multiple_numpy with the convergence check enabled
    # we should get less than 100 bootstraps as result

    n_bootstraps = 100

    bootstraps = bootstrap_and_embed_multiple_numpy(
        dataset=test_numpy_dataset,
        n_bootstraps=n_bootstraps,
        embedding=embedding,
        n_components=2,
        seed=0,
        bootstrap_convergence_check=True,
    )

    assert len(bootstraps) < n_bootstraps


@pytest.mark.parametrize("embedding", [EmbeddingAlgorithm.PCA, EmbeddingAlgorithm.MDS])
def test_bootstrap_and_embed_multiple_numpy_stops_with_pandora_exception_if_subprocess_fails(
    test_numpy_dataset, embedding
):
    # we trigger a failure in the bootstrapping computation by demanding for way too many n_components
    n_bootstraps = 100

    with pytest.raises(
        PandoraException, match="Something went wrong during the bootstrap computation."
    ):
        bootstrap_and_embed_multiple_numpy(
            dataset=test_numpy_dataset,
            n_bootstraps=n_bootstraps,
            embedding=embedding,
            n_components=100,
            seed=0,
            bootstrap_convergence_check=True,
        )
