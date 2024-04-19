import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

from pandora.custom_types import EmbeddingAlgorithm
from pandora.dataset import EigenDataset, NumpyDataset
from pandora.embedding import Embedding
from pandora.sliding_window import (
    sliding_window_embedding,
    sliding_window_embedding_numpy,
)


@pytest.fixture
def example_eigen_sliding_window_dataset_prefix() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data" / "example_sliding_window"


@pytest.fixture
def example_sliding_window_dataset(
    example_eigen_sliding_window_dataset_prefix,
) -> EigenDataset:
    return EigenDataset(example_eigen_sliding_window_dataset_prefix)


@pytest.fixture
def test_numpy_dataset_sliding_window() -> NumpyDataset:
    test_data = np.asarray(
        [
            [1, 2, 1, 2, 0, 1, 0, 1, 2, 0, 2, 0, 0, 1, 0, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 2, 2, 1, 0, 2, 1, 2, 2, 0, 2, 0, 2, 1, 1, 0],
            [0, 0, 0, 2, 0, 2, 1, 1, 1, 0, 1, 2, 2, 0, 2, 2, 1, 0, 0, 2],
        ]
    )
    sample_ids = pd.Series(["sample1", "sample2", "sample3"])
    populations = pd.Series(["population1", "population2", "population3"])
    dataset = NumpyDataset(test_data, sample_ids, populations)
    return dataset


@pytest.mark.parametrize("embedding", [EmbeddingAlgorithm.PCA, EmbeddingAlgorithm.MDS])
@pytest.mark.parametrize("keep_files", [True, False])
def test_sliding_window_embedding(
    example_sliding_window_dataset, smartpca, embedding, keep_files
):
    n_windows = 5
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        windows = sliding_window_embedding(
            dataset=example_sliding_window_dataset,
            n_windows=n_windows,
            result_dir=tmpdir,
            smartpca=smartpca,
            embedding=embedding,
            n_components=2,
            threads=1,
            keep_windows=keep_files,
        )

        assert len(windows) == n_windows

        if embedding == EmbeddingAlgorithm.PCA:
            # each window should have embedding.pca != None, but embedding.mds == None
            assert all(w.pca is not None for w in windows)
            assert all(isinstance(w.pca, Embedding) for w in windows)
            assert all(w.mds is None for w in windows)
        elif embedding == EmbeddingAlgorithm.MDS:
            # each window should have embedding.mds != None, but embedding.pca == None
            assert all(w.mds is not None for w in windows)
            assert all(isinstance(w.mds, Embedding) for w in windows)
            assert all(w.pca is None for w in windows)

        # make sure that all files are present if keep_files == True, otherwise check that they are deleted
        if keep_files:
            assert all(w.files_exist() for w in windows)
        else:
            assert not any(w.files_exist() for w in windows)


@pytest.mark.parametrize("embedding", [EmbeddingAlgorithm.PCA, EmbeddingAlgorithm.MDS])
def test_sliding_window_embedding_numpy(test_numpy_dataset_sliding_window, embedding):
    n_windows = 4
    sliding_windows = sliding_window_embedding_numpy(
        dataset=test_numpy_dataset_sliding_window,
        n_windows=n_windows,
        embedding=embedding,
        n_components=2,
        threads=2,
        # we don't test for 'remove' here because we have a small dataset and we might end up getting an error
        # because the windowed datasets contains only nan-columns
        imputation="mean",
    )

    assert len(sliding_windows) == n_windows

    if embedding == EmbeddingAlgorithm.PCA:
        # each window should have embedding.pca != None, but embedding.mds == None
        assert all(w.pca is not None for w in sliding_windows)
        assert all(isinstance(w.pca, Embedding) for w in sliding_windows)
        assert all(w.mds is None for w in sliding_windows)
    elif embedding == EmbeddingAlgorithm.MDS:
        # each window should have embedding.mds != None, but embedding.pca == None
        assert all(w.mds is not None for w in sliding_windows)
        assert all(isinstance(w.mds, Embedding) for w in sliding_windows)
        assert all(w.pca is None for w in sliding_windows)
