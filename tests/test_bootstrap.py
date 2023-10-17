import pathlib
import tempfile

import pytest

from pandora.bootstrap import (
    bootstrap_and_embed_multiple,
    bootstrap_and_embed_multiple_numpy,
)
from pandora.custom_types import EmbeddingAlgorithm
from pandora.embedding import MDS, PCA


@pytest.mark.parametrize("embedding", [EmbeddingAlgorithm.PCA, EmbeddingAlgorithm.MDS])
@pytest.mark.parametrize("keep_files", [True, False])
def test_bootstrap_and_embed_multiple(example_dataset, smartpca, embedding, keep_files):
    n_bootstraps = 2
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
    n_bootstraps = 2
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
