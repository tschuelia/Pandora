import numpy as np
import pandas as pd
import pytest

from pandora.embedding_comparison import *
from pandora.embedding_comparison import (
    _clip_missing_samples_for_comparison,
    _numpy_to_dataframe,
    _pad_missing_samples,
)


def test_match_and_transform_identical_pcas(pca_example):
    pca1, pca2, disparity = match_and_transform(pca_example, pca_example)

    # identical PCAs should have difference 0.0
    assert disparity == pytest.approx(0.0, abs=1e-6)

    # pca1 and pca2 should have identical data
    np.testing.assert_allclose(pca1.embedding_matrix, pca2.embedding_matrix)

    # pca1 and pca2 should have identical sample IDs
    pd.testing.assert_series_equal(pca1.sample_ids, pca2.sample_ids)


def test_clip_missing_samples_for_comparison(pca_example):
    # use only four of the samples from pca_reference
    pca_data = pca_example.embedding.copy()[:4]
    pca_comparable_fewer_samples = PCA(
        pca_data, pca_example.n_components, pca_example.explained_variances
    )

    # Pandora should also raise a Warning that >= 20% of samples were clipped
    with pytest.warns(UserWarning, match="More than 20% of samples"):
        comparable_clipped, reference_clipped = _clip_missing_samples_for_comparison(
            pca_comparable_fewer_samples, pca_example
        )

    # both PCAs should now contain only the sample_ids present in both PCAs
    present_in_both = set(pca_example.sample_ids).intersection(
        set(pca_comparable_fewer_samples.sample_ids)
    )
    n_samples = len(present_in_both)

    assert comparable_clipped.embedding.shape[0] == n_samples
    assert reference_clipped.embedding.shape[0] == n_samples

    assert set(comparable_clipped.sample_ids) == present_in_both
    assert set(reference_clipped.sample_ids) == present_in_both


def test_match_and_transform_fails_for_different_sample_ids(pca_example):
    with pytest.warns(UserWarning, match="More than 20% of samples"), pytest.raises(
        PandoraException, match="No samples left for comparison after clipping."
    ):
        pca_data = pca_example.embedding.copy()
        pca_data.sample_id += "_foo"
        pca_comparable_with_different_sample_ids = PCA(
            pca_data, pca_example.n_components, pca_example.explained_variances
        )
        match_and_transform(pca_comparable_with_different_sample_ids, pca_example)


@pytest.mark.parametrize(
    "embedding_type", [EmbeddingAlgorithm.PCA, EmbeddingAlgorithm.MDS]
)
def test_pad_missing_samples(embedding_type):
    sample_ids = pd.Series(["sample1", "sample2", "sample3"])
    embedding_data = _numpy_to_dataframe(
        np.asarray([[1, 1], [2, 2], [3, 3]]),
        sample_ids,
        pd.Series(
            ["p1", "p2", "p3"],
        ),
    )
    if embedding_type == EmbeddingAlgorithm.PCA:
        embedding = PCA(embedding_data, 2, np.asarray([0, 0]))
    else:
        embedding = MDS(embedding_data, 2, 0.0)

    new_samples = pd.Series(["new1", "new2"])
    all_samples = pd.concat([sample_ids, new_samples], ignore_index=True)
    embedding_padded = _pad_missing_samples(all_samples, embedding)

    pd.testing.assert_series_equal(
        embedding_padded.sample_ids.sort_values(),
        all_samples.sort_values(),
        check_index=False,
        check_names=False,
    )
    assert embedding_padded.embedding.shape == (
        all_samples.shape[0],
        embedding.embedding.shape[1],
    )

    # embedding data for the new_samples should be all zeros
    for new in new_samples.values:
        new_data = embedding_padded.embedding.loc[lambda x: x.sample_id == new]
        assert new_data.D0.item() == 0.0
        assert new_data.D1.item() == 0.0


class TestEmbeddingComparison:
    def test_init_fails_for_incorrect_type(self):
        with pytest.raises(PandoraException, match="Embedding objects"):
            EmbeddingComparison(None, None)

    def test_compare_for_identical_pcas(self, pca_example):
        comparison = EmbeddingComparison(pca_example, pca_example)

        assert comparison.compare() == pytest.approx(1.0, abs=1e-6)

    def test_with_transformations(self, pca_example):
        # REFLECTION
        # when multiplying pca_reference with -1 we should still get a similarity of 1.0
        reflected_pca_data = pca_example.embedding.copy()
        for col in reflected_pca_data.columns:
            if "D" not in col:
                continue
            reflected_pca_data[col] *= -1

        pca_reflected = PCA(
            embedding=reflected_pca_data,
            n_components=pca_example.n_components,
            explained_variances=pca_example.explained_variances,
        )

        comparison = EmbeddingComparison(pca_reflected, pca_example)

        assert comparison.compare() == pytest.approx(1.0, abs=1e-6)

        # TRANSLATION
        # when adding +1 to pca_reference we should still get a similarity of 1.0
        shifted_pca_data = pca_example.embedding.copy()
        for col in shifted_pca_data.columns:
            if "D" not in col:
                continue
            reflected_pca_data[col] += 1

        pca_shifted = PCA(
            embedding=shifted_pca_data,
            n_components=pca_example.n_components,
            explained_variances=pca_example.explained_variances,
        )

        comparison = EmbeddingComparison(pca_shifted, pca_example)

        assert comparison.compare() == pytest.approx(1.0, abs=1e-6)

        # ROTATION
        # when rotating pca_reference by 180 degrees we should still get a similarity of 1.0
        embedding_vector = pca_example.embedding_matrix
        rotated_embedding_vector = np.fliplr(embedding_vector)

        pca_data_rotated = pd.DataFrame(
            data={
                "sample_id": pca_example.sample_ids.values,
                "population": pca_example.populations.values,
                "D0": rotated_embedding_vector[:, 0],
                "D1": rotated_embedding_vector[:, 1],
            }
        )
        pca_rotated = PCA(
            embedding=pca_data_rotated,
            n_components=pca_example.n_components,
            explained_variances=pca_example.explained_variances,
        )

        comparison = EmbeddingComparison(pca_rotated, pca_example)
        assert comparison.compare() == pytest.approx(1.0, abs=1e-6)

    def test_compare_is_valid_score(self):
        pca1 = PCA(
            pd.DataFrame(
                data={
                    "sample_id": ["sample1", "sample2", "sample3"],
                    "population": ["population1", "population2", "population3"],
                    "D0": [1, 2, 3],
                    "D1": [1, 2, 3],
                }
            ),
            2,
            np.asarray([0.0, 0.0]),
        )

        pca2 = PCA(
            pd.DataFrame(
                data={
                    "sample_id": ["sample1", "sample2", "sample3"],
                    "population": ["population1", "population2", "population3"],
                    "D0": [1, 1, 2],
                    "D1": [1, 2, 1],
                }
            ),
            2,
            np.asarray([0.0, 0.0]),
        )

        # the PS for pca1 and pca2 should be lower than 1

        comparison = EmbeddingComparison(pca1, pca2)

        assert 0 < comparison.compare() < 1


@pytest.mark.parametrize(
    "embedding_type", [EmbeddingAlgorithm.PCA, EmbeddingAlgorithm.MDS]
)
class TestBatchEmbeddingComparison:
    samples_embedding1 = pd.Series(
        ["sample1", "sample2", "sample3", "sample4", "sample5", "sample6"]
    )
    samples_embedding2 = pd.Series(
        ["sample2", "sample3", "sample4", "sample5", "sample6", "sample7"]
    )
    populations = pd.Series(
        [
            "population1",
            "population2",
            "population3",
            "population4",
            "population5",
            "population6",
        ]
    )

    data_embedding1 = np.asarray([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])

    data_embedding2 = np.asarray([[1, 1], [1, 2], [2, 1], [2, 2], [2, 3], [2, 4]])

    def _get_batch(self, embedding_type) -> BatchEmbeddingComparison:
        """
        The returned BatchEmbeddingComparison object contains three embeddings.
        Two are identical (with indices 0 and 2), and one is different.
        """
        embedding_data1 = _numpy_to_dataframe(
            self.data_embedding1, self.samples_embedding1, self.populations
        )
        embedding_data2 = _numpy_to_dataframe(
            self.data_embedding2, self.samples_embedding2, self.populations
        )
        if embedding_type == EmbeddingAlgorithm.PCA:
            e1 = PCA(embedding_data1, 2, np.asarray([0, 0]))
            e2 = PCA(embedding_data2, 2, np.asarray([0, 0]))
        elif embedding_type == EmbeddingAlgorithm.MDS:
            e1 = MDS(embedding_data1, 2, 0.0)
            e2 = MDS(embedding_data2, 2, 0.0)
        else:
            raise ValueError("Unsupported embedding type ", embedding_type)

        return BatchEmbeddingComparison([e1, e2, e1])

    def test_compare(self, embedding_type):
        batch = self._get_batch(embedding_type)
        ps = batch.compare()
        # since only two of the three embeddings are identical, the value should be neither 0 nor 1
        assert 0 <= ps < 1

    def test_get_pairwise_stabilities(self, embedding_type):
        batch = self._get_batch(embedding_type)
        pairwise_stabilities = batch.get_pairwise_stabilities()

        # three embeddings -> three unique pairs
        assert pairwise_stabilities.shape == (3,)
        # the stability for embeddings 0, 2 should be 1 since they are identical
        assert pairwise_stabilities[(0, 2)] == pytest.approx(1)
        # the stability for (0, 1) and (1, 2) should be unequal 1, but between 0 and 1
        assert 0 <= pairwise_stabilities[(0, 1)] < 1
        assert 0 <= pairwise_stabilities[(1, 2)] < 1

    def test_compare_clustering(self, embedding_type):
        batch = self._get_batch(embedding_type)
        pcs = batch.compare_clustering(kmeans_k=2)
        assert 0 <= pcs <= 1

    def test_get_pairwise_cluster_stabilities(self, embedding_type):
        batch = self._get_batch(embedding_type)
        pairwise_stabilities = batch.get_pairwise_cluster_stabilities(kmeans_k=2)

        # three embeddings -> three unique pairs
        assert pairwise_stabilities.shape == (3,)
        # the stability for embeddings 0, 2 should be 1 since they are identical
        assert pairwise_stabilities[(0, 2)] == pytest.approx(1)
        # the stability for (0, 1) and (1, 2) should be between 0 and 1
        assert 0 <= pairwise_stabilities[(0, 1)] <= 1
        assert 0 <= pairwise_stabilities[(1, 2)] <= 1

    def test_get_sample_support_values(self, embedding_type):
        unique_samples = set(self.samples_embedding1).union(
            set(self.samples_embedding2)
        )
        n_unique_samples = len(unique_samples)

        batch = self._get_batch(embedding_type)
        support_values = batch.get_sample_support_values()

        # we deliberately set the sample IDs not identical in both embeddings
        # we should however get a support value for all sample IDs present in any of the batch.embeddings
        # not only for the ones present in all
        assert support_values.shape == (n_unique_samples,)
        assert (0 <= support_values).all()
        assert (support_values <= 1).all()
