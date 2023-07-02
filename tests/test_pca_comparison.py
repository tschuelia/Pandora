import pytest
from numpy import testing

from pandora.pca_comparison import *
from pandora.pca_comparison import _clip_missing_samples_for_comparison


def test_match_and_transform_identical_pcas(pca_reference):
    pca1, pca2, disparity = match_and_transform(pca_reference, pca_reference)

    # identical PCAs should have difference 0.0
    assert disparity == pytest.approx(0.0, abs=1e-6)

    # pca1 and pca2 should have identical data
    testing.assert_allclose(pca1.pc_vectors, pca2.pc_vectors)

    # pca1 and pca2 should have identical sample IDs
    assert all(pca1.pca_data.sample_id == pca2.pca_data.sample_id)


def test_clip_missing_samples_for_comparison(
    pca_reference, pca_comparable_fewer_samples
):
    # Pandora should also raise a Warning that >= 20% of samples were clipped
    with pytest.warns(UserWarning, match="More than 20% of samples"):
        comparable_clipped, reference_clipped = _clip_missing_samples_for_comparison(
            pca_comparable_fewer_samples, pca_reference
        )

    # both PCAs should now contain only the samples present in both PCAs
    present_in_both = set(pca_reference.pca_data.sample_id).intersection(
        set(pca_comparable_fewer_samples.pca_data.sample_id)
    )
    n_samples = len(present_in_both)

    assert comparable_clipped.pca_data.shape[0] == n_samples
    assert reference_clipped.pca_data.shape[0] == n_samples

    assert set(comparable_clipped.pca_data.sample_id) == present_in_both
    assert set(reference_clipped.pca_data.sample_id) == present_in_both


def test_match_and_transform_fails_for_different_sample_ids(
    pca_reference, pca_comparable_with_different_sample_ids
):
    with pytest.warns(UserWarning, match="More than 20% of samples"), pytest.raises(
        PandoraException, match="No samples left for comparison after clipping."
    ):
        match_and_transform(pca_comparable_with_different_sample_ids, pca_reference)


class TestPCAComparison:
    def test_compare_fails_for_incorrect_type(self):
        with pytest.raises(PandoraException, match="PCA objects"):
            PCAComparison(None, None)

    def test_compare_for_identical_pcas(self, pca_reference):
        comparison = PCAComparison(pca_reference, pca_reference)

        assert comparison.compare() == pytest.approx(1.0, abs=1e-6)

    def test_get_sample_support_values_for_identical_pcas(self, pca_reference):
        comparison = PCAComparison(pca_reference, pca_reference)
        support_values = comparison.get_sample_support_values()

        assert all(sv == pytest.approx(1.0, abs=1e-6) for sv in support_values)

    def test_with_transformations(self, pca_reference):
        # REFLECTION
        # when multiplying pca_reference with -1 we should still get a similarity of 1.0
        reflected_pca_data = pca_reference.pca_data.copy()
        for col in reflected_pca_data.columns:
            if "PC" not in col:
                continue
            reflected_pca_data[col] *= -1

        pca_reflected = PCA(
            pca_data=reflected_pca_data,
            explained_variances=pca_reference.explained_variances,
            n_pcs=pca_reference.n_pcs,
        )

        comparison = PCAComparison(pca_reflected, pca_reference)

        assert comparison.compare() == pytest.approx(1.0, abs=1e-6)

        # TRANSLATION
        # when adding +1 to pca_reference we should still get a similarity of 1.0
        shifted_pca_data = pca_reference.pca_data.copy()
        for col in shifted_pca_data.columns:
            if "PC" not in col:
                continue
            reflected_pca_data[col] += 1

        pca_shifted = PCA(
            pca_data=shifted_pca_data,
            explained_variances=pca_reference.explained_variances,
            n_pcs=pca_reference.n_pcs,
        )

        comparison = PCAComparison(pca_shifted, pca_reference)

        assert comparison.compare() == pytest.approx(1.0, abs=1e-6)

        # ROTATION
        # when rotating pca_reference by 180 degrees we should still get a similarity of 1.0
        pc_vectors = pca_reference.pc_vectors
        rotated_pc_vectors = np.fliplr(pc_vectors)

        pca_data_rotated = pd.DataFrame(
            data={
                "sample_id": pca_reference.pca_data.sample_id.values,
                "population": pca_reference.pca_data.population.values,
                "PC0": rotated_pc_vectors[:, 0],
                "PC1": rotated_pc_vectors[:, 1],
            }
        )
        pca_rotated = PCA(
            pca_data=pca_data_rotated,
            explained_variances=pca_reference.explained_variances,
            n_pcs=pca_reference.n_pcs,
        )

        comparison = PCAComparison(pca_rotated, pca_reference)
        assert comparison.compare() == pytest.approx(1.0, abs=1e-6)

    def test_compare_with_manually_computed_score(self, pca_reference_and_comparable_with_score_lower_than_one):
        pca1, pca2 = pca_reference_and_comparable_with_score_lower_than_one

        comparison = PCAComparison(pca1, pca2)

        assert 0 < comparison.compare() < 1
