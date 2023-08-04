import pandas as pd
import pytest

from pandora.plotting import *

# NOTE: the tests in this file mostly just call the respective plotting function
# comparing plots is a bit difficult, and we just want to make sure the code does not crash


def test_get_distinct_colors():
    for i in range(15):
        colors = get_distinct_colors(i)
        # make sure get_distinct_colors returns the correct number of colors
        assert len(colors) == i
        #  make sure there are no duplicate colors
        assert len(set(colors)) == i
        # make sure that all colors are HSV color strings
        assert all(c.startswith("hsv(") for c in colors)


def test_plot_pca_populations(pca_reference):
    plot_populations(pca_reference)


def test_plot_pca_projections(pca_reference):
    pca_populations = pd.Series(pca_reference.embedding.population.unique()).head(1)
    plot_projections(pca_reference, pca_populations)


def test_plot_pca_clusters(pca_reference):
    plot_clusters(pca_reference, kmeans_k=2)


def test_plot_support_values(pca_reference):
    support_values = pd.Series(
        [0.0] * pca_reference.embedding.shape[0],
        index=pca_reference.embedding.sample_id,
    )
    plot_support_values(pca_reference, support_values)


def test_plot_support_values_with_different_sample_sets_issues_warnings():
    pca = PCA(
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
    support_values = pd.Series(
        [0.0, 0.3, 0.6, 1.0], index=["sample1", "sample2", "sample3", "otherSample2"]
    )

    # in this case we expect the warning message for otherSample2 and the embedding.embedding data
    with pytest.warns(
        UserWarning,
        match=r"Some of the provided sample_support_values sample IDs are not present in the embedding.embedding data.[\s\w{:'.]+otherSample2",
    ):
        plot_support_values(pca, support_values)

    # in this case we expect the warning message for sample3 and the support_values
    pca = PCA(
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
    support_values = pd.Series([0.0, 0.3], index=["sample1", "sample2"])

    with pytest.warns(
        UserWarning,
        match=r"Not all samples in embedding.embedding data have a support value in sample_support_values.[\s\w{:'.]+sample3",
    ):
        plot_support_values(pca, support_values)


def test_plot_support_values_with_disjoint_sample_ids_raises_pandora_exception():
    pca = PCA(
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
    support_values = pd.Series([0.0, 1.0], index=["otherSample1", "otherSample2"])

    with pytest.warns(
        # we catch both set warnings as generic UserWarning since pytest does not support matching two distinct
        # UserWarnings simultaneously
        UserWarning,
    ), pytest.raises(
        PandoraException, match="No samples left to plot after filtering."
    ):
        plot_support_values(pca, support_values)


def test_plot_pca_comparison(pca_reference):
    comparison = EmbeddingComparison(pca_reference, pca_reference)
    plot_embedding_comparison(comparison)


def test_plot_pca_comparison_rogue_samples(pca_reference):
    comparison = EmbeddingComparison(pca_reference, pca_reference)
    sample_ids = pca_reference.embedding.sample_id
    support_values = pd.Series([1.0] * sample_ids.shape[0], index=sample_ids)
    plot_embedding_comparison_rogue_samples(
        comparison, support_values=support_values, support_value_rogue_cutoff=1.0
    )
