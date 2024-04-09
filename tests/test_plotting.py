import numpy as np
import pandas as pd
import pytest

from pandora.custom_errors import PandoraException
from pandora.embedding import Embedding
from pandora.embedding_comparison import EmbeddingComparison
from pandora.plotting import (
    get_distinct_colors,
    plot_clusters,
    plot_embedding_comparison,
    plot_embedding_comparison_rogue_samples,
    plot_populations,
    plot_projections,
    plot_support_values,
)

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


def test_plot_pca_populations(pca_example):
    plot_populations(pca_example)


def test_plot_pca_projections(pca_example):
    pca_populations = pd.Series(pca_example.populations.unique()).head(1)
    plot_projections(pca_example, pca_populations)


def test_plot_pca_clusters(pca_example):
    plot_clusters(pca_example, kmeans_k=2)


def test_plot_support_values(pca_example):
    support_values = pd.Series(
        [0.0] * pca_example.embedding.shape[0],
        index=pca_example.sample_ids,
    )
    plot_support_values(pca_example, support_values)


def test_plot_support_values_with_different_sample_sets_issues_warnings():
    pca = Embedding(
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
        match=r"Some of the provided sample_support_values sample IDs are not present in the "
        r"embedding.embedding data.[\s\w{:'.]+otherSample2",
    ):
        plot_support_values(pca, support_values)

    # in this case we expect the warning message for sample3 and the support_values
    pca = Embedding(
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
        match=r"Not all samples in embedding.embedding data have a support value in "
        r"sample_support_values.[\s\w{:'.]+sample3",
    ):
        plot_support_values(pca, support_values)


def test_plot_support_values_with_disjoint_sample_ids_raises_pandora_exception():
    pca = Embedding(
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


def test_plot_pca_comparison(pca_example):
    comparison = EmbeddingComparison(pca_example, pca_example)
    plot_embedding_comparison(comparison)


def test_plot_pca_comparison_rogue_samples(pca_example):
    comparison = EmbeddingComparison(pca_example, pca_example)
    sample_ids = pca_example.sample_ids
    support_values = pd.Series([1.0] * sample_ids.shape[0], index=sample_ids)
    plot_embedding_comparison_rogue_samples(
        comparison, support_values=support_values, support_value_rogue_cutoff=1.0
    )
