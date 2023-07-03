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
    plot_pca_populations(pca_reference)


def test_plot_pca_projections(pca_reference):
    pca_populations = pca_reference.pca_data.population.unique()[:1]
    plot_pca_projections(pca_reference, pca_populations)


def test_plot_pca_clusters(pca_reference):
    plot_pca_clusters(pca_reference, kmeans_k=2)


def test_plot_support_values(pca_reference):
    support_values = pd.Series([0.0] * pca_reference.pca_data.shape[0], index=pca_reference.pca_data.sample_id)
    plot_support_values(pca_reference, support_values)


def test_plot_pca_comparison(pca_reference):
    comparison = PCAComparison(pca_reference, pca_reference)
    plot_pca_comparison(comparison)


def test_plot_pca_comparison_rogue_samples(pca_reference):
    comparison = PCAComparison(pca_reference, pca_reference)
    plot_pca_comparison_rogue_samples(comparison, support_value_rogue_cutoff=1.0)