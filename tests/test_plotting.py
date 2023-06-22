from pandora.plotting import get_distinct_colors


def test_get_distinct_colors(example_eigen_dataset_prefix):
    for i in range(15):
        colors = get_distinct_colors(i)
        # make sure get_distinct_colors returns the correct number of colors
        assert len(colors) == i
        #  make sure there are no duplicate colors
        assert len(set(colors)) == i
        # make sure that all colors are HSV color strings
        assert all(c.startswith("hsv(") for c in colors)
