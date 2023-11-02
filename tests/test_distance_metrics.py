import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest
from numpy import testing as npt

from pandora.custom_errors import PandoraException
from pandora.dataset import EigenDataset, numpy_dataset_from_eigenfiles
from pandora.distance_metrics import (
    euclidean_population_distance,
    euclidean_sample_distance,
    fst_population_distance,
    hamming_sample_distance,
    manhattan_population_distance,
    manhattan_sample_distance,
    missing_corrected_hamming_sample_distance,
)

# scikit-allel throws these two errors most likely due to the small examples we are looking at in these tests
# TODO: investigate where they are coming from and if they might be an issue
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*[converting a masked, invalid value encountered].*"
)


@pytest.mark.parametrize(
    "sample_distance_metric",
    [
        euclidean_sample_distance,
        manhattan_sample_distance,
        hamming_sample_distance,
        missing_corrected_hamming_sample_distance,
    ],
)
def test_sample_distance_no_imputation(sample_distance_metric):
    # create some test data without missing data
    input_data = np.asarray([[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 1, 2]])
    populations = pd.Series(["p0", "p1", "p2", "p2"])

    distance_matrix, corresponding_populations = sample_distance_metric(
        input_data, populations, imputation=None
    )

    # despite sample 3 and 4 having the sample population, the resulting populations should remain the same
    # as the distance metric operates per sample
    pd.testing.assert_series_equal(populations, corresponding_populations)

    # the size of the distance matrix should be n_samples x n_samples
    n_samples = input_data.shape[0]
    assert distance_matrix.shape == (n_samples, n_samples)


@pytest.mark.parametrize(
    "sample_distance_metric", [euclidean_sample_distance, manhattan_sample_distance]
)
def test_sample_distance_fails_for_no_imputation_and_missing_data(
    sample_distance_metric,
):
    # Euclidean and Manhattan distances require some kind of data imputation as they cannot handle missing values
    # so calling the sample_distance_metric with missing data and imputation=None should fail

    input_data = np.asarray([[np.nan, 0, 0], [1, 1, 1], [2, 2, 2], [0, 1, 2]])
    populations = pd.Series(["p0", "p1", "p2", "p2"])

    with pytest.raises(
        PandoraException,
        match="Imputation method cannot be None if input_data contains NaN values.",
    ):
        sample_distance_metric(input_data, populations, imputation=None)


@pytest.mark.parametrize(
    "sample_distance_metric", [euclidean_sample_distance, manhattan_sample_distance]
)
@pytest.mark.parametrize("imputation", ["mean", "remove"])
def test_sample_distance_with_imputation_and_missing_data(
    sample_distance_metric, imputation
):
    # Euclidean and Manhattan distances require some kind of data imputation as they cannot handle missing values
    # so calling the sample_distance_metric with missing data and imputation=imputation should run without any errors
    input_data = np.asarray([[np.nan, 0, 0], [1, 1, 1], [2, 2, 2], [0, 1, 2]])
    populations = pd.Series(["p0", "p1", "p2", "p2"])

    distance_matrix, corresponding_populations = sample_distance_metric(
        input_data, populations, imputation=imputation
    )

    # despite sample 3 and 4 having the sample population, the resulting populations should remain the same
    # as the distance metric operates per sample
    pd.testing.assert_series_equal(populations, corresponding_populations)

    # the size of the distance matrix should be n_samples x n_samples
    n_samples = input_data.shape[0]
    assert distance_matrix.shape == (n_samples, n_samples)


@pytest.mark.parametrize(
    "population_distance_metric",
    [euclidean_population_distance, manhattan_population_distance],
)
def test_population_distance_no_imputation(population_distance_metric):
    # create some test data without missing data
    input_data = np.asarray([[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 1, 2]])
    populations = pd.Series(["p0", "p1", "p2", "p2"])

    distance_matrix, corresponding_populations = population_distance_metric(
        input_data, populations, imputation=None
    )

    # sample 3 and 4 have the sample population
    # the resulting populations should thus be reduced to three unique populations
    assert len(corresponding_populations) == 3
    # the resulting populations should be identical to the unique ones of the original populations
    assert populations.unique().tolist() == corresponding_populations.tolist()

    # the size of the distance matrix should be n_populations x n_populations
    # with n_populations > n_samples
    n_samples = input_data.shape[0]
    n_populations = populations.unique().shape[0]
    assert n_samples > n_populations
    assert distance_matrix.shape == (n_populations, n_populations)


@pytest.mark.parametrize(
    "population_distance_metric",
    [euclidean_population_distance, manhattan_population_distance],
)
def test_population_distance_fails_for_no_imputation_and_missing_data(
    population_distance_metric,
):
    # Euclidean and Manhattan distances require some kind of data imputation as they cannot handle missing values
    # so calling the population_distance_metric with missing data and imputation=None should fail
    input_data = np.asarray([[np.nan, 0, 0], [1, 1, 1], [2, 2, 2], [0, 1, 2]])
    populations = pd.Series(["p0", "p1", "p2", "p2"])

    with pytest.raises(
        PandoraException,
        match="Imputation method cannot be None if input_data contains NaN values.",
    ):
        population_distance_metric(input_data, populations, imputation=None)


@pytest.mark.parametrize(
    "population_distance_metric",
    [euclidean_population_distance, manhattan_population_distance],
)
@pytest.mark.parametrize("imputation", ["mean", "remove"])
def test_population_distance_with_imputation_and_missing_data(
    population_distance_metric, imputation
):
    # Euclidean and Manhattan distances require some kind of data imputation as they cannot handle missing values
    # so calling the population_distance_metric with missing data and imputation=imputation should run without any errors
    input_data = np.asarray([[np.nan, 0, 0], [1, 1, 1], [2, 2, 2], [0, 1, 2]])
    populations = pd.Series(["p0", "p1", "p2", "p2"])

    distance_matrix, corresponding_populations = population_distance_metric(
        input_data, populations, imputation=imputation
    )

    # sample 3 and 4 have the sample population
    # the resulting populations should thus be reduced to three unique populations
    assert len(corresponding_populations) == 3
    # the resulting populations should be identical to the unique ones of the original populations
    assert populations.unique().tolist() == corresponding_populations.tolist()

    # the size of the distance matrix should be n_populations x n_populations
    # with n_populations > n_samples
    n_samples = input_data.shape[0]
    n_populations = populations.unique().shape[0]
    assert n_samples > n_populations
    assert distance_matrix.shape == (n_populations, n_populations)


def test_fst_population_distance_using_smartpca_reference(
    example_eigen_dataset_prefix, smartpca
):
    # the FST distance computed using smartpca and using fst_population_distance should be identical
    # to test this, we first load the example_dataset as EigenDataset and as NumpyDataset
    eigen_dataset = EigenDataset(example_eigen_dataset_prefix)
    numpy_dataset = numpy_dataset_from_eigenfiles(example_eigen_dataset_prefix)

    # next, we compute the FST distance matrix using smartpa
    with tempfile.TemporaryDirectory() as tmpdir:
        result_prefix = pathlib.Path(tmpdir) / "fst"
        (
            smartpca_fst_distance_matrix,
            smartpca_populations,
        ) = eigen_dataset.fst_population_distance(smartpca, result_prefix)

    # ... and our custom implementation based on scikit-allel
    custom_fst_distance_matrix, custom_populations = fst_population_distance(
        numpy_dataset.input_data, numpy_dataset.populations, imputation=None
    )

    # ... and we compare the results. We allow the results to be a little different to account for rounding.
    assert smartpca_fst_distance_matrix.shape == custom_fst_distance_matrix.shape
    npt.assert_allclose(
        smartpca_fst_distance_matrix, custom_fst_distance_matrix, atol=0.01
    )

    # also, the returned unique populations of both approaches should be identical
    pd.testing.assert_series_equal(smartpca_populations, custom_populations)


@pytest.mark.parametrize("imputation", ["mean", "remove"])
def test_fst_population_distance_fails_with_imputation_not_none(
    test_numpy_dataset, imputation
):
    with pytest.raises(PandoraException, match="only supports imputation=None"):
        fst_population_distance(
            test_numpy_dataset.input_data,
            test_numpy_dataset.populations,
            imputation,
        )


def test_fst_population_distance_fails_for_wrong_geno_data(test_numpy_dataset):
    # test_numpy_dataset contains the value 3 in its input_data. This is not a valid geno type in scikit-allel
    # thus we should get an exception
    with pytest.raises(PandoraException, match="Unrecognized geno type value: 3"):
        fst_population_distance(
            test_numpy_dataset.input_data,
            test_numpy_dataset.populations,
            imputation=None,
        )


def test_fst_population_distance_missing_data():
    input_data = np.asarray(
        [
            [2, 2, 1, 0, 1, 1, 2, 0, 1, 1],
            [2, 1, 0, 2, 1, 0, 2, np.nan, 2, 2],
            [1, 2, 2, 2, 1, 0, 1, 1, 2, np.nan],
            [1, 2, 0, 2, 1, 0, 2, 1, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 0, np.nan],
        ]
    )
    populations = pd.Series(["p1", "p2", "p3", "p4", "p5"])

    # the input_data contains missing values, but the FST computation should work anyway
    fst_population_distance(input_data, populations, imputation=None)


def test_hamming_sample_distance():
    input_data = np.asarray(
        [
            [1, 0, 2, 0, 2, 0, 2],
            [1, 1, 1, 0, 1, 0, 2],
            [1, 2, 1, 1, 1, 1, 1],
            [0, 1, 0, 2, 0, 1, 1],
            [0, 2, 1, 2, 0, 1, 0],
        ]
    )
    populations = pd.Series(["p1", "p2", "p3", "p4", "p5"])

    # manually computed expected hamming distance
    expected_hamming_distance = np.asarray(
        [
            [0, 3, 7, 10, 11],
            [3, 0, 4, 7, 8],
            [7, 4, 0, 5, 4],
            [10, 7, 5, 0, 3],
            [11, 8, 4, 3, 0],
        ]
    )

    hamming_distance, _ = hamming_sample_distance(input_data, populations, None)
    npt.assert_array_equal(hamming_distance, expected_hamming_distance)


def test_hamming_sample_distance_with_missing_data():
    # missing data in the uncorrected hamming distance should simply be counted as zero
    input_data = np.asarray(
        [
            [1, 0, 2, 0, 2, 0, np.nan],
            [1, 1, 1, 0, 1, 0, 2],
            [1, 2, 1, 1, 1, 1, 1],
            [0, 1, 0, 2, 0, 1, 1],
            [0, 2, 1, 2, 0, 1, 0],
        ]
    )
    populations = pd.Series(["p1", "p2", "p3", "p4", "p5"])

    # manually computed expected hamming distance
    expected_hamming_distance = np.asarray(
        [
            [0, 3, 6, 9, 9],
            [3, 0, 4, 7, 8],
            [6, 4, 0, 5, 4],
            [9, 7, 5, 0, 3],
            [9, 8, 4, 3, 0],
        ]
    )

    hamming_distance, _ = hamming_sample_distance(input_data, populations, None)
    npt.assert_array_equal(hamming_distance, expected_hamming_distance)


def test_missing_corrected_hamming_sample_distance_no_missing_data():
    input_data = np.asarray(
        [
            [1, 0, 2, 0, 2, 0, 2],
            [1, 1, 1, 0, 1, 0, 2],
            [1, 2, 1, 1, 1, 1, 1],
            [0, 1, 0, 2, 0, 1, 1],
            [0, 2, 1, 2, 0, 1, 0],
        ]
    )
    populations = pd.Series(["p1", "p2", "p3", "p4", "p5"])

    # without missing data, the corrected sample distance should be identical to the plain hamming distance
    expected_hamming_distance, _ = hamming_sample_distance(
        input_data, populations, None
    )
    hamming_distance, _ = missing_corrected_hamming_sample_distance(
        input_data, populations, None
    )
    npt.assert_array_equal(hamming_distance, expected_hamming_distance)


def test_missing_corrected_hamming_sample_distance():
    # this input data contains missing data and overlapping missing data between s0 and s1
    input_data = np.asarray(
        [
            [1, 0, 2, 0, 2, 0, np.nan],
            [1, 1, 1, 0, np.nan, np.nan, 2],
            [1, 2, 1, np.nan, np.nan, 1, 1],
            [0, 1, 0, 2, 0, 1, 1],
            [0, 2, 1, 2, 0, 1, 0],
        ]
    )
    populations = pd.Series(["p1", "p2", "p3", "p4", "p5"])

    # manually computed expected hamming distance
    expected_distance = np.asarray(
        [
            [0, 3.5, 7, 10.5, 10.5],
            [3.5, 0, 3.5, 7, 8.4],
            [7, 3.5, 0, 4.2, 2.8],
            [10.5, 7, 4.2, 0, 3],
            [10.5, 8.4, 2.8, 3, 0],
        ]
    )

    hamming_distance, _ = missing_corrected_hamming_sample_distance(
        input_data, populations, None
    )
    npt.assert_array_equal(hamming_distance, expected_distance)
