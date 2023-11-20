import numpy as np
import pandas as pd
import pytest
from numpy import testing as npt

from pandora.custom_errors import PandoraException
from pandora.dataset import NumpyDataset
from pandora.imputation import impute_data


def test_mean_imputation():
    input_data = np.asarray([[2, 2], [np.nan, 3], [4, 4]])
    expected_data = np.asarray([[2, 2], [3, 3], [4, 4]])
    imputed_data = impute_data(input_data, imputation="mean")
    npt.assert_array_equal(imputed_data, expected_data)


def test_remove_imputation():
    input_data = np.asarray([[2, 2], [np.nan, 3], [4, 4]])
    expected_data = np.asarray([[2], [3], [4]])
    imputed_data = impute_data(input_data, imputation="remove")
    npt.assert_array_equal(imputed_data, expected_data)


def test_remove_imputation_fails_if_no_data_left():
    input_data = np.asarray([[2, 2], [np.nan, 3], [4, np.nan]])

    # all columns have missing data, so imputation should raise a PandoraException
    with pytest.raises(PandoraException, match="No data left after imputation."):
        impute_data(input_data, imputation="remove")


def test_none_imputation_returns_unmodified_data():
    input_data = np.asarray([[2, 2], [np.nan, 3], [4, 4]])
    imputed_data = impute_data(input_data, imputation=None)
    npt.assert_array_equal(input_data, imputed_data)


def test_data_imputation():
    test_data = np.asarray([[np.nan, 1, 1], [2, np.nan, 2], [3, 3, 3]])
    sample_ids = pd.Series(["sample1", "sample2", "sample3"])
    populations = pd.Series(["population1", "population2", "population3"])
    dataset = NumpyDataset(test_data, sample_ids, populations)

    # imputation remove -> should result in a single column being left
    imputed_data = impute_data(
        dataset.input_data, imputation="remove", missing_value=dataset._missing_value
    )
    assert imputed_data.shape == (test_data.shape[0], 1), dataset.input_data

    # mean imputation should result in the following data matrix:
    expected_imputed_data = np.asarray(
        [[2.5, 1, 1], [2, 2, 2], [3, 3, 3]], dtype="float"
    )
    imputed_data = impute_data(
        dataset.input_data, imputation="mean", missing_value=dataset._missing_value
    )
    np.testing.assert_equal(imputed_data, expected_imputed_data)


def test_data_imputation_remove_fails_for_snps_with_all_nan():
    test_data = np.asarray([[np.nan, 1, 1], [2, np.nan, 2], [3, 3, np.nan]])
    sample_ids = pd.Series(["sample1", "sample2", "sample3"])
    populations = pd.Series(["population1", "population2", "population3"])
    dataset = NumpyDataset(test_data, sample_ids, populations)
    with pytest.raises(PandoraException, match="No data left after imputation."):
        impute_data(
            dataset.input_data,
            imputation="remove",
            missing_value=dataset._missing_value,
        )
