import numpy as np
import pytest
from numpy import testing as npt

from pandora.custom_errors import PandoraException
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
