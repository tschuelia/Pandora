from typing import Optional

import numpy as np
from numpy import typing as npt
from sklearn.impute import SimpleImputer

from pandora.custom_errors import PandoraException


def impute_data(input_data: npt.NDArray, imputation: Optional[str]) -> npt.NDArray:
    """Imputes missing values in the given input data using the given imputation strategy.

    Parameters
    ----------
    input_data : npt.NDArray
        Numpy array containing the input data to impute. Missing values are expected to be np.NaN.
    imputation : Optional[str]
        Imputation method to use. Available options are:\n
        - ``"mean"``: Imputes missing values with the average of the respective column.
        - ``"remove"``: Removes all columns with at least one missing value.
        - ``None``: Does not impute the given data.

    Returns
    -------
    imputed_data : npt.NDArray
        Imputed input data with imputation according to the specified method.

    Raises
    ------
    PandoraException
        - If no data is left in case of ``"remove"`` imputation strategy. That means that all columns in the input data contained at least one missing value.
        - If the imputation method is not supported.
    """
    if imputation is None:
        return input_data
    elif imputation == "mean":
        imputer = SimpleImputer()
        return imputer.fit_transform(input_data)
    elif imputation == "remove":
        # remove all columns containing at least one missing value
        imputed_data = input_data[:, ~np.isnan(input_data).any(axis=0)]

        if imputed_data.size == 0:
            raise PandoraException(
                "No data left after imputation. Every SNP seems to contain at least one missing value. "
                "Consider using a different dataset or set imputation='mean'."
            )

        return imputed_data
    else:
        raise PandoraException(
            f"Unrecognized imputation method {imputation}. "
            f"Allowed methods are 'mean', 'remove', and None."
        )
