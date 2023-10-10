from typing import Optional

import numpy as np
from numpy import typing as npt
from sklearn.impute import SimpleImputer

from pandora.custom_errors import PandoraException


def impute_data(input_data, imputation: Optional[str] = None) -> npt.NDArray:
    """
    TODO

    Parameters
    ----------
    input_data
    imputation

    Returns
    -------

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
