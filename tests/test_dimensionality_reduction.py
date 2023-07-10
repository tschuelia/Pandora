import numpy as np
import pandas as pd
import pathlib
import pytest

from pandora.custom_errors import PandoraException
from pandora.dimensionality_reduction import *


class TestPCA:
    def test_init_with_pandas_dataframe(self):
        n_components = 2

        explained_variances = np.asarray([0.0, 0.0])

        data = pd.DataFrame(
            {
                "D0": [0.1, 0.2, 0.3],
                "D1": [0.4, 0.5, 0.6],
                "sample_id": ["sample1", "sample2", "sample3"],
                "population": ["pop1", "pop2", "pop3"]
            }
        )

        pca = PCA(data, n_components, explained_variances)

        assert isinstance(pca, PCA)
        assert isinstance(pca.embedding, pd.DataFrame)

        # incorrect dimensions of explained variances
        with pytest.raises(PandoraException, match="1D numpy array"):
            PCA(data, n_components, np.asarray([[0.0, 0.0]]))

        # incorrect number of explained variances
        with pytest.raises(PandoraException, match="Explained variance required for each PC"):
            PCA(data, n_components, np.asarray([0.0, 0.0, 0.0]))

        # missing sample_id column
        with pytest.raises(PandoraException, match="`sample_id`"):
            data = pd.DataFrame(
                {
                    "D0": [0.1, 0.2, 0.3],
                    "D1": [0.4, 0.5, 0.6],
                    "population": ["pop1", "pop2", "pop3"]
                }
            )
            PCA(data, n_components, explained_variances)

        # missing population column
        with pytest.raises(PandoraException, match="`population`"):
            data = pd.DataFrame(
                {
                    "D0": [0.1, 0.2, 0.3],
                    "D1": [0.4, 0.5, 0.6],
                    "sample_id": ["sample1", "sample2", "sample3"],
                }
            )
            PCA(data, n_components, explained_variances)

        # incorrect number of columns compared to n_components
        with pytest.raises(PandoraException, match="One data column required for each PC"):
            data = pd.DataFrame(
                {
                    "D0": [0.1, 0.2, 0.3],
                    "D1": [0.4, 0.5, 0.6],
                    "D2": [0.4, 0.5, 0.6],
                    "sample_id": ["sample1", "sample2", "sample3"],
                    "population": ["pop1", "pop2", "pop3"]
                }
            )
            PCA(data, n_components, explained_variances)

        # incorrect PC columns
        with pytest.raises(PandoraException, match="Expected all of the following columns"):
            data = pd.DataFrame(
                {
                    "D0": [0.1, 0.2, 0.3],
                    "wrong_name": [0.4, 0.5, 0.6],
                    "sample_id": ["sample1", "sample2", "sample3"],
                    "population": ["pop1", "pop2", "pop3"]
                }
            )
            # column PC1 incorrectly named
            PCA(data, n_components, explained_variances)


def test_check_smartpca_results_passes_for_correct_results(correct_smartpca_result_prefix):
    evec = pathlib.Path(f"{correct_smartpca_result_prefix}.evec")
    eval = pathlib.Path(f"{correct_smartpca_result_prefix}.eval")

    # should run without any issues
    check_smartpca_results(evec, eval)