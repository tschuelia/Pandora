import numpy as np
import pandas as pd
import pathlib
import pytest

from pandora.custom_errors import PandoraException
from pandora.pca import *


class TestPCA:
    def test_init_with_pandas_dataframe(self):
        n_pcs = 2

        explained_variances = np.asarray([0.0, 0.0])

        data = pd.DataFrame(
            {
                "PC0": [0.1, 0.2, 0.3],
                "PC1": [0.4, 0.5, 0.6],
                "sample_id": ["sample1", "sample2", "sample3"],
                "population": ["pop1", "pop2", "pop3"]
            }
        )

        pca = PCA(data, explained_variances, n_pcs)

        assert isinstance(pca, PCA)
        assert isinstance(pca.pca_data, pd.DataFrame)

        # incorrect dimensions of explained variances
        with pytest.raises(PandoraException, match="1D numpy array"):
            PCA(data, np.asarray([[0.0, 0.0]]), n_pcs)

        # incorrect number of explained variances
        with pytest.raises(PandoraException, match="Explained variance required for each PC"):
            PCA(data, np.asarray([0.0, 0.0, 0.0]), n_pcs)

        # missing sample_id column
        with pytest.raises(PandoraException, match="`sample_id`"):
            data = pd.DataFrame(
                {
                    "PC0": [0.1, 0.2, 0.3],
                    "PC1": [0.4, 0.5, 0.6],
                    "population": ["pop1", "pop2", "pop3"]
                }
            )
            PCA(data, explained_variances, n_pcs)

        # missing population column
        with pytest.raises(PandoraException, match="`population`"):
            data = pd.DataFrame(
                {
                    "PC0": [0.1, 0.2, 0.3],
                    "PC1": [0.4, 0.5, 0.6],
                    "sample_id": ["sample1", "sample2", "sample3"],
                }
            )
            PCA(data, explained_variances, n_pcs)

        # incorrect number of columns compared to n_pcs
        with pytest.raises(PandoraException, match="One data column required for each PC"):
            data = pd.DataFrame(
                {
                    "PC0": [0.1, 0.2, 0.3],
                    "PC1": [0.4, 0.5, 0.6],
                    "PC2": [0.4, 0.5, 0.6],
                    "sample_id": ["sample1", "sample2", "sample3"],
                    "population": ["pop1", "pop2", "pop3"]
                }
            )
            PCA(data, explained_variances, n_pcs)

        # incorrect PC columns
        with pytest.raises(PandoraException, match="Expected all of the following columns"):
            data = pd.DataFrame(
                {
                    "PC0": [0.1, 0.2, 0.3],
                    "wrong_name": [0.4, 0.5, 0.6],
                    "sample_id": ["sample1", "sample2", "sample3"],
                    "population": ["pop1", "pop2", "pop3"]
                }
            )
            # column PC1 incorrectly named
            PCA(data, explained_variances, n_pcs)


def test_check_smartpca_results_passes_for_correct_results(correct_smartpca_result_prefix):
    evec = pathlib.Path(f"{correct_smartpca_result_prefix}.evec")
    eval = pathlib.Path(f"{correct_smartpca_result_prefix}.eval")

    # should run without any issues
    check_smartpca_results(evec, eval)


# def test_check_smartpca_resuts_fails_for_incorrect_results():
#     evec_temp
#     evec = pathlib.Path(f"{incorrect_smartpca_npcs_result_prefix}.evec")
#     eval = pathlib.Path(f"{incorrect_smartpca_npcs_result_prefix}.eval")