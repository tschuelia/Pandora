import numpy as np
import pandas as pd
import pytest

from pandora.custom_errors import PandoraException
from pandora.pca import PCA


class TestPCA:
    def test_init_with_numpy_array(self):
        n_samples = 20
        n_pcs = 2

        explained_variances = [0.0, 0.0]

        # correct number of dimensions
        data = np.random.random_sample(size=(n_samples, n_pcs))
        pca = PCA(data, explained_variances, n_pcs)

        assert isinstance(pca, PCA)
        assert isinstance(pca.pca_data, pd.DataFrame)
        # we did not pass any sample IDs or populations, so all should be None
        assert ~any(pca.pca_data.sample_id)
        assert ~any(pca.pca_data.population)
        np.testing.assert_array_equal(pca.pc_vectors, data, strict=True)

        # mismatch in n_pcs and explained_variances
        with pytest.raises(PandoraException, match="Explained variance required for each PC"):
            PCA(data, explained_variances, n_pcs + 1)

        # incorrect number of dimensions
        data = np.random.random_sample(size=(n_samples,))
        with pytest.raises(PandoraException, match="two dimensional"):
            PCA(data, explained_variances, n_pcs)

        # incorrect number of PCs
        data = np.random.random_sample(size=(n_samples, n_pcs + 1))
        with pytest.raises(PandoraException, match="n_samples, n_pcs="):
            PCA(data, explained_variances, n_pcs)

    def test_init_with_pandas_dataframe(self):
        n_pcs = 2

        explained_variances = [0.0, 0.0]

        data = pd.DataFrame(
            {
                "PC0": [0.1, 0.2, 0.3],
                "PC1": [0.4, 0.5, 0.6]
            }
        )

        pca = PCA(data, explained_variances, n_pcs)

        assert isinstance(pca, PCA)
        assert isinstance(pca.pca_data, pd.DataFrame)
        # we did not pass any sample IDs or populations, so all should be None
        assert ~any(pca.pca_data.sample_id)
        assert ~any(pca.pca_data.population)

        # pass sample IDs
        sample_ids = ["sample1", "sample2", "sample3"]
        pca = PCA(data, explained_variances, n_pcs, sample_ids)
        assert all(pca.pca_data.sample_id == sample_ids)
        assert ~any(pca.pca_data.population)

        # pass incorrect number of sample IDs
        sample_ids = ["sample1", "sample2"]
        with pytest.raises(PandoraException, match="One sample ID required for each sample."):
            PCA(data, explained_variances, n_pcs, sample_ids)

        # pass populations
        populations = ["pop1", "pop2", "pop3"]
        pca = PCA(data, explained_variances, n_pcs, None, populations)
        assert all(pca.pca_data.population == populations)
        assert ~any(pca.pca_data.sample_id)

        # pass incorrect number of populations
        populations = ["pop1", "pop2"]
        with pytest.raises(PandoraException, match="One population required for each sample."):
            PCA(data, explained_variances, n_pcs, None, populations)

    def test_init_with_pandas_dataframe_containing_samples_and_populations(self):
        n_pcs = 2

        explained_variances = [0.0, 0.0]

        sample_ids = ["sample1", "sample2", "sample3"]
        populations = ["pop1", "pop2", "pop3"]

        data = pd.DataFrame(
            {
                "PC0": [0.1, 0.2, 0.3],
                "PC1": [0.4, 0.5, 0.6],
                "sample_id": sample_ids,
                "population": populations
            }
        )

        pca = PCA(data.copy(), explained_variances, n_pcs)

        # make sure the sample IDs and populations are not overwritten
        assert all(pca.pca_data.sample_id == sample_ids)
        assert all(pca.pca_data.population == populations)

        # pass new sample IDs and make sure they are correctly set
        new_sample_ids = ["new1", "new2", "new3"]
        pca = PCA(data.copy(), explained_variances, n_pcs, new_sample_ids)
        assert all(pca.pca_data.sample_id == new_sample_ids)
        # populations should not change
        assert all(pca.pca_data.population == populations)

        # pass new populations and make sure they are correctly set
        new_populations = ["newPop1", "newPop2", "newPop3"]
        pca = PCA(data.copy(), explained_variances, n_pcs, None, new_populations)
        assert all(pca.pca_data.population == new_populations)
        # sample IDs should not change
        assert all(pca.pca_data.sample_id == sample_ids)
