import tempfile

import pandas as pd
import pytest

from pandora.dataset import *
from pandora.dataset import _deduplicate_snp_id
from pandora.distance_metrics import DISTANCE_METRICS
from pandora.embedding import PCA


class TestEigenDataset:
    def test_get_embedding_populations(self, example_population_list):
        populations = get_embedding_populations(example_population_list)
        pd.testing.assert_series_equal(populations, pd.Series(["Pop1", "Pop2", "Pop3"]))

    def test_get_windows(self, example_dataset):
        n_snps = example_dataset.get_sequence_length()
        n_windows = 3

        expected_overlap = int((n_snps / n_windows) / 2)
        expected_stride = int(n_snps / n_windows)
        expected_window_size = expected_stride + expected_overlap

        with tempfile.TemporaryDirectory() as tmpdir:
            window_dir = pathlib.Path(tmpdir)
            windows = example_dataset.get_windows(window_dir, n_windows=n_windows)

            assert len(windows) == n_windows
            assert all(isinstance(w, EigenDataset) for w in windows)

            # all windows should be in correct EIGENSTRAT format
            for w in windows:
                w.check_files()

            # all windows except the last one should have length expected_window_size
            for w in windows[:-1]:
                assert w.get_sequence_length() == expected_window_size

            # the last window should have less than or equal the expected_window_size
            assert windows[-1].get_sequence_length() <= expected_window_size

            # now let's check that they are actually overlapping
            # for window 0 and window 1, the last / first expected_overlap SNPs should be identical
            window0_genos = windows[0]._geno_file.open().readlines()
            window0_genos = [list(line.strip()) for line in window0_genos]
            window0_genos = np.asarray(window0_genos).T

            window1_genos = windows[1]._geno_file.open().readlines()
            window1_genos = [list(line.strip()) for line in window1_genos]
            window1_genos = np.asarray(window1_genos).T

            overlap_window0 = window0_genos[:, expected_overlap + 1 :]
            overlap_window1 = window1_genos[:, :expected_overlap]

            assert overlap_window0.shape == overlap_window1.shape
            assert overlap_window0.shape[1] == expected_overlap

            assert np.all(overlap_window0 == overlap_window1)

    def test_get_projected_samples_for_dataset_without_projections(
        self, example_dataset
    ):
        # example_dataset is initialized without embedding_populations
        # so projected_samples should be empty
        projected_samples = example_dataset.get_projected_samples()
        assert projected_samples.empty

    def test_get_projected_samples(self, example_dataset_with_poplist):
        # Pop4 is not listed in the embeddings populations, so example_dataset_with_poplist.get_projected_samples()
        # should return non-empty series with two samples (SAMPLE3 and SAMPLE4)
        projected_samples = example_dataset_with_poplist.get_projected_samples()
        assert projected_samples.shape[0] == 2

    @pytest.mark.parametrize("seed", [0, 10, 100, 885440])
    def test_bootstrap_files_correct(self, example_dataset, seed):
        """
        Tests whether the bootstrapped .geno, .ind, and .snp files have the correct file file_format
        """
        in_ind_count = sum(1 for _ in example_dataset._ind_file.open())
        in_geno_count = sum(1 for _ in example_dataset._geno_file.open())
        in_snp_count = sum(1 for _ in example_dataset._snp_file.open())

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            bootstrap_prefix = tmpdir / "bootstrap"
            bootstrap = example_dataset.bootstrap(
                bootstrap_prefix, seed=seed, redo=True
            )

            # check that the number of lines is correct
            bs_ind_count = sum(1 for _ in bootstrap._ind_file.open())
            bs_geno_count = sum(1 for _ in bootstrap._geno_file.open())
            bs_snp_count = sum(1 for _ in bootstrap._snp_file.open())

            assert in_ind_count == bs_ind_count
            assert in_geno_count == bs_geno_count
            assert in_snp_count == bs_snp_count

            # check that all files are correctly formatted
            bootstrap.check_files()

    def test_bootstrap_files_correct_with_redo(self, example_dataset):
        in_ind_count = sum(1 for _ in example_dataset._ind_file.open())
        in_geno_count = sum(1 for _ in example_dataset._geno_file.open())
        in_snp_count = sum(1 for _ in example_dataset._snp_file.open())

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            bootstrap_prefix = tmpdir / "bootstrap"
            _ = example_dataset.bootstrap(bootstrap_prefix, seed=42, redo=True)
            # do bootstrap again and check if the files are correctly redone
            bootstrap = example_dataset.bootstrap(bootstrap_prefix, seed=42, redo=True)

            # check that the number of lines is correct
            bs_ind_count = sum(1 for _ in bootstrap._ind_file.open())
            bs_geno_count = sum(1 for _ in bootstrap._geno_file.open())
            bs_snp_count = sum(1 for _ in bootstrap._snp_file.open())

            assert in_ind_count == bs_ind_count
            assert in_geno_count == bs_geno_count
            assert in_snp_count == bs_snp_count

            # check that all files are correctly formatted
            bootstrap.check_files()

    def test_bootstrap_using_existing_files(self, example_dataset):
        """
        Test that when setting redo=False, the resulting bootstrap dataset files
        are identical to the example_dataset files and no bootstrapping is performed
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            bootstrap_prefix = tmpdir / "bootstrap"

            # make sure we do not overwrite the correct input example_dataset files
            shutil.copy(example_dataset._ind_file, f"{bootstrap_prefix}.ind")
            shutil.copy(example_dataset._geno_file, f"{bootstrap_prefix}.geno")
            shutil.copy(example_dataset._snp_file, f"{bootstrap_prefix}.snp")

            bootstrap = example_dataset.bootstrap(bootstrap_prefix, seed=0, redo=False)

            # compare example_dataset and bootstrap contents
            assert (
                example_dataset._ind_file.open().read()
                == bootstrap._ind_file.open().read()
            )
            assert (
                example_dataset._geno_file.open().read()
                == bootstrap._geno_file.open().read()
            )
            assert (
                example_dataset._snp_file.open().read()
                == bootstrap._snp_file.open().read()
            )

            # check that all files are correctly formatted
            bootstrap.check_files()

    @pytest.mark.parametrize("seed", [0, 10, 100, 885440])
    def test_bootstrap_using_checkpoint(self, example_dataset, seed):
        """
        Test that when setting redo=False, and the results files do not exist, but a checkpoint does
        that the bootstrap based on the checkpoint is identical
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            bootstrap_prefix = tmpdir / "bootstrap"
            bootstrap = example_dataset.bootstrap(
                bootstrap_prefix, seed=seed, redo=True
            )

            bootstrap_backup_prefix = tmpdir / "backup"
            backup_ind = pathlib.Path(f"{bootstrap_backup_prefix}.ind")
            backup_geno = pathlib.Path(f"{bootstrap_backup_prefix}.geno")
            backup_snp = pathlib.Path(f"{bootstrap_backup_prefix}.snp")

            # move bootstrap's geno, ind, snp files such that we can delete them, but compare them later
            shutil.move(bootstrap._ind_file, backup_ind)
            shutil.move(bootstrap._geno_file, backup_geno)
            shutil.move(bootstrap._snp_file, backup_snp)

            # make sure moving actually worked such that a subsequent bootstrap run will use the checkpoint
            assert not bootstrap.files_exist()

            # make sure the checkpoint file exits
            ckp_file = pathlib.Path(f"{bootstrap_prefix}.ckp")
            assert ckp_file.exists()

            # rerun the bootstrap, this time use a different seed to reduce the chance of creating identical
            # files randomly
            bootstrap = example_dataset.bootstrap(
                bootstrap_prefix, seed=seed + 1, redo=False
            )

            # compare the file contents to make sure the checkpoint was actually used
            assert backup_ind.open().read() == bootstrap._ind_file.open().read()
            assert backup_geno.open().read() == bootstrap._geno_file.open().read()
            assert backup_snp.open().read() == bootstrap._snp_file.open().read()

    def test_deduplicate_snp_id(self):
        n_ids = 5
        seen_ids = set()
        for i, snp_id in enumerate(n_ids * ["snp_id"]):
            deduplicate = _deduplicate_snp_id(snp_id, seen_ids)
            seen_ids.add(deduplicate)

            if i > 0:
                assert f"r{i}" in deduplicate

        assert len(seen_ids) == n_ids

    @pytest.mark.parametrize(
        "embedding", [EmbeddingAlgorithm.PCA, EmbeddingAlgorithm.MDS]
    )
    @pytest.mark.parametrize("keep_files", [True, False])
    def test_bootstrap_and_embed_multiple(
        self, example_dataset, smartpca, embedding, keep_files
    ):
        n_bootstraps = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            bootstraps = bootstrap_and_embed_multiple(
                dataset=example_dataset,
                n_bootstraps=n_bootstraps,
                result_dir=tmpdir,
                smartpca=smartpca,
                embedding=embedding,
                n_components=2,
                seed=0,
                threads=1,
                keep_bootstraps=keep_files,
            )

            assert len(bootstraps) == n_bootstraps

            if embedding == EmbeddingAlgorithm.PCA:
                # each bootstrap should have embedding.pca != None, but embedding.mds == None
                assert all(b.pca is not None for b in bootstraps)
                assert all(isinstance(b.pca, PCA) for b in bootstraps)
                assert all(b.mds is None for b in bootstraps)
            elif embedding == EmbeddingAlgorithm.MDS:
                # each bootstrap should have embedding.mds != None, but embedding.pca == None
                assert all(b.mds is not None for b in bootstraps)
                assert all(isinstance(b.mds, MDS) for b in bootstraps)
                assert all(b.pca is None for b in bootstraps)

            # make sure that all files are present if keep_files == True, otherwise check that they are deleted
            if keep_files:
                assert all(b.files_exist() for b in bootstraps)
            else:
                assert not any(b.files_exist() for b in bootstraps)

    @pytest.mark.parametrize(
        "embedding", [EmbeddingAlgorithm.PCA, EmbeddingAlgorithm.MDS]
    )
    @pytest.mark.parametrize("keep_files", [True, False])
    def test_sliding_window_embedding(
        self, example_sliding_window_dataset, smartpca, embedding, keep_files
    ):
        n_windows = 5
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            windows = sliding_window_embedding(
                dataset=example_sliding_window_dataset,
                n_windows=n_windows,
                result_dir=tmpdir,
                smartpca=smartpca,
                embedding=embedding,
                n_components=2,
                threads=1,
                keep_windows=keep_files,
            )

            assert len(windows) == n_windows

            if embedding == EmbeddingAlgorithm.PCA:
                # each window should have embedding.pca != None, but embedding.mds == None
                assert all(w.pca is not None for w in windows)
                assert all(isinstance(w.pca, PCA) for w in windows)
                assert all(w.mds is None for w in windows)
            elif embedding == EmbeddingAlgorithm.MDS:
                # each window should have embedding.mds != None, but embedding.pca == None
                assert all(w.mds is not None for w in windows)
                assert all(isinstance(w.mds, MDS) for w in windows)
                assert all(w.pca is None for w in windows)

            # make sure that all files are present if keep_files == True, otherwise check that they are deleted
            if keep_files:
                assert all(w.files_exist() for w in windows)
            else:
                assert not any(w.files_exist() for w in windows)


class TestEigenDatasetPCA:
    def test_smartpca_finished(self, correct_smartpca_result_prefix):
        n_pcs = 2
        assert smartpca_finished(n_pcs, correct_smartpca_result_prefix)

    def test_smartpca_not_finished(self, unfinished_smartpca_result_prefix):
        n_pcs = 2
        assert not smartpca_finished(n_pcs, unfinished_smartpca_result_prefix)

    def test_smartpca_wrong_n_pcs(self, incorrect_smartpca_npcs_result_prefix):
        n_pcs = 2
        assert not smartpca_finished(n_pcs, incorrect_smartpca_npcs_result_prefix)

    def test_smartpca_results_missing(self, missing_smartpca_result_prefix):
        n_pcs = 2
        assert not smartpca_finished(n_pcs, missing_smartpca_result_prefix)

    def test_smartpca(self, smartpca, example_dataset):
        n_pcs = 2
        with tempfile.TemporaryDirectory() as result_dir:
            assert example_dataset.pca is None
            result_dir = pathlib.Path(result_dir)
            example_dataset.run_pca(smartpca, n_pcs, result_dir)
            assert isinstance(example_dataset.pca, PCA)

    def test_smartpca_with_additional_settings(self, smartpca, example_dataset):
        n_pcs = 2
        smartpca_settings = {"numoutlieriter": 0, "shrinkmode": True}
        with tempfile.TemporaryDirectory() as result_dir:
            assert example_dataset.pca is None
            result_dir = pathlib.Path(result_dir)
            example_dataset.run_pca(
                smartpca,
                n_pcs,
                result_dir,
                smartpca_optional_settings=smartpca_settings,
            )
            assert isinstance(example_dataset.pca, PCA)

    def test_smartpca_from_existing(self, smartpca, example_dataset):
        n_pcs = 2
        with tempfile.TemporaryDirectory() as result_dir:
            assert example_dataset.pca is None
            result_dir = pathlib.Path(result_dir)
            example_dataset.run_pca(smartpca, n_pcs, result_dir)
            assert isinstance(example_dataset.pca, PCA)

            # rerun with redo=False to make sure loading from previous finished runs works
            example_dataset.run_pca(smartpca, n_pcs, result_dir, redo=False)

    def test_smartpca_with_poplist(self, smartpca, example_dataset_with_poplist):
        n_pcs = 2
        with tempfile.TemporaryDirectory() as result_dir:
            assert example_dataset_with_poplist.pca is None
            result_dir = pathlib.Path(result_dir)
            example_dataset_with_poplist.run_pca(smartpca, n_pcs, result_dir)
            assert isinstance(example_dataset_with_poplist.pca, PCA)

    def test_smartpca_ignores_nonsense_setting(self, smartpca, example_dataset):
        n_pcs = 2
        smartpca_settings = {
            "nonsenseSetting": "fail",
        }
        with tempfile.TemporaryDirectory() as result_dir:
            assert example_dataset.pca is None
            result_dir = pathlib.Path(result_dir)
            example_dataset.run_pca(
                smartpca,
                n_pcs,
                result_dir,
                smartpca_optional_settings=smartpca_settings,
            )
            assert isinstance(example_dataset.pca, PCA)

    def test_smartpca_fails_for_too_many_components(self, smartpca, example_dataset):
        n_pcs = example_dataset.get_sequence_length() + 10

        with pytest.raises(
            PandoraException, match="Number of Principal Components needs to be smaller"
        ):
            example_dataset.run_pca(smartpca, n_pcs)


class TestNumpyDataset:
    def test_init(self):
        test_data = np.asarray([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        sample_ids = pd.Series(["sample1", "sample2", "sample3"])
        populations = pd.Series(["population1", "population2", "population3"])

        # everything should work fine with no error
        dataset = NumpyDataset(test_data, sample_ids, populations)
        np.testing.assert_equal(dataset.input_data, test_data)
        pd.testing.assert_series_equal(dataset.sample_ids, sample_ids)
        pd.testing.assert_series_equal(dataset.populations, populations)
        assert dataset.pca is None
        assert dataset.mds is None

        # now change the number of populations to mismatch, this should cause an error
        populations = pd.Series(["population1", "population2"])

        with pytest.raises(PandoraException, match="a population for each sample"):
            NumpyDataset(test_data, sample_ids, populations)

        # now change the number of sample_ids in test_data to mismatch sample_ids, this should cause an error
        populations = pd.Series(["population1", "population2", "population3"])
        test_data = np.asarray([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])

        with pytest.raises(PandoraException, match="a sample ID for each sample"):
            NumpyDataset(test_data, sample_ids, populations)

    @pytest.mark.parametrize("impute_missing", [True, False])
    @pytest.mark.parametrize("imputation", ["mean", "remove"])
    def test_run_pca(self, test_numpy_dataset, impute_missing, imputation):
        n_pcs = 2
        test_numpy_dataset.run_pca(
            n_pcs, impute_missing=impute_missing, imputation=imputation
        )
        # dataset should now have a PCA object attached with embedding data (n_samples, n_pcs)
        # dataset.mds should still be None
        assert test_numpy_dataset.pca is not None
        assert test_numpy_dataset.mds is None
        assert isinstance(test_numpy_dataset.pca, PCA)
        assert test_numpy_dataset.pca.embedding_matrix.shape == (
            test_numpy_dataset.input_data.shape[0],
            n_pcs,
        )
        pd.testing.assert_series_equal(
            test_numpy_dataset.pca.sample_ids,
            test_numpy_dataset.sample_ids,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            test_numpy_dataset.pca.populations,
            test_numpy_dataset.populations,
            check_names=False,
        )

    def test_run_pca_fails_for_too_many_pcs(self, test_numpy_dataset):
        n_pcs = test_numpy_dataset.input_data.shape[1] + 10

        with pytest.raises(
            PandoraException, match="Number of Principal Components needs to be smaller"
        ):
            test_numpy_dataset.run_pca(n_pcs)

    def test_bootstrap(self, test_numpy_dataset):
        bootstrap = test_numpy_dataset.bootstrap(0)

        assert isinstance(bootstrap, NumpyDataset)
        assert bootstrap.input_data.shape == test_numpy_dataset.input_data.shape
        pd.testing.assert_series_equal(
            bootstrap.sample_ids, test_numpy_dataset.sample_ids
        )
        pd.testing.assert_series_equal(
            bootstrap.populations, test_numpy_dataset.populations
        )

    def test_get_windows(self, test_numpy_dataset):
        n_snps = test_numpy_dataset.input_data.shape[1]
        n_windows = 3

        expected_overlap = int((n_snps / n_windows) / 2)
        expected_stride = int(n_snps / n_windows)
        expected_window_size = expected_stride + expected_overlap

        windows = test_numpy_dataset.get_windows(n_windows)

        assert len(windows) == n_windows
        assert all(isinstance(w, NumpyDataset) for w in windows)

        # all windows except the last one should have length expected_window_size
        for w in windows[:-1]:
            assert w.input_data.shape[1] == expected_window_size

        # the last window should have less than or equal the expected_window_size
        assert windows[-1].input_data.shape[1] <= expected_window_size

        # now let's check that they are actually overlapping
        # for window 0 and window 1, the last / first expected_overlap SNPs should be identical
        window0_genos = windows[0].input_data
        window1_genos = windows[1].input_data

        overlap_window0 = window0_genos[:, expected_overlap + 1 :]
        overlap_window1 = window1_genos[:, :expected_overlap]

        assert overlap_window0.shape == overlap_window1.shape
        assert overlap_window0.shape[1] == expected_overlap

        assert np.all(overlap_window0 == overlap_window1)

    @pytest.mark.parametrize(
        "embedding", [EmbeddingAlgorithm.PCA, EmbeddingAlgorithm.MDS]
    )
    @pytest.mark.parametrize("impute_missing", [True, False])
    def test_bootstrap_and_embed_multiple_numpy(
        self, test_numpy_dataset, embedding, impute_missing
    ):
        n_bootstraps = 2
        bootstraps = bootstrap_and_embed_multiple_numpy(
            dataset=test_numpy_dataset,
            n_bootstraps=n_bootstraps,
            embedding=embedding,
            n_components=2,
            seed=0,
            threads=2,
            impute_missing=impute_missing,
            missing_value=0,
            # we don't test for 'remove' here because we have a small dataset and we might end up getting an error
            # because the bootstrapped dataset contains only nan-columns
            imputation="mean",
        )

        assert len(bootstraps) == n_bootstraps

        if embedding == EmbeddingAlgorithm.PCA:
            # each bootstrap should have embedding.pca != None, but embedding.mds == None
            assert all(b.pca is not None for b in bootstraps)
            assert all(isinstance(b.pca, PCA) for b in bootstraps)
            assert all(b.mds is None for b in bootstraps)
        elif embedding == EmbeddingAlgorithm.MDS:
            # each bootstrap should have embedding.mds != None, but embedding.pca == None
            assert all(b.mds is not None for b in bootstraps)
            assert all(isinstance(b.mds, MDS) for b in bootstraps)
            assert all(b.pca is None for b in bootstraps)

    @pytest.mark.parametrize(
        "embedding", [EmbeddingAlgorithm.PCA, EmbeddingAlgorithm.MDS]
    )
    @pytest.mark.parametrize("impute_missing", [True, False])
    def test_sliding_window_embedding_numpy(
        self, test_numpy_dataset_sliding_window, embedding, impute_missing
    ):
        n_windows = 4
        sliding_windows = sliding_window_embedding_numpy(
            dataset=test_numpy_dataset_sliding_window,
            n_windows=n_windows,
            embedding=embedding,
            n_components=2,
            threads=2,
            impute_missing=impute_missing,
            missing_value=0,
            # we don't test for 'remove' here because we have a small dataset and we might end up getting an error
            # because the windowed datasets contains only nan-columns
            imputation="mean",
        )

        assert len(sliding_windows) == n_windows

        if embedding == EmbeddingAlgorithm.PCA:
            # each window should have embedding.pca != None, but embedding.mds == None
            assert all(w.pca is not None for w in sliding_windows)
            assert all(isinstance(w.pca, PCA) for w in sliding_windows)
            assert all(w.mds is None for w in sliding_windows)
        elif embedding == EmbeddingAlgorithm.MDS:
            # each window should have embedding.mds != None, but embedding.pca == None
            assert all(w.mds is not None for w in sliding_windows)
            assert all(isinstance(w.mds, MDS) for w in sliding_windows)
            assert all(w.pca is None for w in sliding_windows)

    @pytest.mark.parametrize("distance_metric", DISTANCE_METRICS)
    @pytest.mark.parametrize("impute_missing", [True, False])
    @pytest.mark.parametrize("imputation", ["mean", "remove"])
    def test_run_mds(
        self,
        test_numpy_dataset,
        distance_metric,
        impute_missing,
        imputation,
    ):
        n_components = 2

        test_numpy_dataset.run_mds(
            n_components=n_components,
            distance_metric=distance_metric,
            impute_missing=impute_missing,
            imputation=imputation,
        )
        # test_numpy_dataset should now have an MDS object attached with embedding data (n_samples, n_components)
        # test_numpy_dataset.pca should still be None
        assert test_numpy_dataset.mds is not None
        assert test_numpy_dataset.pca is None
        assert isinstance(test_numpy_dataset.mds, MDS)
        assert test_numpy_dataset.mds.embedding_matrix.shape == (
            test_numpy_dataset.input_data.shape[0],
            n_components,
        )
        pd.testing.assert_series_equal(
            test_numpy_dataset.mds.sample_ids,
            test_numpy_dataset.sample_ids,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            test_numpy_dataset.mds.populations,
            test_numpy_dataset.populations,
            check_names=False,
        )

    def test_data_imputation(self):
        test_data = np.asarray([[np.nan, 1, 1], [2, np.nan, 2], [3, 3, 3]])
        sample_ids = pd.Series(["sample1", "sample2", "sample3"])
        populations = pd.Series(["population1", "population2", "population3"])
        dataset = NumpyDataset(test_data, sample_ids, populations)

        # imputation remove -> should result in a single column being left
        imputed_data = dataset._get_imputed_data(
            missing_value=np.nan, imputation="remove"
        )
        assert imputed_data.shape == (test_data.shape[0], 1)

        # mean imputation should result in the following data matrix:
        expected_imputed_data = np.asarray(
            [[2.5, 1, 1], [2, 2, 2], [3, 3, 3]], dtype="float"
        )
        imputed_data = dataset._get_imputed_data(
            missing_value=np.nan, imputation="mean"
        )
        np.testing.assert_equal(imputed_data, expected_imputed_data)

    def test_data_imputation_remove_fails_for_snps_with_all_nan(self):
        test_data = np.asarray([[np.nan, 1, 1], [2, np.nan, 2], [3, 3, np.nan]])
        sample_ids = pd.Series(["sample1", "sample2", "sample3"])
        populations = pd.Series(["population1", "population2", "population3"])
        dataset = NumpyDataset(test_data, sample_ids, populations)
        with pytest.raises(PandoraException, match="No data left after imputation."):
            dataset._get_imputed_data(missing_value=np.nan, imputation="remove")

    def test_numpy_dataset_from_eigenfiles(self, example_eigen_dataset_prefix):
        np_dataset = numpy_dataset_from_eigenfiles(example_eigen_dataset_prefix)

        assert isinstance(np_dataset, NumpyDataset)

        expected_geno = np.asarray(
            [
                [1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0],
                [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 2.0],
                [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0],
                [0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0],
            ]
        )
        np.testing.assert_equal(expected_geno, np_dataset.input_data)

        expected_sample_ids = pd.Series(
            ["SAMPLE0", "SAMPLE1", "SAMPLE2", "SAMPLE3", "SAMPLE4"]
        )
        pd.testing.assert_series_equal(np_dataset.sample_ids, expected_sample_ids)

        expected_populations = pd.Series(["Pop1", "Pop2", "Pop3", "Pop4", "Pop4"])
        pd.testing.assert_series_equal(np_dataset.populations, expected_populations)

        assert np_dataset.pca is None
        assert np_dataset.mds is None

    def test_numpy_dataset_from_eigenfiles_fails_for_packed_eigen_dataset(
        self, example_packed_eigen_dataset_prefix
    ):
        with pytest.raises(
            PandoraException,
            match="The provided dataset does not seem to be in EIGENSTRAT format.",
        ):
            numpy_dataset_from_eigenfiles(example_packed_eigen_dataset_prefix)

    def test_numpy_dataset_from_eigenfiles_fails_for_plink_dataset(
        self, example_ped_dataset_prefix
    ):
        with pytest.raises(PandoraException, match="Not all required input files"):
            numpy_dataset_from_eigenfiles(example_ped_dataset_prefix)
