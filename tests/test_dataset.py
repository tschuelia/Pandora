import tempfile
import pathlib
import pytest
import shutil

from pandora.dataset import *
from pandora.dimensionality_reduction import PCA


class TestDataset:
    def test_get_pca_populations(self, example_population_list):
        populations = get_embedding_populations(example_population_list)
        assert len(populations) == 4

    def test_sample_dataframe_correct(self, example_dataset_with_poplist):
        samples = example_dataset_with_poplist.samples

        # example.ind has 5 samples
        assert samples.shape[0] == 5
        # samples should have 4 columns: sample_id, sex, population, used_for_embedding
        assert samples.shape[1] == 4
        assert set(samples.columns) == {"sample_id", "sex", "population", "used_for_embedding"}

        # this dataset has a population list, so some samples should not be used_for_embedding
        assert not samples.used_for_embedding.all()

        # each sample in this example dataset has a unique population
        assert samples.population.unique().shape[0] == samples.shape[0]


class TestDatasetBootstrap:
    @pytest.mark.parametrize("seed", [0, 10, 100, 885440])
    def test_bootstrap_files_correct(self, example_dataset, seed):
        """
        Tests whether the bootstrapped .geno, .ind, and .snp files have the correct file format
        """
        in_ind_count = sum(1 for _ in example_dataset.ind_file.open())
        in_geno_count = sum(1 for _ in example_dataset.geno_file.open())
        in_snp_count = sum(1 for _ in example_dataset.snp_file.open())

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            bootstrap_prefix = tmpdir / "bootstrap"
            bootstrap = example_dataset.create_bootstrap(bootstrap_prefix, seed=seed, redo=True)

            # check that the number of lines is correct
            bs_ind_count = sum(1 for _ in bootstrap.ind_file.open())
            bs_geno_count = sum(1 for _ in bootstrap.geno_file.open())
            bs_snp_count = sum(1 for _ in bootstrap.snp_file.open())

            assert in_ind_count == bs_ind_count
            assert in_geno_count == bs_geno_count
            assert in_snp_count == bs_snp_count

            # check that all files are correctly formatted
            bootstrap.check_files()

    def test_bootstrap_files_correct_with_redo(self, example_dataset):
        in_ind_count = sum(1 for _ in example_dataset.ind_file.open())
        in_geno_count = sum(1 for _ in example_dataset.geno_file.open())
        in_snp_count = sum(1 for _ in example_dataset.snp_file.open())

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            bootstrap_prefix = tmpdir / "bootstrap"
            _ = example_dataset.create_bootstrap(bootstrap_prefix, seed=42, redo=True)
            # do bootstrap again and check if the files are correctly redone
            bootstrap = example_dataset.create_bootstrap(bootstrap_prefix, seed=42, redo=True)

            # check that the number of lines is correct
            bs_ind_count = sum(1 for _ in bootstrap.ind_file.open())
            bs_geno_count = sum(1 for _ in bootstrap.geno_file.open())
            bs_snp_count = sum(1 for _ in bootstrap.snp_file.open())

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
            shutil.copy(example_dataset.ind_file, f"{bootstrap_prefix}.ind")
            shutil.copy(example_dataset.geno_file, f"{bootstrap_prefix}.geno")
            shutil.copy(example_dataset.snp_file, f"{bootstrap_prefix}.snp")

            bootstrap = example_dataset.create_bootstrap(bootstrap_prefix, seed=0, redo=False)

            # compare example_dataset and bootstrap contents
            assert example_dataset.ind_file.open().read() == bootstrap.ind_file.open().read()
            assert example_dataset.geno_file.open().read() == bootstrap.geno_file.open().read()
            assert example_dataset.snp_file.open().read() == bootstrap.snp_file.open().read()

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
            bootstrap = example_dataset.create_bootstrap(bootstrap_prefix, seed=seed, redo=True)

            bootstrap_backup_prefix = tmpdir / "backup"
            backup_ind = pathlib.Path(f"{bootstrap_backup_prefix}.ind")
            backup_geno = pathlib.Path(f"{bootstrap_backup_prefix}.geno")
            backup_snp = pathlib.Path(f"{bootstrap_backup_prefix}.snp")

            # move bootstrap's geno, ind, snp files such that we can delete them, but compare them later
            shutil.move(bootstrap.ind_file, backup_ind)
            shutil.move(bootstrap.geno_file, backup_geno)
            shutil.move(bootstrap.snp_file, backup_snp)

            # make sure moving actually worked such that a subsequent bootstrap run will use the checkpoint
            assert not bootstrap.files_exist()

            # make sure the checkpoint file exits
            ckp_file = pathlib.Path(f"{bootstrap_prefix}.ckp")
            assert ckp_file.exists()

            # rerun the bootstrap, this time use a different seed to reduce the chance of creating identical
            # files randomly
            bootstrap = example_dataset.create_bootstrap(bootstrap_prefix, seed=seed + 1, redo=False)

            # compare the file contents to make sure the checkpoint was actually used
            assert backup_ind.open().read() == bootstrap.ind_file.open().read()
            assert backup_geno.open().read() == bootstrap.geno_file.open().read()
            assert backup_snp.open().read() == bootstrap.snp_file.open().read()

    def test_deduplicate_snp_id(self):
        n_ids = 5
        seen_ids = set()
        for i, snp_id in enumerate(n_ids * ["snp_id"]):
            deduplicate = deduplicate_snp_id(snp_id, seen_ids)
            seen_ids.add(deduplicate)

            if i > 0:
                assert f"r{i}" in deduplicate

        assert len(seen_ids) == n_ids


class TestDatasetSmartPCA:
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
            example_dataset.smartpca(smartpca, n_pcs, result_dir)
            assert isinstance(example_dataset.pca, PCA)

    def test_smartpca_with_additional_settings(self, smartpca, example_dataset):
        n_pcs = 2
        smartpca_settings = {
            "numoutlieriter": 0,
            "shrinkmode": True
        }
        with tempfile.TemporaryDirectory() as result_dir:
            assert example_dataset.pca is None
            result_dir = pathlib.Path(result_dir)
            example_dataset.smartpca(smartpca, n_pcs, result_dir, smartpca_optional_settings=smartpca_settings)
            assert isinstance(example_dataset.pca, PCA)

    def test_smartpca_from_existing(self, smartpca, example_dataset):
        n_pcs = 2
        with tempfile.TemporaryDirectory() as result_dir:
            assert example_dataset.pca is None
            result_dir = pathlib.Path(result_dir)
            example_dataset.smartpca(smartpca, n_pcs, result_dir)
            assert isinstance(example_dataset.pca, PCA)

            # rerun with redo=False to make sure loading from previous finished runs works
            example_dataset.smartpca(smartpca, n_pcs, result_dir, redo=False)

    def test_smartpca_with_poplist(self, smartpca, example_dataset_with_poplist):
        n_pcs = 2
        with tempfile.TemporaryDirectory() as result_dir:
            assert example_dataset_with_poplist.pca is None
            result_dir = pathlib.Path(result_dir)
            example_dataset_with_poplist.smartpca(smartpca, n_pcs, result_dir)
            assert isinstance(example_dataset_with_poplist.pca, PCA)

    def test_smartpca_ignores_nonsense_setting(self, smartpca, example_dataset):
        n_pcs = 2
        smartpca_settings = {
            "nonsenseSetting": "fail",
        }
        with tempfile.TemporaryDirectory() as result_dir:
            assert example_dataset.pca is None
            result_dir = pathlib.Path(result_dir)
            example_dataset.smartpca(smartpca, n_pcs, result_dir, smartpca_optional_settings=smartpca_settings)
            assert isinstance(example_dataset.pca, PCA)