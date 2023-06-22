import tempfile
import pathlib
import pytest
import shutil

from pandora.dataset import smartpca_finished


def _check_geno_file(geno_file: pathlib.Path) -> bool:
    # should only contain int values and
    line_lengths = set()
    for line in geno_file.open():
        line = line.strip()
        line_lengths.add(len(line))
        try:
            [int(c) for c in line]
        except ValueError:
            # one char seems to not be an int
            return False

    # each line should have the same number of values
    assert len(line_lengths) == 1
    return True


def _check_ind_file(ind_file: pathlib.Path) -> bool:
    # each line should contain three values
    seen_inds = set()
    total_inds = 0

    for line in ind_file.open():
        try:
            ind_id, _, _ = line.strip().split()
        except ValueError:
            # too few or too many lines
            return False

        seen_inds.add(ind_id.strip())
        total_inds += 1

    # make sure all individuals have a unique ID
    assert len(seen_inds) == total_inds
    return True


def _check_snp_file(snp_file: pathlib.Path) -> bool:
    seen_snps = set()
    total_snps = 0

    for line in snp_file.open():
        line = line.strip()
        # each line contains 4, 5, or 6 values
        n_values = len(line.split())
        if n_values < 4 or n_values > 6:
            return False

        snp_name, chrom, *_ = line.split()
        seen_snps.add(snp_name.strip())
        total_snps += 1

        # the chromosome needs to be between 1 - 22, 23 (X), 24 (Y), 90 (mtDNA), 91 (XY)
        assert int(chrom) in [*range(1, 23), 23, 24, 90, 91]

    # make sure all SNPs have a unique ID
    assert len(seen_snps) == total_snps

    return True


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
            assert _check_geno_file(bootstrap.geno_file)
            assert _check_ind_file(bootstrap.ind_file)
            assert _check_snp_file(bootstrap.snp_file)

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
            assert _check_geno_file(bootstrap.geno_file)
            assert _check_ind_file(bootstrap.ind_file)
            assert _check_snp_file(bootstrap.snp_file)

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