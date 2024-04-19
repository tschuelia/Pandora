import pathlib
import pickle
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from pandora.custom_errors import PandoraException
from pandora.dataset import (
    EigenDataset,
    NumpyDataset,
    _deduplicate_snp_id,
    _get_embedding_populations,
    _process_input_data,
    _smartpca_finished,
    numpy_dataset_from_eigenfiles,
)
from pandora.distance_metrics import (
    DISTANCE_METRICS,
    fst_population_distance,
    missing_corrected_hamming_sample_distance,
)
from pandora.embedding import Embedding
from pandora.embedding_comparison import EmbeddingComparison

DTYPES_AND_MISSING_VALUES = [
    # signed integers
    (np.int8, -1),
    (np.int16, -1),
    (np.int32, -1),
    (np.int64, -1),
    # unsigned integers
    (np.uint8, 255),
    (np.uint16, 65535),
    (np.uint32, 4294967295),
    # floating point values
    (np.float16, np.nan),
    (np.float32, np.nan),
    (np.float64, np.nan),
]


@pytest.fixture
def example_packed_eigen_dataset_prefix() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data" / "converted" / "example.packed"


@pytest.fixture
def example_dataset_with_poplist(
    example_eigen_dataset_prefix, example_population_list
) -> EigenDataset:
    return EigenDataset(example_eigen_dataset_prefix, example_population_list)


@pytest.fixture
def unfinished_smartpca_result_prefix() -> pathlib.Path:
    """The log file is incomplete as an indicator of an interrupted smartPCA run."""
    return pathlib.Path(__file__).parent / "data" / "smartpca" / "unfinished"


@pytest.fixture
def incorrect_smartpca_npcs_result_prefix() -> pathlib.Path:
    """Number of PCs in n_pcs_mismatch.evec is 3."""
    return pathlib.Path(__file__).parent / "data" / "smartpca" / "example_3pcs"


@pytest.fixture
def missing_smartpca_result_prefix() -> pathlib.Path:
    """SmartPCA result files do not exist."""
    return pathlib.Path(__file__).parent / "data" / "smartpca" / "does_not_exist"


@pytest.fixture
def example_mds_data_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data" / "mds"


class TestEigenDataset:
    def test_get_embedding_populations(self, example_population_list):
        populations = _get_embedding_populations(example_population_list)
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
            np.testing.assert_equal(overlap_window0, overlap_window1)

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
        """Tests whether the bootstrapped .geno, .ind, and .snp files have the correct file file_format."""
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
        """Test that when setting redo=False, the resulting bootstrap dataset files are identical to the example_dataset
        files and no bootstrapping is performed."""
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
        """Test that when setting redo=False, and the results files do not exist, but a checkpoint does that the
        bootstrap based on the checkpoint is identical."""
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


class TestEigenDatasetPCA:
    def test_smartpca_finished(self, correct_smartpca_result_prefix):
        n_pcs = 2
        assert _smartpca_finished(n_pcs, correct_smartpca_result_prefix)

    def test_smartpca_not_finished(self, unfinished_smartpca_result_prefix):
        n_pcs = 2
        assert not _smartpca_finished(n_pcs, unfinished_smartpca_result_prefix)

    def test_smartpca_wrong_n_pcs(self, incorrect_smartpca_npcs_result_prefix):
        n_pcs = 2
        assert not _smartpca_finished(n_pcs, incorrect_smartpca_npcs_result_prefix)

    def test_smartpca_results_missing(self, missing_smartpca_result_prefix):
        n_pcs = 2
        assert not _smartpca_finished(n_pcs, missing_smartpca_result_prefix)

    def test_smartpca(self, smartpca, example_dataset):
        n_pcs = 2
        with tempfile.TemporaryDirectory() as result_dir:
            assert example_dataset.pca is None
            result_dir = pathlib.Path(result_dir)
            example_dataset.run_pca(smartpca, n_pcs, result_dir)
            assert isinstance(example_dataset.pca, Embedding)

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
            assert isinstance(example_dataset.pca, Embedding)

    def test_smartpca_from_existing(self, smartpca, example_dataset):
        n_pcs = 2
        with tempfile.TemporaryDirectory() as result_dir:
            assert example_dataset.pca is None
            result_dir = pathlib.Path(result_dir)
            example_dataset.run_pca(smartpca, n_pcs, result_dir)
            assert isinstance(example_dataset.pca, Embedding)

            # rerun with redo=False to make sure loading from previous finished runs works
            example_dataset.run_pca(smartpca, n_pcs, result_dir, redo=False)

    def test_smartpca_with_poplist(self, smartpca, example_dataset_with_poplist):
        n_pcs = 2
        with tempfile.TemporaryDirectory() as result_dir:
            assert example_dataset_with_poplist.pca is None
            result_dir = pathlib.Path(result_dir)
            example_dataset_with_poplist.run_pca(smartpca, n_pcs, result_dir)
            assert isinstance(example_dataset_with_poplist.pca, Embedding)

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
            assert isinstance(example_dataset.pca, Embedding)

    def test_smartpca_fails_for_too_many_components(self, smartpca, example_dataset):
        n_pcs = example_dataset.get_sequence_length() + 10

        with pytest.raises(
            PandoraException, match="Number of Principal Components needs to be smaller"
        ):
            example_dataset.run_pca(smartpca, n_pcs)


def _modify_population(
    original_prefix: pathlib.Path, new_prefix: pathlib.Path, new_population: str
):
    old_geno = pathlib.Path(f"{original_prefix}.geno")
    old_snp = pathlib.Path(f"{original_prefix}.snp")
    old_ind = pathlib.Path(f"{original_prefix}.ind")

    new_geno = pathlib.Path(f"{new_prefix}.geno")
    new_snp = pathlib.Path(f"{new_prefix}.snp")
    new_ind = pathlib.Path(f"{new_prefix}.ind")

    # we only need to modify the .indfile and can copy the .geno and .snp files
    shutil.copy(old_geno, new_geno)
    shutil.copy(old_snp, new_snp)

    with new_ind.open("w") as new_ind_handle:
        for i, l in enumerate(old_ind.open()):
            # only for the first line, we change the population to the passsed new_population
            if i > 0:
                new_ind_handle.write(l)
                continue

            sample_id, sex, population = l.split()
            new_ind_handle.write(
                f"{sample_id.strip()}\t{sex.strip()}\t{new_population}\n"
            )


class TestEigenDatasetMDS:
    def test_run_mds_with_dash_in_population_name(
        self, smartpca, example_eigen_dataset_prefix
    ):
        # a bug in a previous Pandora version caused MDS to fail if there is a dash in the populatjon name
        # to make sure we fixed it, we explicitly test for this here
        with tempfile.TemporaryDirectory() as tmpdir:
            # first, we need to modify the example dataset a bit to include a dash in one of the population names
            new_prefix = pathlib.Path(tmpdir) / "dataset"
            _modify_population(
                example_eigen_dataset_prefix, new_prefix, "popname-with-dash"
            )
            dataset = EigenDataset(new_prefix)
            dataset.run_mds(smartpca)
            assert dataset.mds is not None

    def test_run_mds_with_population_ignore(
        self, smartpca, example_eigen_dataset_prefix
    ):
        # smartpca ignores all samples with "IGNORE" as population
        # in a previous pandora version, we were not aware of this so the MDS computation failed
        # We now remove all such samples for the MDS computation and with this test we want to make sure that
        # the dimensions of the resulting MDS are correct
        with tempfile.TemporaryDirectory() as tmpdir:
            # first, we need to modify the example dataset to change one population to "Ignore"
            new_prefix = pathlib.Path(tmpdir) / "dataset"
            _modify_population(example_eigen_dataset_prefix, new_prefix, "Ignore")
            dataset = EigenDataset(new_prefix)
            dataset.run_mds(smartpca)

            # the number of unique populations in this MDS should be one less than when not modifying the population name
            number_of_populations_ignore_dataset = (
                dataset.mds.populations.unique().shape[0]
            )
            # to compare, we need to perform MDS for the unmodified dataset
            unmodified_dataset = EigenDataset(example_eigen_dataset_prefix)
            unmodified_dataset.run_mds(smartpca)
            number_of_populations_orig_dataset = (
                unmodified_dataset.mds.populations.unique().shape[0]
            )
            assert (
                number_of_populations_ignore_dataset
                == number_of_populations_orig_dataset - 1
            )


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

    @pytest.mark.parametrize("imputation", ["mean", "remove"])
    def test_run_pca(self, test_numpy_dataset, imputation):
        n_pcs = 2
        test_numpy_dataset.run_pca(n_pcs, imputation=imputation)
        # dataset should now have a PCA object attached with embedding data (n_samples, n_pcs)
        # dataset.mds should still be None
        assert test_numpy_dataset.pca is not None
        assert test_numpy_dataset.mds is None
        assert isinstance(test_numpy_dataset.pca, Embedding)
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
        np.testing.assert_equal(overlap_window0, overlap_window1)

    @pytest.mark.parametrize("distance_metric", DISTANCE_METRICS)
    @pytest.mark.parametrize("imputation", ["mean", "remove"])
    def test_run_mds(
        self,
        test_numpy_dataset,
        distance_metric,
        imputation,
    ):
        if distance_metric == fst_population_distance:
            # imputation mean or remove not supported for fst_population_distance
            return

        n_components = 2

        test_numpy_dataset.run_mds(
            n_components=n_components,
            distance_metric=distance_metric,
            imputation=imputation,
        )
        # test_numpy_dataset should now have an MDS object attached with embedding data (n_samples, n_components)
        # test_numpy_dataset.pca should still be None
        assert test_numpy_dataset.mds is not None
        assert test_numpy_dataset.pca is None
        assert isinstance(test_numpy_dataset.mds, Embedding)
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

    def test_run_mds_fst_distance_with_missing_data(self):
        # the following dataset contains missing data
        # since the default dtype is uint8, missing values should be represented by the value 255
        # however, prior to the distance matrix computation, 255 should be replaced by np.nan
        # if this does not work properly, the matrix compuation will fail
        test_data = np.asarray(
            [[0, 1, 1, 1, 1, 1, 1], [2, 2, 0, 2, 2, 2, 2], [1, 2, 1, 0, 2, 1, 1]]
        )
        sample_ids = pd.Series(["sample1", "sample2", "sample3"])
        populations = pd.Series(["population1", "population2", "population3"])
        dataset = NumpyDataset(
            test_data, sample_ids, populations, missing_value=0, dtype=np.uint8
        )
        dataset.run_mds(
            n_components=2, distance_metric=fst_population_distance, imputation=None
        )

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

    @pytest.mark.parametrize("dataset_id", range(4))
    def test_mds_simulated_data(self, dataset_id, example_mds_data_dir):
        """
        Since we experienced problems with MDS analyses in Pandora < 2.0.0, we decided to implement a sanity check
        of the current state of MDS results when preparing version 2.0.0.
        For this, we performed MDS analyses using the R cmdscale function as a reference for a set of 4 datasets.
        The 4 datasets are simulated datasets with a varying fraction of missing data obtained from this publication:

        Yi, X. & Latch, E. K. (2022). Nonrandom missing data can bias Principal Component Analysis inference of
        population genetic structure. Molecular Ecology Resources, 22, 602–611. https://doi.org/10.1111/1755-0998.13498.

        The data is publicly available on GitHub: https://github.com/xuelingyi/missing_data_PCA

        For the reference MDS results, we used the missing_corrected_hamming_distance distance metric. To compare the
        MDS results, we use Pandora's EmbeddingComparison and expect the Pandora stability to be close to 1
        indicating that the MDS implementation in Pandora produces the same MDS results as the R cmdscale method.
        """
        dataset: NumpyDataset = pickle.load(
            (example_mds_data_dir / f"example_{dataset_id}_dataset.pckl").open("rb")
        )
        expected_mds = pickle.load(
            (example_mds_data_dir / f"example_{dataset_id}_mds.pckl").open("rb")
        )

        dataset.run_mds(
            n_components=2,
            distance_metric=missing_corrected_hamming_sample_distance,
            imputation=None,
        )

        comparison = EmbeddingComparison(dataset.mds, expected_mds)
        assert comparison.compare() == pytest.approx(1.0)


@pytest.mark.parametrize("dtype, expected_missing_value", DTYPES_AND_MISSING_VALUES)
def test_process_input_data_no_missing_values(dtype, expected_missing_value):
    test_data_no_missing = np.asarray(
        [[0, 1, 1, 1, 1, 1, 1], [2, 2, 0, 2, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0]]
    )
    data, nan_value = _process_input_data(test_data_no_missing, np.nan, dtype)
    # the missing value should not be present in data since there was not missing data initially
    assert not np.any(data == nan_value)
    # nan_value should be the expected one
    if np.isnan(expected_missing_value):
        assert np.isnan(nan_value)
    else:
        assert nan_value == expected_missing_value


@pytest.mark.parametrize("dtype, expected_missing_value", DTYPES_AND_MISSING_VALUES)
def test_process_input_data_with_missing_values(dtype, expected_missing_value):
    test_data_with_missing = np.asarray(
        [
            [np.nan, 1, 1, 1, 1, 1, 1],
            [np.nan, 2, 0, 2, 2, 2, 2],
            [0, 1, 2, 0, np.nan, 2, 0],
        ]
    )
    data, nan_value = _process_input_data(test_data_with_missing, np.nan, dtype)
    if np.isnan(expected_missing_value):
        assert np.isnan(nan_value)
    else:
        assert nan_value == expected_missing_value
    # the missing value should be present three times now
    if np.isnan(nan_value):
        n_missing = np.sum(np.isnan(data))
    else:
        # missing value is not np.nan
        n_missing = np.sum(data == nan_value)
    assert n_missing == 3


@pytest.mark.parametrize("dtype", [np.bool_, np.str_, np.bytes_])
def test_process_input_data_fails_for_unsupported_dtype(dtype):
    test_data_no_missing = np.asarray(
        [[0, 1, 1, 1, 1, 1, 1], [2, 2, 0, 2, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0]]
    )
    with pytest.raises(PandoraException, match="Unsupported numpy dtype passed"):
        _process_input_data(test_data_no_missing, np.nan, dtype)


def test_process_input_data_fails_for_uint64_dtype_conversion():
    # since float64 cannot represent the computed max value for np.uint64, the _safe_cast should fail
    with pytest.raises(PandoraException, match="Could not safely cast to"):
        test_data_with_missing = np.asarray(
            [
                [np.nan, 1, 1, 1, 1, 1, 1],
                [np.nan, 2, 0, 2, 2, 2, 2],
                [0, 1, 2, 0, np.nan, 2, 0],
            ]
        )
        _process_input_data(
            test_data_with_missing, missing_value=np.nan, dtype=np.uint64
        )
