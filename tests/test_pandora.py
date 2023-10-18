import pathlib
import tempfile
from typing import Dict

import pandas as pd
import pytest
import yaml

from pandora.custom_errors import PandoraException
from pandora.custom_types import EmbeddingAlgorithm
from pandora.dataset import EigenDataset
from pandora.embedding import MDS, PCA
from pandora.pandora import (
    Pandora,
    PandoraConfig,
    convert_to_eigenstrat_format,
    pandora_config_from_configfile,
)


@pytest.fixture
def pandora_test_config_file() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data" / "test_config.yaml"


@pytest.fixture
def pandora_test_config_file_sliding_window() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data" / "test_config_sliding_window.yaml"


@pytest.fixture
def pandora_test_config_with_embedding_populations_file() -> pathlib.Path:
    return (
        pathlib.Path(__file__).parent
        / "data"
        / "test_config_with_embedding_populations.yaml"
    )


@pytest.fixture
def pandora_test_config(pandora_test_config_file, smartpca, convertf) -> PandoraConfig:
    # load the config file and manually set smartpca and convertf options based on the test_config
    pandora_config = pandora_config_from_configfile(pandora_test_config_file)
    pandora_config.smartpca = smartpca
    pandora_config.convertf = convertf
    return pandora_config


@pytest.fixture
def pandora_test_config_mds(pandora_test_config) -> PandoraConfig:
    pandora_test_config.embedding_algorithm = EmbeddingAlgorithm.MDS
    return pandora_test_config


@pytest.fixture
def pandora_test_config_sliding_window(
    pandora_test_config_file_sliding_window, smartpca, convertf
) -> PandoraConfig:
    pandora_config = pandora_config_from_configfile(
        pandora_test_config_file_sliding_window
    )
    pandora_config.smartpca = smartpca
    pandora_config.convertf = convertf
    return pandora_config


@pytest.fixture
def pandora_test_config_with_ped_files(smartpca, convertf) -> PandoraConfig:
    pandora_config = pandora_config_from_configfile(
        pathlib.Path(__file__).parent / "data" / "test_config_ped_format.yaml"
    )
    pandora_config.smartpca = smartpca
    pandora_config.convertf = convertf
    return pandora_config


@pytest.fixture
def pandora_test_config_with_embedding_populations(
    pandora_test_config_with_embedding_populations_file, smartpca, convertf
) -> PandoraConfig:
    # load the config file and manually set smartpca and convertf options based on the test_config
    pandora_config = pandora_config_from_configfile(
        pandora_test_config_with_embedding_populations_file
    )
    pandora_config.smartpca = smartpca
    pandora_config.convertf = convertf
    return pandora_config


@pytest.fixture
def pandora_test_config_sliding_window_mds(
    pandora_test_config_sliding_window,
) -> PandoraConfig:
    pandora_test_config_sliding_window.embedding_algorithm = EmbeddingAlgorithm.MDS
    return pandora_test_config_sliding_window


@pytest.fixture
def pandora_test_config_yaml(pandora_test_config_file) -> Dict:
    return yaml.safe_load(pandora_test_config_file.open())


def test_pandora_config_from_configfile(
    pandora_test_config_file, pandora_test_config_yaml
):
    pandora_config = pandora_config_from_configfile(pandora_test_config_file)

    # manually check some settings to make sure the yaml is correctly parsed into the PandoraConfig
    assert str(pandora_config.dataset_prefix) == str(
        pandora_test_config_yaml.get("dataset_prefix")
    )
    assert pandora_config.n_replicates == pandora_test_config_yaml.get("n_replicates")
    assert pandora_config.threads == pandora_test_config_yaml.get("threads")


def test_convert_to_eigenstrat(
    example_ped_dataset_prefix, pandora_test_config_with_ped_files
):
    # make sure the dataset prefix prior to convertion is the unconverted prefix
    assert (
        pandora_test_config_with_ped_files.dataset_prefix.absolute()
        == example_ped_dataset_prefix.absolute()
    )

    # the dataset prefix should be the following
    expected_prefix = (
        pandora_test_config_with_ped_files.convertf_result_dir
        / pandora_test_config_with_ped_files.dataset_prefix.name
    )

    # now convert
    actual_prefix = convert_to_eigenstrat_format(
        convertf=pandora_test_config_with_ped_files.convertf,
        convertf_result_dir=pandora_test_config_with_ped_files.convertf_result_dir,
        dataset_prefix=pandora_test_config_with_ped_files.dataset_prefix,
        file_format=pandora_test_config_with_ped_files.file_format,
    )

    assert actual_prefix == expected_prefix


class TestPandoraConfig:
    def test_get_configuration(
        self, pandora_test_config_file, pandora_test_config_yaml
    ):
        pandora_config = pandora_config_from_configfile(pandora_test_config_file)

        # make sure all settings in pandora_test_config_yaml are identical when exporting
        pandora_config_export = pandora_config.get_configuration()

        for key, expected in pandora_test_config_yaml.items():
            if isinstance(expected, str) and "/" in expected:
                # file path, manually convert in order to compare
                expected = str(pathlib.Path(expected).absolute())
            actual = pandora_config_export.get(key)

            if isinstance(expected, dict):
                for k, v in expected.items():
                    assert str(actual.get(k)) == str(v), (
                        key,
                        k,
                        (str(actual.get(k)), str(v)),
                    )
                continue

            assert actual == expected, (key, (actual, expected))

    def test_save_config(self, pandora_test_config_file):
        # make sure the export is valid yaml
        pandora_config = pandora_config_from_configfile(pandora_test_config_file)
        # set the result_dir to a tempdir
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            pandora_config.result_dir = tmpdir
            pandora_config.save_config()

            assert pandora_config.configfile.exists()

            # loading the data again should lead to identical settings
            pandora_config_reload = pandora_config_from_configfile(
                pandora_config.configfile
            )

            for key, expected in pandora_config.get_configuration().items():
                actual = pandora_config_reload.get_configuration().get(key)
                if key == "result_dir":
                    # skip check since we manually changed that
                    continue
                assert expected == actual


class TestPandora:
    def test_init(self, pandora_test_config):
        pandora = Pandora(pandora_test_config)

        # check that everything was initialized correctly
        # pandora_test_config does not specify embedding_populations
        assert pandora.dataset.embedding_populations.empty
        # file prefix should be set correctly and files should exist
        assert pandora.dataset.file_prefix == pandora_test_config.dataset_prefix
        assert pandora.dataset.files_exist()

        # replicates and similarities should be empty
        assert len(pandora.replicates) == 0
        assert len(pandora.pairwise_stabilities) == 0
        assert len(pandora.pairwise_cluster_stabilities) == 0
        assert pandora.sample_support_values.empty

    def test_init_with_embedding_populations(
        self, pandora_test_config_with_embedding_populations, example_population_list
    ):
        pandora = Pandora(pandora_test_config_with_embedding_populations)

        # check that the embedding_populations were initialized correctly
        pca_populations_expected = {
            line.strip() for line in example_population_list.open()
        }
        assert set(pandora.dataset.embedding_populations) == pca_populations_expected

    def test_do_pca(self, pandora_test_config):
        pandora = Pandora(pandora_test_config)

        assert pandora.dataset.pca is None

        pandora.embed_dataset()

        # pandora's dataset's PCA should be a PCA object now and not None
        assert isinstance(pandora.dataset.pca, PCA)
        # plot directory should contain two plots
        assert len(list(pandora.pandora_config.plot_dir.iterdir())) == 2

    def test_do_mds(self, pandora_test_config_mds):
        pandora = Pandora(pandora_test_config_mds)

        assert pandora.dataset.mds is None

        pandora.embed_dataset()

        # pandora's dataset's MDS should be a MDS object now and not None
        assert isinstance(pandora.dataset.mds, MDS)
        # plot directory should contain two plots
        assert len(list(pandora.pandora_config.plot_dir.iterdir())) == 2

    def test_do_pca_with_pca_populations(
        self, pandora_test_config_with_embedding_populations
    ):
        pandora = Pandora(pandora_test_config_with_embedding_populations)

        assert pandora.dataset.pca is None
        assert len(pandora.dataset.embedding_populations) > 0

        pandora.embed_dataset()

        # pandora's dataset's PCA should be a PCA object now and not None
        assert isinstance(pandora.dataset.pca, PCA)

        # plot directory should contain three plots
        assert len(list(pandora.pandora_config.plot_dir.iterdir())) == 3

    def test_plot_dataset_fails_if_pca_is_missing(self, pandora_test_config):
        pandora = Pandora(pandora_test_config)
        with pytest.raises(PandoraException, match="Embedding not yet run for dataset"):
            pandora._plot_dataset(pandora.dataset, pandora.dataset.name)

    def test_plot_dataset_fails_if_mds_is_missing(self, pandora_test_config_mds):
        pandora = Pandora(pandora_test_config_mds)
        with pytest.raises(PandoraException, match="Embedding not yet run for dataset"):
            pandora._plot_dataset(pandora.dataset, pandora.dataset.name)

    def test_bootstrap_embeddings_with_pca(self, pandora_test_config):
        pandora = Pandora(pandora_test_config)
        pandora.pandora_config.keep_replicates = True
        n_bootstraps_expected = pandora.pandora_config.n_replicates

        assert len(pandora.replicates) == 0

        # for plotting we need the empirical embedding data
        pandora.embed_dataset()
        pandora.bootstrap_embeddings()

        assert len(pandora.replicates) == n_bootstraps_expected
        assert all(isinstance(bs, EigenDataset) for bs in pandora.replicates)
        assert all(b.pca is not None for b in pandora.replicates)

    def test_bootstrap_embedding_with_mds(self, pandora_test_config_mds):
        pandora = Pandora(pandora_test_config_mds)
        pandora.pandora_config.keep_replicates = True
        n_bootstraps_expected = pandora.pandora_config.n_replicates

        assert len(pandora.replicates) == 0

        # for plotting we need the empirical embedding data
        pandora.embed_dataset()
        pandora.bootstrap_embeddings()

        # since we are asking for MDS analyses, pandora.dataset.mds should be set, but pandora.dataset.pca shouldn't
        assert pandora.dataset.mds is not None
        assert isinstance(pandora.dataset.mds, MDS)
        assert pandora.dataset.pca is None

        assert len(pandora.replicates) == n_bootstraps_expected
        assert all(isinstance(bs, EigenDataset) for bs in pandora.replicates)
        assert all(b.mds is not None for b in pandora.replicates)

    def test_sliding_window_with_pca(self, pandora_test_config_sliding_window):
        pandora = Pandora(pandora_test_config_sliding_window)
        pandora.pandora_config.keep_replicates = True
        n_windows_expected = pandora.pandora_config.n_replicates

        assert len(pandora.replicates) == 0

        # for plotting we need the empirical embedding data
        pandora.embed_dataset()
        pandora.sliding_window()

        assert len(pandora.replicates) == n_windows_expected
        assert all(isinstance(bs, EigenDataset) for bs in pandora.replicates)
        assert all(b.pca is not None for b in pandora.replicates)

    def test_sliding_window_with_mds(self, pandora_test_config_sliding_window_mds):
        pandora = Pandora(pandora_test_config_sliding_window_mds)
        pandora.pandora_config.keep_replicates = True
        n_windows_expected = pandora.pandora_config.n_replicates

        assert len(pandora.replicates) == 0

        # for plotting we need the empirical embedding data
        pandora.embed_dataset()
        pandora.sliding_window()

        # since we are asking for MDS analyses, pandora.dataset.mds should be set, but pandora.dataset.pca shouldn't
        assert pandora.dataset.mds is not None
        assert isinstance(pandora.dataset.mds, MDS)
        assert pandora.dataset.pca is None

        assert len(pandora.replicates) == n_windows_expected
        assert all(isinstance(bs, EigenDataset) for bs in pandora.replicates)
        assert all(b.mds is not None for b in pandora.replicates)

    def test_log_and_save_results_fails_if_no_embedding_run_yet(
        self, pandora_test_config
    ):
        pandora = Pandora(pandora_test_config)
        with pytest.raises(PandoraException, match="No results to log!"):
            pandora.log_and_save_replicates_results()

    def test_log_and_save_results(self, pandora_test_config_with_embedding_populations):
        # modify the config to not plot anything for these tests
        pandora_test_config_with_embedding_populations.plot_results = False
        # also we disable the bootstrap convergence check
        pandora_test_config_with_embedding_populations.bootstrap_convergence_check = (
            False
        )
        pandora = Pandora(pandora_test_config_with_embedding_populations)

        # first lets run the bootstrap analyses to make sure there is something to log
        pandora.bootstrap_embeddings()

        # next log the results and check if the expected files exist
        pandora.log_and_save_replicates_results()

        expected_files = [
            pandora.pandora_config.pairwise_stability_result_file,  # csv with pairwise Pandora stabilities
            pandora.pandora_config.sample_support_values_csv,  # csv with pandora support values
            pandora.pandora_config.projected_sample_support_values_csv,  # support values for projected samples only
        ]

        assert all(f.exists() for f in expected_files)

        # also check if the file contents are what we expect in terms of number of rows and columns
        n_bootstraps = pandora.pandora_config.n_replicates
        n_bootstrap_combinations = n_bootstraps * (n_bootstraps - 1) / 2

        n_samples = pandora.dataset.sample_ids.shape[0]
        n_projected_samples = pandora.dataset.projected_samples.shape[0]

        # for this pandora config the number of projected samples should be smaller than n_samples -> sanity check
        assert n_projected_samples < n_samples

        pairwise_bootstrap_results = pd.read_csv(
            pandora.pandora_config.pairwise_stability_result_file, index_col=0
        )
        # we expect one row for each pairwise combination and two columns (PS and PCS)
        assert pairwise_bootstrap_results.shape == (n_bootstrap_combinations, 2)

        sample_support_values = pd.read_csv(
            pandora.pandora_config.sample_support_values_csv, index_col=0
        )
        # we expect one row per sample and 1 column (PSV)
        assert sample_support_values.shape == (n_samples, 1)

        projected_sample_support_values = pd.read_csv(
            pandora.pandora_config.projected_sample_support_values_csv, index_col=0
        )
        print(projected_sample_support_values)
        # we expect one row per projected sample and one column
        assert projected_sample_support_values.shape == (
            n_projected_samples,
            1,
        )
