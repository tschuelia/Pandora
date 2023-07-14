import pathlib
import tempfile

import pytest

from pandora.pandora import *


def test_pandora_config_from_configfile(
    pandora_test_config_file, pandora_test_config_yaml
):
    pandora_config = pandora_config_from_configfile(pandora_test_config_file)

    # manually check some settings to make sure the yaml is correctly parsed into the PandoraConfig
    assert str(pandora_config.dataset_prefix) == str(
        pandora_test_config_yaml.get("dataset_prefix")
    )
    assert pandora_config.n_bootstraps == pandora_test_config_yaml.get("n_bootstraps")
    assert pandora_config.threads == pandora_test_config_yaml.get("threads")


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
        assert pandora.dataset.embedding_populations == []
        # file prefix should be set correctly and files should exist
        assert pandora.dataset.file_prefix == pandora_test_config.dataset_prefix
        assert pandora.dataset.files_exist()

        # bootstraps and similarities should be empty
        assert len(pandora.bootstraps) == 0
        assert len(pandora.bootstrap_stabilities) == 0
        assert len(pandora.bootstrap_cluster_stabilities) == 0
        assert pandora.sample_support_values.empty

    def test_init_with_pca_populations(
        self, pandora_test_config_with_pca_populations, example_population_list
    ):
        pandora = Pandora(pandora_test_config_with_pca_populations)

        # check that the embedding_populations were initialized correctly
        pca_populations_expected = {l.strip() for l in example_population_list.open()}
        assert set(pandora.dataset.embedding_populations) == pca_populations_expected

    def test_do_pca(self, pandora_test_config):
        pandora = Pandora(pandora_test_config)

        assert pandora.dataset.pca is None

        pandora.do_pca()

        # pandora's dataset's PCA should be a PCA object now and not None
        assert isinstance(pandora.dataset.pca, PCA)
        # plot directory should contain two plots
        assert len(list(pandora.pandora_config.plot_dir.iterdir())) == 2

    def test_do_pca_with_pca_populations(
        self, pandora_test_config_with_pca_populations
    ):
        pandora = Pandora(pandora_test_config_with_pca_populations)

        assert pandora.dataset.pca is None
        assert len(pandora.dataset.embedding_populations) > 0

        pandora.do_pca()

        # pandora's dataset's PCA should be a PCA object now and not None
        assert isinstance(pandora.dataset.pca, PCA)

        # plot directory should contain three plots
        assert len(list(pandora.pandora_config.plot_dir.iterdir())) == 3

    def test_plot_dataset_fails_if_pca_is_missing(self, pandora_test_config):
        pandora = Pandora(pandora_test_config)
        with pytest.raises(PandoraException, match="No PCA run for dataset yet"):
            pandora._plot_dataset()

    def test_bootstrap_pcas(self, pandora_test_config):
        pandora = Pandora(pandora_test_config)
        pandora.pandora_config.keep_bootstraps = True
        n_bootstraps_expected = pandora.pandora_config.n_bootstraps

        assert len(pandora.bootstraps) == 0

        # for plotting we need the empirical PCA data
        pandora.do_pca()
        pandora.bootstrap_pcas()

        assert len(pandora.bootstraps) == n_bootstraps_expected
        assert all(isinstance(bs, Dataset) for bs in pandora.bootstraps)
