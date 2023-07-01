import tempfile

from pandora.pandora import *


def test_pandora_config_from_configfile(pandora_test_config, pandora_test_config_yaml):
    pandora_config = pandora_config_from_configfile(pandora_test_config)

    # manually check some settings to make sure the yaml is correctly parsed into the PandoraConfig
    assert str(pandora_config.dataset_prefix) == str(pandora_test_config_yaml.get("dataset_prefix"))
    assert pandora_config.n_bootstraps == pandora_test_config_yaml.get("n_bootstraps")
    assert pandora_config.keep_bootstraps == pandora_test_config_yaml.get("keep_bootstraps")
    assert pandora_config.threads == pandora_test_config_yaml.get("threads")


class TestPandoraConfig:
    def test_get_configuration(self, pandora_test_config, pandora_test_config_yaml):
        pandora_config = pandora_config_from_configfile(pandora_test_config)

        # make sure all settings in pandora_test_config_yaml are identical when exporting
        pandora_config_export = pandora_config.get_configuration()

        for key, expected in pandora_test_config_yaml.items():
            if isinstance(expected, str) and "/" in expected:
                # File path, manually convert in order to compare
                expected = str(pathlib.Path(expected).absolute())
            actual = pandora_config_export.get(key)
            assert actual == expected, (key, (expected, actual))

    def test_save_config(self, pandora_test_config):
        # make sure the export is valid yaml
        pandora_config = pandora_config_from_configfile(pandora_test_config)
        # set the result_dir to a tempdir
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            pandora_config.result_dir = tmpdir
            pandora_config.save_config()

            assert pandora_config.configfile.exists()

            # loading the data again should lead to identical settings
            pandora_config_reload = pandora_config_from_configfile(pandora_config.configfile)

            for key, expected in pandora_config.get_configuration().items():
                actual = pandora_config_reload.get_configuration().get(key)
                if key == "result_dir":
                    # skip check since we manually changed that
                    continue
                assert expected == actual
