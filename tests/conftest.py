import pathlib
import shutil

import pytest
import yaml

from pandora.dataset import EigenDataset, NumpyDataset
from pandora.embedding import *
from pandora.pandora import PandoraConfig, pandora_config_from_configfile

from .test_config import CONVERTF, SMARTPCA


@pytest.fixture
def smartpca() -> Executable:
    return SMARTPCA


@pytest.fixture
def convertf() -> Executable:
    return CONVERTF


@pytest.fixture
def example_eigen_dataset_prefix() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data" / "example"


@pytest.fixture
def example_eigen_sliding_window_dataset_prefix() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data" / "example_sliding_window"


@pytest.fixture
def example_ped_dataset_prefix() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data" / "converted" / "example"


@pytest.fixture
def example_packed_eigen_dataset_prefix() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data" / "converted" / "example.packed"


@pytest.fixture
def example_population_list() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data" / "example_populations.txt"


@pytest.fixture
def example_dataset(example_eigen_dataset_prefix) -> EigenDataset:
    return EigenDataset(example_eigen_dataset_prefix)


@pytest.fixture
def example_sliding_window_dataset(
    example_eigen_sliding_window_dataset_prefix,
) -> EigenDataset:
    return EigenDataset(example_eigen_sliding_window_dataset_prefix)


@pytest.fixture
def example_dataset_with_poplist(
    example_eigen_dataset_prefix, example_population_list
) -> EigenDataset:
    return EigenDataset(example_eigen_dataset_prefix, example_population_list)


@pytest.fixture
def correct_smartpca_result_prefix() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data" / "smartpca" / "example"


@pytest.fixture
def unfinished_smartpca_result_prefix() -> pathlib.Path:
    """
    The log file is incomplete as an indicator of an interrupted smartPCA run
    """
    return pathlib.Path(__file__).parent / "data" / "smartpca" / "unfinished"


@pytest.fixture
def incorrect_smartpca_npcs_result_prefix() -> pathlib.Path:
    """
    Number of PCs in n_pcs_mismatch.evec is 3
    """
    return pathlib.Path(__file__).parent / "data" / "smartpca" / "example_3pcs"


@pytest.fixture
def missing_smartpca_result_prefix() -> pathlib.Path:
    """
    SmartPCA result files do not exist
    """
    return pathlib.Path(__file__).parent / "data" / "smartpca" / "does_not_exist"


@pytest.fixture
def pca_example(correct_smartpca_result_prefix) -> PCA:
    return from_smartpca(
        evec=pathlib.Path(f"{correct_smartpca_result_prefix}.evec"),
        eval=pathlib.Path(f"{correct_smartpca_result_prefix}.eval"),
    )


@pytest.fixture
def pandora_test_config_file() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data" / "test_config.yaml"


@pytest.fixture
def pandora_test_config_file_sliding_window() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data" / "test_config_sliding_window.yaml"


@pytest.fixture
def pandora_test_config(pandora_test_config_file, smartpca, convertf) -> PandoraConfig:
    # load the config file and manually set smartpca and convertf options based on the test_config
    pandora_config = pandora_config_from_configfile(pandora_test_config_file)
    pandora_config.smartpca = smartpca
    pandora_config.convertf = convertf
    return pandora_config


@pytest.fixture
def pandora_test_config_mds(pandora_test_config):
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
def pandora_test_config_sliding_window_mds(
    pandora_test_config_sliding_window,
) -> PandoraConfig:
    pandora_test_config_sliding_window.embedding_algorithm = EmbeddingAlgorithm.MDS
    return pandora_test_config_sliding_window


@pytest.fixture
def pandora_test_config_yaml(pandora_test_config_file) -> Dict:
    return yaml.safe_load(pandora_test_config_file.open())


@pytest.fixture
def pandora_test_config_with_embedding_populations_file() -> pathlib.Path:
    return (
        pathlib.Path(__file__).parent
        / "data"
        / "test_config_with_embedding_populations.yaml"
    )


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
def test_numpy_dataset():
    test_data = np.asarray(
        [[0, 1, 1, 1, 1, 1, 1], [2, 2, 0, 2, 2, 2, 2], [3, 3, 3, 0, 3, 3, 3]]
    )
    sample_ids = pd.Series(["sample1", "sample2", "sample3"])
    populations = pd.Series(["population1", "population2", "population3"])
    dataset = NumpyDataset(test_data, sample_ids, populations, missing_value=0)
    return dataset


@pytest.fixture
def test_numpy_dataset_sliding_window():
    test_data = np.asarray(
        [
            [1, 2, 1, 2, 0, 1, 0, 1, 2, 0, 2, 0, 0, 1, 0, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 2, 2, 1, 0, 2, 1, 2, 2, 0, 2, 0, 2, 1, 1, 0],
            [0, 0, 0, 2, 0, 2, 1, 1, 1, 0, 1, 2, 2, 0, 2, 2, 1, 0, 0, 2],
        ]
    )
    sample_ids = pd.Series(["sample1", "sample2", "sample3"])
    populations = pd.Series(["population1", "population2", "population3"])
    dataset = NumpyDataset(test_data, sample_ids, populations)
    return dataset


@pytest.fixture(autouse=True)
def cleanup_pandora_test_results():
    yield
    results_dir = pathlib.Path("tests") / "data" / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
