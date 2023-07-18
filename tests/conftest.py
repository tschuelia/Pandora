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
def example_population_list() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data" / "example_populations.txt"


@pytest.fixture
def example_dataset(example_eigen_dataset_prefix) -> EigenDataset:
    return EigenDataset(example_eigen_dataset_prefix)


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
def pca_reference(correct_smartpca_result_prefix) -> PCA:
    return from_smartpca(
        evec=pathlib.Path(f"{correct_smartpca_result_prefix}.evec"),
        eval=pathlib.Path(f"{correct_smartpca_result_prefix}.eval"),
    )


@pytest.fixture
def pca_comparable_fewer_samples(pca_reference) -> PCA:
    # use only four of the samples from pca_reference
    pca_data = pca_reference.embedding.copy()[:4]
    return PCA(pca_data, pca_reference.n_components, pca_reference.explained_variances)


@pytest.fixture
def pca_comparable_with_different_sample_ids(pca_reference) -> PCA:
    pca_data = pca_reference.embedding.copy()
    pca_data.sample_id = pca_data.sample_id + "_foo"
    return PCA(pca_data, pca_reference.n_components, pca_reference.explained_variances)


@pytest.fixture
def pca_reference_and_comparable_with_score_lower_than_one() -> Tuple[PCA, PCA]:
    pca1 = PCA(
        pd.DataFrame(
            data={
                "sample_id": ["sample1", "sample2", "sample3"],
                "population": ["population1", "population2", "population3"],
                "D0": [1, 2, 3],
                "D1": [1, 2, 3],
            }
        ),
        2,
        np.asarray([0.0, 0.0]),
    )

    pca2 = PCA(
        pd.DataFrame(
            data={
                "sample_id": ["sample1", "sample2", "sample3"],
                "population": ["population1", "population2", "population3  "],
                "D0": [1, 1, 2],
                "D1": [1, 2, 1],
            }
        ),
        2,
        np.asarray([0.0, 0.0]),
    )

    return pca1, pca2


@pytest.fixture
def pandora_test_config_file() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data" / "test_config.yaml"


@pytest.fixture
def pandora_test_config(pandora_test_config_file, smartpca, convertf) -> PandoraConfig:
    # load the config file and manually set smartpca and convertf options based on the test_config
    pandora_config = pandora_config_from_configfile(pandora_test_config_file)
    pandora_config.smartpca = smartpca
    pandora_config.convertf = convertf
    return pandora_config


@pytest.fixture
def pandora_test_config_yaml(pandora_test_config_file) -> Dict:
    return yaml.safe_load(pandora_test_config_file.open())


@pytest.fixture
def pandora_test_config_with_pca_populations_file() -> pathlib.Path:
    return (
        pathlib.Path(__file__).parent / "data" / "test_config_with_pca_populations.yaml"
    )


@pytest.fixture
def pandora_test_config_with_pca_populations(
    pandora_test_config_with_pca_populations_file, smartpca, convertf
) -> PandoraConfig:
    # load the config file and manually set smartpca and convertf options based on the test_config
    pandora_config = pandora_config_from_configfile(
        pandora_test_config_with_pca_populations_file
    )
    pandora_config.smartpca = smartpca
    pandora_config.convertf = convertf
    return pandora_config


@pytest.fixture
def test_numpy_dataset():
    test_data = np.asarray(
        [[1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3, 3]]
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
