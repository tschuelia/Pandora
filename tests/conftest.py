import pytest
import pathlib

from pandora.dataset import Dataset
from .test_config import SMARTPCA, CONVERTF


@pytest.fixture
def smartpca():
    return SMARTPCA


@pytest.fixture
def convertf():
    return CONVERTF


@pytest.fixture
def example_eigen_dataset_prefix():
    return pathlib.Path(__file__).parent / "data" / "example"


@pytest.fixture
def example_population_list():
    return pathlib.Path(__file__).parent / "data" / "example_populations.txt"


@pytest.fixture
def example_dataset(example_eigen_dataset_prefix):
    return Dataset(example_eigen_dataset_prefix)


@pytest.fixture
def example_dataset_with_poplist(example_eigen_dataset_prefix, example_population_list):
    return Dataset(example_eigen_dataset_prefix, example_population_list)


@pytest.fixture
def correct_smartpca_result_prefix():
    return pathlib.Path(__file__).parent / "data" / "smartpca" / "example"


@pytest.fixture
def unfinished_smartpca_result_prefix():
    """
    The log file is incomplete as an indicator of an interrupted smartPCA run
    """
    return pathlib.Path(__file__).parent / "data" / "smartpca" / "unfinished"


@pytest.fixture
def incorrect_smartpca_npcs_result_prefix():
    """
    Number of PCs in n_pcs_mismatch.evec is 3
    """
    return pathlib.Path(__file__).parent / "data" / "smartpca" / "example_3pcs"


@pytest.fixture
def missing_smartpca_result_prefix():
    """
    SmartPCA result files do not exist
    """
    return pathlib.Path(__file__).parent / "data" / "smartpca" / "does_not_exist"
