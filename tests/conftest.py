import pytest
import pathlib
import os

from pandora.dataset import Dataset


@pytest.fixture
def example_eigen_dataset_prefix():
    return pathlib.Path(__file__).parent / "data" / "example"


@pytest.fixture
def example_dataset(example_eigen_dataset_prefix):
    return Dataset(example_eigen_dataset_prefix)


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
    return pathlib.Path(__file__).parent / "data" / "smartpca" / "n_pcs_mismatch"


@pytest.fixture
def missing_smartpca_result_prefix():
    """
    SmartPCA result files do not exist
    """
    return pathlib.Path(__file__).parent / "data" / "smartpca" / "does_not_exist"