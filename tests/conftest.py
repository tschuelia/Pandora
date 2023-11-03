import pathlib
import shutil

import numpy as np
import pandas as pd
import pytest

from pandora.custom_types import Executable
from pandora.dataset import EigenDataset, NumpyDataset
from pandora.embedding import PCA, from_smartpca


@pytest.fixture
def smartpca() -> Executable:
    return "smartpca"


@pytest.fixture
def convertf() -> Executable:
    return "convertf"


@pytest.fixture
def example_eigen_dataset_prefix() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data" / "example"


@pytest.fixture
def example_ped_dataset_prefix() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data" / "converted" / "example"


@pytest.fixture
def example_population_list() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data" / "example_populations.txt"


@pytest.fixture
def correct_smartpca_result_prefix() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data" / "smartpca" / "example"


@pytest.fixture
def example_dataset(example_eigen_dataset_prefix) -> EigenDataset:
    return EigenDataset(example_eigen_dataset_prefix)


@pytest.fixture
def pca_example(correct_smartpca_result_prefix) -> PCA:
    return from_smartpca(
        evec=pathlib.Path(f"{correct_smartpca_result_prefix}.evec"),
        eval=pathlib.Path(f"{correct_smartpca_result_prefix}.eval"),
    )


@pytest.fixture
def test_numpy_dataset():
    test_data = np.asarray(
        [[0, 1, 1, 1, 1, 1, 1], [2, 2, 0, 2, 2, 2, 2], [3, 3, 3, 0, 3, 3, 3]]
    )
    sample_ids = pd.Series(["sample1", "sample2", "sample3"])
    populations = pd.Series(["population1", "population2", "population3"])
    dataset = NumpyDataset(test_data, sample_ids, populations, missing_value=0)
    return dataset


@pytest.fixture(autouse=True)
def cleanup_pandora_test_results():
    yield
    results_dir = pathlib.Path("tests") / "data" / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
