import pathlib
from enum import Enum
from typing import Union

Executable = Union[str, pathlib.Path]


class FileFormat(Enum):
    """Supported population genetics file formats."""

    ANCESTRYMAP = "ANCESTRYMAP"
    EIGENSTRAT = "EIGENSTRAT"
    PED = "PED"
    PACKEDPED = "PACKEDPED"
    PACKEDANCESTRYMAP = "PACKEDANCESTRYMAP"


class EmbeddingAlgorithm(Enum):
    """Supported options for dimensionality reduction."""

    PCA = "PCA"
    MDS = "MDS"


class AnalysisMode(Enum):
    """Supported options for analyses to conduct."""

    BOOTSTRAP = "BOOTSTRAP"
    SLIDING_WINDOW = "SLIDING_WINDOW"
