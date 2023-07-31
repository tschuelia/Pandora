import pathlib
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

Executable = Union[str, pathlib.Path]


class FileFormat(Enum):
    """
    Supported population genetics file formats.
    """

    ANCESTRYMAP = "ANCESTRYMAP"
    EIGENSTRAT = "EIGENSTRAT"
    PED = "PED"
    PACKEDPED = "PACKEDPED"
    PACKEDANCESTRYMAP = "PACKEDANCESTRYMAP"


class EmbeddingAlgorithm(Enum):
    """
    Supported options for dimensionality reduction.
    """

    PCA = "PCA"
    MDS = "MDS"


class AnalysisMode(Enum):
    """
    Supported options for analyses to conduct.
    """

    BOOTSTRAP = "Bootstrap"
    SLIDING_WINDOW = "Sliding window"
