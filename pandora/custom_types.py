import pathlib
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

Executable = Union[str, pathlib.Path]


class FileFormat(Enum):
    ANCESTRYMAP = "ANCESTRYMAP"
    EIGENSTRAT = "EIGENSTRAT"
    PED = "PED"
    PACKEDPED = "PACKEDPED"
    PACKEDANCESTRYMAP = "PACKEDANCESTRYMAP"


class EmbeddingAlgorithm(Enum):
    PCA = "PCA"
    MDS = "MDS"
