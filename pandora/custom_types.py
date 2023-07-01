import numpy as np
import numpy.typing as npt
import pandas as pd
import pathlib
from enum import Enum
from typing import Union, Tuple, Dict, List, Iterable, Optional, Set


Executable = Union[str, pathlib.Path]
_PCA = Union[np.ndarray, pd.DataFrame]


class FileFormat(Enum):
    ANCESTRYMAP = "ANCESTRYMAP"
    EIGENSTRAT = "EIGENSTRAT"
    PED = "PED"
    PACKEDPED = "PACKEDPED"
    PACKEDANCESTRYMAP = "PACKEDANCESTRYMAP"
