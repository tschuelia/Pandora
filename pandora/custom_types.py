import numpy as np
import numpy.typing as npt
import pandas as pd
import pathlib
from enum import Enum
from typing import Union, Tuple, Dict, List, Iterable, Optional, Set


Executable = Union[str, pathlib.Path]


class FileFormat(Enum):
    ANCESTRYMAP = "ANCESTRYMAP"
    EIGENSTRAT = "EIGENSTRAT"
    PED = "PED"
    PACKEDPED = "PACKEDPED"
    PACKEDANCESTRYMAP = "PACKEDANCESTRYMAP"
