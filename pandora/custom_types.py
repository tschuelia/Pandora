import numpy as np
import pandas as pd
import pathlib
from typing import Union, Tuple, Dict, List


FilePath = Union[str, pathlib.Path]
Executable = Union[str, pathlib.Path]
_PCA = Union[np.ndarray, pd.DataFrame]