"""Numpy uses OMP and BLAS for its linear algebra under the hood.

Per default, both libraries use all cores on the machine. Since we are doing a pairwise
comparison of all bootstrap/window replicates, for a substantial amount of time Pandora
would require all cores on the machine, which is not an option when working on shared
servers/clusters. Therefore, we set the OMP and BLAS threads manually to 1 to prevent
this. Setting these variables needs to happen before the first import of numpy.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__name__)
except Exception:
    __version__ = "unknown"
