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


# Fix a race condition in the internal ``multiprocessing.process`` cleanup logic that could manifest as an unintended
# ``AttributeError`` during ``process.start``.
# Backport of '_cleanup' from Python 3.12 to older Python distributions
# https://github.com/python/cpython/pull/104537
import multiprocessing
import sys


def multiprocessing_process_cleanup():
    # check for processes which have finished
    for p in list(multiprocessing.process._children):
        if (child_popen := p._popen) and child_popen.poll() is not None:
            multiprocessing.process._children.discard(p)


if sys.version_info.major == 3 and sys.version_info.minor < 12:
    # race condition is fixed in Python 3.12
    multiprocessing.process._cleanup = multiprocessing_process_cleanup


import importlib.metadata

try:
    __version__ = importlib.metadata.version(__name__)
except Exception:
    __version__ = "unknown"
