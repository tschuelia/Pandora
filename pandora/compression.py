import gzip
import os
import shutil

from pandora.custom_types import *


def compress_file(file_in: FilePath):
    with open(file_in, "rb") as f_in:
        with gzip.open(file_in + ".gz", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(file_in)
