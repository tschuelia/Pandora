import tempfile

import pytest

from pandora.converter import *


@pytest.mark.parametrize("out_format", FileFormat._member_names_)
def test_convert_forth_and_back_should_result_in_identical_files(
    example_eigen_dataset_prefix, convertf, out_format
):
    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = pathlib.Path(tmpdir) / ""

        # all file conversions should work without any issues
        run_convertf(
            convertf=convertf,
            in_prefix=example_eigen_dataset_prefix,
            in_format=FileFormat.EIGENSTRAT,
            out_prefix=out_prefix,
            out_format=FileFormat[out_format],
            redo=True,
        )
