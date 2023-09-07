import pathlib
import tempfile

import pandas as pd
import pytest

from pandora.converter import get_filenames, run_convertf
from pandora.custom_types import FileFormat


@pytest.mark.parametrize("out_format", FileFormat._member_names_)
def test_convert_forth_and_back_should_result_in_identical_files(
    example_eigen_dataset_prefix, convertf, out_format
):
    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = pathlib.Path(tmpdir) / "converted"
        reconverted_prefix = pathlib.Path(tmpdir) / "reconverted"

        # all file conversions should work without any issues
        run_convertf(
            convertf=convertf,
            in_prefix=example_eigen_dataset_prefix,
            in_format=FileFormat.EIGENSTRAT,
            out_prefix=out_prefix,
            out_format=FileFormat[out_format],
            redo=True,
        )

        # convert the data back to EIGENSTRAT format and compare the files
        run_convertf(
            convertf=convertf,
            in_prefix=out_prefix,
            in_format=FileFormat[out_format],
            out_prefix=reconverted_prefix,
            out_format=FileFormat.EIGENSTRAT,
            redo=True,
        )

        geno_in, snp_in, ind_in = get_filenames(
            example_eigen_dataset_prefix, FileFormat.EIGENSTRAT
        )
        geno_out, snp_out, ind_out = get_filenames(
            reconverted_prefix, FileFormat.EIGENSTRAT
        )

        # we can compare the geno files directly
        assert geno_in.open().read() == geno_out.open().read()

        # for comparing the SNP files we load them as pandas dataframes
        content_in = pd.read_table(
            geno_in.open(), skipinitialspace=True, delimiter=" ", header=None
        )
        content_out = pd.read_table(
            geno_out.open(), skipinitialspace=True, delimiter=" ", header=None
        )

        assert content_in.shape == content_out.shape
        assert content_in.equals(content_out)

        # for the ind file we can only compare the sample IDs as the population and sex information might be lost for
        # most file conversions (e.g. PED format has no populations)
        content_in = pd.read_table(
            geno_in.open(),
            skipinitialspace=True,
            delimiter=" ",
            names=["sample_id", "sex", "population"],
        )
        content_out = pd.read_table(
            geno_out.open(),
            skipinitialspace=True,
            delimiter=" ",
            names=["sample_id", "sex", "population"],
        )
        pd.testing.assert_series_equal(content_in.sample_id, content_out.sample_id)
