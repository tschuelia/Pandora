import pandas as pd
import pathlib
import subprocess
import tempfile
import textwrap
import tqdm.contrib.concurrent


from .custom_types import *


def get_number_of_populations(indfile: FilePath):
    df = pd.read_table(indfile, delimiter=" ", skipinitialspace=True, header=None)
    return df[2].unique().shape[0]


def run_smartpca(infile_prefix: FilePath, outfile_prefix: FilePath, smartpca: Executable):
    with tempfile.NamedTemporaryFile(mode="w") as tmpfile:
        num_populations = get_number_of_populations(f"{infile_prefix}.ind")

        conversion_content = f"""
        genotypename: {infile_prefix}.geno
        snpname: {infile_prefix}.snp
        indivname: {infile_prefix}.ind
        evecoutname: {outfile_prefix}.evec
        evaloutname: {outfile_prefix}.eval
        numoutevec: 20
        numoutlieriter: 0
        maxpops: {num_populations}
        """

        tmpfile.write(textwrap.dedent(conversion_content))
        tmpfile.flush()

        cmd = [
            smartpca,
            "-p",
            tmpfile.name,
            ">",
            f"{outfile_prefix}.log",
            "2>&1"
        ]
        subprocess.check_output(cmd)


if __name__ == "__main__":
    """
    TODO: CL arguments
    - type of analysis:
        - bootstrapping on SNP level (bool)
        - bootstrapping on individual level (bool)
        - clustering uncertainty (bool)
        - all (= all 4 analysis)
        - n bootstrap samples (default = 100 / bootstopping criterion) (int)
        - n threads (int)
        - save results (bool)
        - data format (string) -> implement conversions of the most common data formats
        - seed (int)
    """
    n_bootstraps = 100
    n_threads = 20


