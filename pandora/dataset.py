from __future__ import (
    annotations,
)  # allows type hint Dataset inside Dataset class

import pathlib
import shutil
import tempfile
import textwrap
import subprocess
import random

import pandas as pd

from pandora.custom_types import *
from pandora.custom_errors import *
from pandora.pca import PCA, from_smartpca


def smartpca_finished(n_pcs: int, result_prefix: FilePath,):
    """
    Checks whether existing smartPCA results are correct.
    We consider them to be correct if
    1. the smartPCA run finished properly as indicated by the respective log file
    2. the number of principal components of the existing PCA matches the requested number n_pcs
    """
    evec_out = pathlib.Path(f"{result_prefix}.evec")
    eval_out = pathlib.Path(f"{result_prefix}.eval")
    smartpca_log = pathlib.Path(f"{result_prefix}.smartpca.log")

    results_exist = all([evec_out.exists(), eval_out.exists(), smartpca_log.exists()])
    if not results_exist:
        return False

    # 1. check if the run finished properly, that is indicated by a line "##end of smartpca:"
    # in the smartpca_log file

    run_finished = any(["##end of smartpca:" in line for line in smartpca_log.open()])
    if not run_finished:
        return False

    # 2. check that the number of PCs of the existing results matches
    # the number of PCs set in the function call here
    pca = from_smartpca(evec_out, eval_out)
    if pca.n_pcs != n_pcs:
        return False

    return True


def get_pca_populations(pca_populations: Optional[FilePath]):
    if pca_populations is None:
        return []
    return [pop.strip() for pop in pca_populations.open()]


class Dataset:
    def __init__(self, file_prefix: FilePath, pca_populations: Optional[FilePath] = None, samples: pd.DataFrame = None):
        self.file_prefix: FilePath = file_prefix
        self.file_dir: FilePath = self.file_prefix.parent
        self.name: str = self.file_prefix.name

        self.ind_file: FilePath = pathlib.Path(f"{self.file_prefix}.ind")
        self.geno_file: FilePath = pathlib.Path(f"{self.file_prefix}.geno")
        self.snp_file: FilePath = pathlib.Path(f"{self.file_prefix}.snp")

        self.pca_populations_file: FilePath = pca_populations
        self.pca_populations: List[str] = get_pca_populations(self.pca_populations_file)
        self.samples: pd.DataFrame = self.get_sample_info() if samples is None else samples
        self.projected_samples: pd.DataFrame = self.samples.loc[lambda x: ~x.used_for_pca]

        self.pca: Union[None, PCA] = None

    def get_sample_info(self) -> pd.DataFrame:
        if not self.ind_file.exists():
            raise PandoraConfigException(f"The .ind file {self.ind_file} does not exist.")

        populations_for_pca = get_pca_populations(self.pca_populations_file)

        data = {
            "sample_id": [],
            "sex": [],
            "population": [],
            "used_for_pca": []
        }
        for sample in self.ind_file.open():
            sample_id, sex, population = sample.split()
            data["sample_id"].append(sample_id.strip())
            data["sex"].append(sex.strip())
            data["population"].append(population.strip())

            if len(populations_for_pca) == 0:
                # all samples used for PCA
                data["used_for_pca"].append(True)
            else:
                data["used_for_pca"].append(population in populations_for_pca)

        return pd.DataFrame(data=data)

    def files_exist(self):
        return all(
            [self.ind_file.exists(), self.geno_file.exists(), self.snp_file.exists()]
        )

    def remove_input_files(self):
        self.ind_file.unlink(missing_ok=True)
        self.geno_file.unlink(missing_ok=True)
        self.snp_file.unlink(missing_ok=True)

    def smartpca(
        self,
        smartpca: Executable,
        n_pcs: int = 20,
        result_dir: FilePath = None,
        redo: bool = False,
        smartpca_optional_settings: Dict = None,
    ):
        if result_dir is None:
            result_dir = self.file_dir

        evec_out = result_dir / (self.name + ".evec")
        eval_out = result_dir / (self.name + ".eval")
        smartpca_log = result_dir / (self.name + ".smartpca.log")

        # check whether the all required output files are already present
        # and whether the smartPCA run finished properly and the number of PCs matches the requested number of PCs

        if smartpca_finished(n_pcs, result_dir / self.name) and not redo:
            self.pca = from_smartpca(evec_out, eval_out)
            return

        # check that all required input files are present
        if not self.files_exist:
            raise PandoraConfigException(
                f"Not all input files for file prefix {self.file_prefix} present. "
                f"Looking for files in EIGEN format with endings .geno, .snp, and .ind"
            )

        with tempfile.NamedTemporaryFile(mode="w") as tmpfile:
            smartpca_params = f"""
                genotypename: {self.geno_file}
                snpname: {self.snp_file}
                indivname: {self.ind_file}
                evecoutname: {evec_out}
                evaloutname: {eval_out}
                numoutevec: {n_pcs}
                maxpops: {self.samples.population.unique().shape[0]}
                """

            smartpca_params = textwrap.dedent(smartpca_params)

            if smartpca_optional_settings is not None:
                for k, v in smartpca_optional_settings.items():
                    if isinstance(v, bool):
                        v = "YES" if v else "NO"
                    smartpca_params += f"{k}: {v}\n"

            if self.pca_populations_file is not None:
                smartpca_params += "lsqproject: YES\n"
                smartpca_params += f"poplistname: {self.pca_populations_file}\n"

            tmpfile.write(smartpca_params)
            tmpfile.flush()

            cmd = [
                smartpca,
                "-p",
                tmpfile.name,
            ]
            with smartpca_log.open("w") as logfile:
                try:
                    subprocess.run(cmd, stdout=logfile, stderr=logfile)
                except subprocess.CalledProcessError:
                    raise RuntimeError(f"Error running smartPCA. "
                                       f"Check the smartPCA logfile {smartpca_log.absolute()} for details.")

        self.pca = from_smartpca(evec_out, eval_out)

    def create_bootstrap(self, bootstrap_prefix: FilePath, seed: int, redo: bool) -> Dataset:
        bs_ind_file = pathlib.Path(f"{bootstrap_prefix}.ind")
        bs_geno_file = pathlib.Path(f"{bootstrap_prefix}.geno")
        bs_snp_file = pathlib.Path(f"{bootstrap_prefix}.snp")

        files_exist = all([bs_ind_file.exists(), bs_geno_file.exists(), bs_snp_file.exists()])

        if files_exist and not redo:
            return Dataset(bootstrap_prefix, self.pca_populations_file)

        # sample the SNPs using the snp file
        # each line in the snp file corresponds to one SNP
        num_snps = sum(1 for _ in self.snp_file.open())

        # check if a checkpoint with SNP indices exists
        ckp_file = pathlib.Path(f"{bootstrap_prefix}.ckp")
        if ckp_file.exists() and not redo:
            seed, indices = ckp_file.open().readline().split(";")
            random.seed(int(seed))
            bootstrap_snp_indices = [int(v) for v in indices.split(",")]
        else:
            random.seed(seed)
            bootstrap_snp_indices = sorted(random.choices(range(num_snps), k=num_snps))

            # store random seed and selected SNP indices as checkpoint for reproducibility
            with ckp_file.open("w") as f:
                f.write(str(seed) + ";" + ", ".join([str(i) for i in bootstrap_snp_indices]))

        # 1. Bootstrap the .snp file
        snps = self.snp_file.open().readlines()
        seen_snps = set()

        with bs_snp_file.open(mode="a") as f:
            for bootstrap_idx in bootstrap_snp_indices:
                snp_line = snps[bootstrap_idx]

                # lines look like this:
                # 1   SampleExample 0 0 1 1
                snp_id, chrom_id, rest = snp_line.split(maxsplit=2)
                deduplicate = snp_id

                snp_id_counter = 0

                while deduplicate in seen_snps:
                    snp_id_counter += 1
                    deduplicate = f"{snp_id}_r{snp_id_counter}"

                seen_snps.add(deduplicate)
                f.write(f"{deduplicate} {chrom_id} {rest}")

        # 2. Bootstrap the .geno file using the bootstrap_snp_indices above
        # the .geno file contains one column for each individual sample
        # to sample SNPs we therefore need to sample the rows
        genos = self.geno_file.open(mode="rb").readlines()
        with bs_geno_file.open(mode="ab") as f:
            for bootstrap_idx in bootstrap_snp_indices:
                geno_line = genos[bootstrap_idx]
                f.write(geno_line)

        # when bootstrapping on SNP level, the .ind file does not change
        shutil.copy(self.ind_file, bs_ind_file)

        return Dataset(bootstrap_prefix, self.pca_populations_file)
