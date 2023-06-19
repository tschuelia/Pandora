from __future__ import (
    annotations,
)  # allows type hint Dataset inside Dataset class

import shutil
import tempfile
import textwrap
import subprocess
import random

from pandora.custom_types import *
from pandora.custom_errors import *
from pandora.converter import run_convertf
from pandora.pca import PCA, from_smartpca


class Dataset:
    def __init__(self, file_prefix: FilePath, projected_populations: FilePath = None):
        self.file_prefix: FilePath = file_prefix
        self.file_dir: FilePath = self.file_prefix.parent
        self.name: str = self.file_prefix.name

        self.ind_file: FilePath = pathlib.Path(f"{self.file_prefix}.ind")
        self.geno_file: FilePath = pathlib.Path(f"{self.file_prefix}.geno")
        self.snp_file: FilePath = pathlib.Path(f"{self.file_prefix}.snp")

        self.projected_populations: Union[None, FilePath] = projected_populations

        self.evec_out: Union[None, FilePath] = None
        self.eval_out: Union[None, FilePath] = None
        self.smartpca_log: Union[None, FilePath] = None

        self.pca: Union[None, PCA] = None

        self.samples: Dict[str, str] = {}

    def set_sample_ids_and_populations(self):
        if not self.ind_file.exists():
            raise PandoraConfigException(f"The .ind file {self.ind_file} does not exist.")

        for sample in self.ind_file.open():
            sample_id, _, population = sample.split()
            self.samples[sample_id] = population

    def get_n_unique_populations(self) -> int:
        self.set_sample_ids_and_populations()
        unique_populations = set(self.samples.values())
        return len(unique_populations)

    def files_exist(self):
        return all(
            [self.ind_file.exists(), self.geno_file.exists(), self.snp_file.exists()]
        )

    def _check_smartpca_results_correct(self, n_pcs: int):
        """
        Checks whether existing smartPCA results are correct.
        We consider them to be correct if
        1. the smartPCA run finished properly as indicated by the respective log file
        2. the number of principal components of the existing PCA matches the requested number n_pcs
        """
        # 1. check if the run finished properly, that is indicated by a line "##end of smartpca:"
        # in the smartpca_log file
        if not self.smartpca_log.exists():
            return False

        run_finished = any(["##end of smartpca:" in line for line in self.smartpca_log.open()])
        if not run_finished:
            return False

        # 2. check that the number of PCs of the existing results matches
        # the number of PCs set in the function call here
        pca = from_smartpca(self.evec_out, self.eval_out)
        if pca.n_pcs != n_pcs:
            return False

        return True

    def smartpca(
        self,
        smartpca: Executable,
        n_pcs: int = 20,
        redo: bool = False,
        result_dir: FilePath = None,
        smartpca_optional_settings: Dict = None,
    ):
        if result_dir is None:
            result_dir = self.file_dir

        self.evec_out = result_dir / (self.name + ".evec")
        self.eval_out = result_dir / (self.name + ".eval")
        self.smartpca_log = result_dir / (self.name + ".smartpca.log")

        # check whether the all required output files are already present
        # and whether the smartPCA run finished properly and the number of PCs matches the requested number of PCs
        results_exist = all([self.evec_out.exists(), self.eval_out.exists(), self.smartpca_log.exists()])
        smartpca_finished = self._check_smartpca_results_correct(n_pcs)

        if results_exist and smartpca_finished and not redo:
            self.pca = from_smartpca(self.evec_out, self.eval_out)
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
                evecoutname: {self.evec_out}
                evaloutname: {self.eval_out}
                numoutevec: {n_pcs}
                maxpops: {self.get_n_unique_populations()}
                """

            smartpca_params = textwrap.dedent(smartpca_params)

            if smartpca_optional_settings is not None:
                for k, v in smartpca_optional_settings.items():
                    if isinstance(v, bool):
                        v = "YES" if v else "NO"
                    smartpca_params += f"{k}: {v}\n"

            if self.projected_populations is not None:
                smartpca_params += "lsqproject: YES\n"
                smartpca_params += f"poplistname: {self.projected_populations}\n"

            tmpfile.write(smartpca_params)
            tmpfile.flush()

            cmd = [
                smartpca,
                "-p",
                tmpfile.name,
            ]
            with self.smartpca_log.open("w") as logfile:
                try:
                    subprocess.run(cmd, stdout=logfile, stderr=logfile)
                except subprocess.CalledProcessError:
                    raise RuntimeError(f"Error running smartPCA. "
                                       f"Check the smartPCA logfile {self.smartpca_log.absolute()} for details.")

        self.pca = from_smartpca(self.evec_out, self.eval_out)

    def create_bootstrap(self, bootstrap_prefix: FilePath, seed: int, redo: bool) -> Dataset:
        random.seed(seed)
        bootstrap = Dataset(bootstrap_prefix)

        if bootstrap.files_exist() and not redo:
            return bootstrap

        # when bootstrapping on SNP level, the .ind file does not change
        shutil.copy(self.ind_file, bootstrap.ind_file)

        # sample the SNPs using the snp file
        # each line in the snp file corresponds to one SNP
        num_snps = sum(1 for _ in self.snp_file.open())
        bootstrap_snp_indices = sorted(random.choices(range(num_snps), k=num_snps))

        # 1. Bootstrap the .snp file
        snps = self.snp_file.open().readlines()
        seen_snps = set()

        with bootstrap.snp_file.open(mode="a") as f:
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
        with bootstrap.geno_file.open(mode="ab") as f:
            for bootstrap_idx in bootstrap_snp_indices:
                geno_line = genos[bootstrap_idx]
                f.write(geno_line)

        return bootstrap
