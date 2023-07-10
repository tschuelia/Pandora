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
from pandora.pca import PCA, from_smartpca


def check_geno_file(geno_file: pathlib.Path):
    # should only contain int values
    allowed_genos = {0, 1, 2, 9}
    seen_genos = set()

    line_lengths = set()
    for line in geno_file.open():
        line = line.strip()
        line_lengths.add(len(line))
        try:
            [seen_genos.add(int(c)) for c in line]
        except ValueError:
            # one char seems to not be an int
            raise PandoraConfigException(f"The .geno file {geno_file} seems to be wrong. "
                                         f"All values must be in {allowed_genos}.")
    if seen_genos > allowed_genos:
        raise PandoraConfigException(f"The .geno file {geno_file} seems to be wrong. "
                                     f"All values must be in {allowed_genos}.")

    # each line should have the same number of values
    if len(line_lengths) != 1:
        raise PandoraConfigException(f"The .geno file {geno_file} seems to be wrong. "
                                     f"All samples must have the same number of SNPs.")


def check_ind_file(ind_file: pathlib.Path):
    # each line should contain three values
    seen_inds = set()
    total_inds = 0

    for line in ind_file.open():
        try:
            ind_id, _, _ = line.strip().split()
        except ValueError:
            # too few or too many lines
            raise PandoraConfigException(f"The .ind file {ind_file} seems to be wrong. All lines should contain three values.")

        seen_inds.add(ind_id.strip())
        total_inds += 1

    # make sure all individuals have a unique ID
    if len(seen_inds) != total_inds:
        raise PandoraConfigException(f"The .ind file {ind_file} seems to be wrong. Duplicate sample IDs found.")


def check_snp_file(snp_file: pathlib.Path):
    seen_snps = set()
    total_snps = 0

    line_lengths = set()

    for line in snp_file.open():
        line = line.strip()
        # each line contains 4, 5, or 6 values
        n_values = len(line.split())
        line_lengths.add(n_values)
        if n_values < 4 or n_values > 6:
            raise PandoraConfigException(f"The .snp file {snp_file} seems to be wrong. All lines need to contain either 4, 5, or 6 values.")

        snp_name, chrom, *_ = line.split()
        seen_snps.add(snp_name.strip())
        total_snps += 1

    # all lines should contain the same number of values
    if len(line_lengths) != 1:
        raise PandoraConfigException(
            f"The .snp file {snp_file} seems to be wrong. All lines need to contain the same number of values.")

    # make sure all SNPs have a unique ID
    if len(seen_snps) != total_snps:
        raise PandoraConfigException(f"The .snp file {snp_file} seems to be wrong. Duplicate SNP IDs found.")


def smartpca_finished(n_pcs: int, result_prefix: pathlib.Path):
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
    if pca.n_components != n_pcs:
        return False

    return True


def get_pca_populations(pca_populations: Optional[pathlib.Path]):
    if pca_populations is None:
        return []
    return [pop.strip() for pop in pca_populations.open()]


def deduplicate_snp_id(snp_id: str, seen_snps: Set[str]):
    deduplicate = snp_id

    snp_id_counter = 0

    while deduplicate in seen_snps:
        snp_id_counter += 1
        deduplicate = f"{snp_id}_r{snp_id_counter}"

    return deduplicate


class Dataset:
    def __init__(self, file_prefix: pathlib.Path, pca_populations: Optional[pathlib.Path] = None, samples: pd.DataFrame = None):
        self.file_prefix: pathlib.Path = file_prefix
        self.file_dir: pathlib.Path = self.file_prefix.parent
        self.name: str = self.file_prefix.name

        self.ind_file: pathlib.Path = pathlib.Path(f"{self.file_prefix}.ind")
        self.geno_file: pathlib.Path = pathlib.Path(f"{self.file_prefix}.geno")
        self.snp_file: pathlib.Path = pathlib.Path(f"{self.file_prefix}.snp")

        self.pca_populations_file: pathlib.Path = pca_populations
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

    def get_sequence_length(self) -> int:
        """
        Counts and returns the number of SNPs in self.snp_file

        Returns:
            (int): Number of SNPs in self.snp_file.
        """
        return sum(1 for _ in self.snp_file.open())

    def files_exist(self):
        return all(
            [self.ind_file.exists(), self.geno_file.exists(), self.snp_file.exists()]
        )

    def check_files(self):
        check_ind_file(self.ind_file)
        check_geno_file(self.geno_file)
        check_snp_file(self.snp_file)

    def remove_input_files(self):
        self.ind_file.unlink(missing_ok=True)
        self.geno_file.unlink(missing_ok=True)
        self.snp_file.unlink(missing_ok=True)

    def smartpca(
        self,
        smartpca: Executable,
        n_pcs: int = 20,
        result_dir: pathlib.Path = None,
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
        # check that all required input files are correctly formatted
        self.check_files()

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
                    subprocess.run(cmd, stdout=logfile, stderr=logfile, check=True)
                except subprocess.CalledProcessError:
                    raise RuntimeError(f"Error running smartPCA. "
                                       f"Check the smartPCA logfile {smartpca_log.absolute()} for details.")

        self.pca = from_smartpca(evec_out, eval_out)


    def mds(self, n_components: int = 20):
        raise NotImplementedError

    def create_bootstrap(self, bootstrap_prefix: pathlib.Path, seed: int, redo: bool) -> Dataset:
        bs_ind_file = pathlib.Path(f"{bootstrap_prefix}.ind")
        bs_geno_file = pathlib.Path(f"{bootstrap_prefix}.geno")
        bs_snp_file = pathlib.Path(f"{bootstrap_prefix}.snp")

        files_exist = all([bs_ind_file.exists(), bs_geno_file.exists(), bs_snp_file.exists()])

        if files_exist and not redo:
            return Dataset(bootstrap_prefix, self.pca_populations_file)

        bs_ind_file.unlink(missing_ok=True)
        bs_geno_file.unlink(missing_ok=True)
        bs_snp_file.unlink(missing_ok=True)

        # sample the SNPs using the snp file
        # each line in the snp file corresponds to one SNP
        num_snps = self.get_sequence_length()

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
                snp_line = snps[bootstrap_idx].strip()

                # lines look like this:
                # 1   SampleExample 0 0 1 1
                snp_id, chrom_id, rest = snp_line.split(maxsplit=2)
                snp_id = deduplicate_snp_id(snp_id, seen_snps)
                seen_snps.add(snp_id)
                f.write(f"{snp_id} {chrom_id} {rest}\n")

        # 2. Bootstrap the .geno file using the bootstrap_snp_indices above
        # the .geno file contains one column for each individual sample
        # to sample SNPs we therefore need to sample the rows
        genos = self.geno_file.open(mode="rb").readlines()
        with bs_geno_file.open(mode="ab") as f:
            for bootstrap_idx in bootstrap_snp_indices:
                geno_line = genos[bootstrap_idx].strip()
                f.write(geno_line)
                f.write(b"\n")

        # when bootstrapping on SNP level, the .ind file does not change
        shutil.copy(self.ind_file, bs_ind_file)

        return Dataset(bootstrap_prefix, self.pca_populations_file)

    def get_windows(self, window_size: Union[int, float], stride: int) -> List[Dataset]:
        """
        Creates new Dataset objects as overlapping sliding windows over self. The resulting datasets have sequences of
        length window_size or window_size * self.get_sequence_length() and their overlap can be controlled via the
        stride parameter.

        Args:
            window_size:
            stride:

        Returns:

        """
        raise NotImplementedError
