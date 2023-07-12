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
from sklearn.manifold import MDS as sklearnMDS

from pandora.custom_types import *
from pandora.custom_errors import *
from pandora.embedding import PCA, from_smartpca, MDS, from_sklearn_mds


def _check_geno_file(geno_file: pathlib.Path):
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
            raise PandoraConfigException(
                f"The .geno file {geno_file} seems to be wrong. "
                f"All values must be in {allowed_genos}."
            )
    if seen_genos > allowed_genos:
        raise PandoraConfigException(
            f"The .geno file {geno_file} seems to be wrong. "
            f"All values must be in {allowed_genos}."
        )

    # each line should have the same number of values
    if len(line_lengths) != 1:
        raise PandoraConfigException(
            f"The .geno file {geno_file} seems to be wrong. "
            f"All samples must have the same number of SNPs."
        )


def _check_ind_file(ind_file: pathlib.Path):
    # each line should contain three values
    seen_inds = set()
    total_inds = 0

    for line in ind_file.open():
        try:
            ind_id, _, _ = line.strip().split()
        except ValueError:
            # too few or too many lines
            raise PandoraConfigException(
                f"The .ind file {ind_file} seems to be wrong. All lines should contain three values."
            )

        seen_inds.add(ind_id.strip())
        total_inds += 1

    # make sure all individuals have a unique ID
    if len(seen_inds) != total_inds:
        raise PandoraConfigException(
            f"The .ind file {ind_file} seems to be wrong. Duplicate sample IDs found."
        )


def _check_snp_file(snp_file: pathlib.Path):
    seen_snps = set()
    total_snps = 0

    line_lengths = set()

    for line in snp_file.open():
        line = line.strip()
        # each line contains 4, 5, or 6 values
        n_values = len(line.split())
        line_lengths.add(n_values)
        if n_values < 4 or n_values > 6:
            raise PandoraConfigException(
                f"The .snp file {snp_file} seems to be wrong. All lines need to contain either 4, 5, or 6 values."
            )

        snp_name, chrom, *_ = line.split()
        seen_snps.add(snp_name.strip())
        total_snps += 1

    # all lines should contain the same number of values
    if len(line_lengths) != 1:
        raise PandoraConfigException(
            f"The .snp file {snp_file} seems to be wrong. All lines need to contain the same number of values."
        )

    # make sure all SNPs have a unique ID
    if len(seen_snps) != total_snps:
        raise PandoraConfigException(
            f"The .snp file {snp_file} seems to be wrong. Duplicate SNP IDs found."
        )


def smartpca_finished(n_components: int, result_prefix: pathlib.Path) -> bool:
    """
    Checks whether existing smartPCA results are correct.
    We consider them to be correct if
    1. the smartPCA run finished properly as indicated by the respective log file
    2. the number of principal components of the existing PCA matches the requested number n_components

    Args:
        n_components (int): Number of principal components expected in the result files.
        result_prefix (pathlib.Path): File path prefix pointing to the output of the smartpca run to check.

    Returns:
        bool: Whether the respective smartpca run finished properly.
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
    if pca.n_components != n_components:
        return False

    return True


def get_embedding_populations(
    embedding_populations: Optional[pathlib.Path],
) -> List[str]:
    """
    Return a list of populations to use for the embedding computation.

    Args:
        embedding_populations (pathlib.Path): Filepath containing the newline-separated list of populations.

    Returns:
        List[str]: List of populations to use for the embedding computation.

    """
    if embedding_populations is None:
        return []
    return [pop.strip() for pop in embedding_populations.open()]


def _deduplicate_snp_id(snp_id: str, seen_snps: Set[str]):
    """
    Deduplicates the SNP IDs to make sure all SNPs are
    Args:
        snp_id:
        seen_snps:

    Returns:

    """
    deduplicate = snp_id

    snp_id_counter = 0

    while deduplicate in seen_snps:
        snp_id_counter += 1
        deduplicate = f"{snp_id}_r{snp_id_counter}"

    return deduplicate


class Dataset:
    def __init__(
        self,
        file_prefix: pathlib.Path,
        embedding_populations: Optional[pathlib.Path] = None,
        samples: pd.DataFrame = None,
    ):
        # TODO: in Documentation muss stehen, dass dataset im EIGENSTRAT format erwartet wird sonst funktioniert bootstrapping nicht
        self.file_prefix: pathlib.Path = file_prefix
        self.file_dir: pathlib.Path = self.file_prefix.parent
        self.name: str = self.file_prefix.name

        self.ind_file: pathlib.Path = pathlib.Path(f"{self.file_prefix}.ind")
        self.geno_file: pathlib.Path = pathlib.Path(f"{self.file_prefix}.geno")
        self.snp_file: pathlib.Path = pathlib.Path(f"{self.file_prefix}.snp")

        self.embedding_populations_file: pathlib.Path = embedding_populations
        self.embedding_populations: List[str] = get_embedding_populations(
            self.embedding_populations_file
        )
        self.samples: pd.DataFrame = (
            self.get_sample_info() if samples is None else samples
        )
        self.projected_samples: pd.DataFrame = self.samples.loc[
            lambda x: ~x.used_for_embedding
        ]

        self.pca: Union[None, PCA] = None
        self.mds: Union[None, MDS] = None

    def get_sample_info(self) -> pd.DataFrame:
        """
        Extracts metadata for each sample in self.ind_file.

        Returns:
            pd.DataFrame: Pandas dataframe with the following columns:
                * sample_id (str): ID of the sample
                * sex (str): sex of the sample
                * population (str): population the sample belongs to
                * used_for_embedding (bool): whether the sample should be used to compute a dimensionality reduction Embedding
                    Decided based on self.embedding_populations
        """
        if not self.ind_file.exists():
            raise PandoraConfigException(
                f"The .ind file {self.ind_file} does not exist."
            )

        populations_for_embedding = get_embedding_populations(
            self.embedding_populations_file
        )

        data = {"sample_id": [], "sex": [], "population": [], "used_for_embedding": []}
        for sample in self.ind_file.open():
            sample_id, sex, population = sample.split()
            data["sample_id"].append(sample_id.strip())
            data["sex"].append(sex.strip())
            data["population"].append(population.strip())

            if len(populations_for_embedding) == 0:
                # all samples used for PCA
                data["used_for_embedding"].append(True)
            else:
                data["used_for_embedding"].append(
                    population in populations_for_embedding
                )

        return pd.DataFrame(data=data)

    def get_sequence_length(self) -> int:
        """
        Counts and returns the number of SNPs in self.geno_file

        Returns:
            int: Number of SNPs in self.geno_file.
        """
        return sum(1 for _ in self.geno_file.open(mode="rb"))

    def files_exist(self) -> bool:
        """
        Checks whether all required input files (geno, snp, ind) exist.

        Returns:
            bool: True, if all three files are present, False otherwise.
        """
        return all(
            [self.ind_file.exists(), self.geno_file.exists(), self.snp_file.exists()]
        )

    def check_files(self) -> None:
        """
        Checks whether all input files (geno, snp, ind) are in correct format according to the EIGENSOFT specification.

        Raises:
            PandoraException: If any of the three files is malformatted.
        """
        _check_ind_file(self.ind_file)
        _check_geno_file(self.geno_file)
        _check_snp_file(self.snp_file)

    def remove_input_files(self) -> None:
        """
        Removes all three input files (self.ind_file, self.geno_file, self.snp_file).
        This is useful if you want to save storage space and don't need the input files anymore (e.g. for bootstrap replicates).

        """
        self.ind_file.unlink(missing_ok=True)
        self.geno_file.unlink(missing_ok=True)
        self.snp_file.unlink(missing_ok=True)

    def smartpca(
        self,
        smartpca: Executable,
        n_pcs: int = 20,
        result_dir: Optional[pathlib.Path] = None,
        redo: bool = False,
        smartpca_optional_settings: Optional[Dict] = None,
    ) -> None:
        """
        Runs the EIGENSOFT smartpca on the dataset and assigns its PCA result to self.pca.
        Additional smartpca options can be passed as dictionary in smartpca_optional_settings, e.g.
        `smartpca_optional_settings = dict(numoutlieriter=0, shrinkmode=True)`.

        Args:
            smartpca (Executable):
            n_pcs (int): Number of principal components to output. Default is 20.
            result_dir (Optional[pathlib.Path]): File path pointing where to write the results to.
                Default is the directory where all input files are.
            redo (bool): Whether to redo the analysis, if all outfiles are already present and correct.
            smartpca_optional_settings (Optional[Dict]): Additional smartpca settings.
                Not allowed are the following options: genotypename, snpname, indivname,
                evecoutname, evaloutname, numoutevec, maxpops.

        """
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

            if self.embedding_populations_file is not None:
                smartpca_params += "lsqproject: YES\n"
                smartpca_params += f"poplistname: {self.embedding_populations_file}\n"

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
                    raise RuntimeError(
                        f"Error running smartPCA. "
                        f"Check the smartPCA logfile {smartpca_log.absolute()} for details."
                    )

        self.pca = from_smartpca(evec_out, eval_out)

    def _compute_fst_matrix(
        self, smartpca: Executable, fst_file: pathlib.Path, smartpca_log: pathlib.Path
    ):
        with tempfile.NamedTemporaryFile(mode="w") as tmpfile:
            smartpca_fst_params = f"""
                genotypename: {self.geno_file}
                snpname: {self.snp_file}
                indivname: {self.ind_file}
                phylipoutname: {fst_file}
                fstonly: YES
                maxpops: {self.samples.population.unique().shape[0]}
                """

            tmpfile.write(smartpca_fst_params)
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
                    raise RuntimeError(
                        f"Error computing FST matrix using smartPCA. "
                        f"Check the smartPCA logfile {smartpca_log.absolute()} for details."
                    )

    def run_mds(
        self,
        smartpca: Executable,
        n_components: int = 20,
        result_dir: Optional[pathlib.Path] = None,
        redo: bool = False,
    ) -> None:
        if result_dir is None:
            result_dir = self.file_dir

        fst_file = result_dir / (self.name + ".fst")
        smartpca_log = result_dir / (self.name + ".smartpca.log")
        print(fst_file)

        if not fst_file.exists() or not smartpca_log.exists() or redo:
            # TODO: check if results exist and are correct
            self._compute_fst_matrix(smartpca, fst_file, smartpca_log)

        with fst_file.open() as f:
            shape = int(f.readline())
            cols = [f"dist{i}" for i in range(shape)]
            fst_results = pd.read_table(
                f, delimiter=" ", skipinitialspace=True, names=["_"] + cols
            )

        # extract correct population names as smartpca truncates them in the output if they are too long
        populations = []
        for line in smartpca_log.open():
            line = line.strip()
            if not line.startswith("population:"):
                continue
            # population:   0   population   11
            _, _, population, *_ = line.split()
            populations.append(population.strip())

        mds = sklearnMDS(n_components=n_components, dissimilarity="precomputed")
        embedding = mds.fit_transform(fst_results[cols])
        embedding = pd.DataFrame(data=embedding)
        embedding["population"] = populations

        self.mds = from_sklearn_mds(embedding, self.samples, mds.stress_)

    def create_bootstrap(
        self, bootstrap_prefix: pathlib.Path, seed: int, redo: bool = False
    ) -> Dataset:
        """
        Creates a bootstrap dataset based on the content of self. Bootstraps the dataset by resampling SNPs with replacement.

        Args:
            bootstrap_prefix (pathlib.Path): Prefix of the file path where to write the bootstrap dataset to.
                The resulting files will be  `bootstrap_prefix.geno`, `bootstrap_prefix.ind`, and `bootstrap_prefix.snp`.
            seed (int): Seed to initialize the random number generator before drawing the bootstraps.
            redo (bool): Whether to redo the bootstrap if all output files are present. Default is False.

        Returns:
            Dataset: A new dataset object containing the bootstrap replicate.
        """
        bs_ind_file = pathlib.Path(f"{bootstrap_prefix}.ind")
        bs_geno_file = pathlib.Path(f"{bootstrap_prefix}.geno")
        bs_snp_file = pathlib.Path(f"{bootstrap_prefix}.snp")

        files_exist = all(
            [bs_ind_file.exists(), bs_geno_file.exists(), bs_snp_file.exists()]
        )

        if files_exist and not redo:
            return Dataset(bootstrap_prefix, self.embedding_populations_file)

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
                f.write(
                    str(seed) + ";" + ", ".join([str(i) for i in bootstrap_snp_indices])
                )

        # 1. Bootstrap the .snp file
        snps = self.snp_file.open().readlines()
        seen_snps = set()

        with bs_snp_file.open(mode="a") as f:
            for bootstrap_idx in bootstrap_snp_indices:
                snp_line = snps[bootstrap_idx].strip()

                # lines look like this:
                # 1   SampleExample 0 0 1 1
                snp_id, chrom_id, rest = snp_line.split(maxsplit=2)
                snp_id = _deduplicate_snp_id(snp_id, seen_snps)
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

        return Dataset(bootstrap_prefix, self.embedding_populations_file)

    def _generate_windowed_dataset(
        self, window_start: int, window_end: int, result_prefix: pathlib.Path
    ) -> Dataset:
        ind_file = pathlib.Path(f"{result_prefix}.ind")
        geno_file = pathlib.Path(f"{result_prefix}.geno")
        snp_file = pathlib.Path(f"{result_prefix}.snp")

        # .ind file does not change
        shutil.copy(self.ind_file, ind_file)

        # .snp file needs to be filtered
        snps = self.snp_file.open().readlines()
        window_snps = snps[window_start:window_end]
        with snp_file.open(mode="w") as f:
            f.write("".join(window_snps))

        # .geno file needs to be filtered as well
        genos = self.geno_file.open(mode="rb").readlines()
        window_genos = genos[window_start:window_end]
        with geno_file.open(mode="wb") as f:
            f.write(b"".join(window_genos))

        return Dataset(result_prefix)

    def get_windows(
        self, result_dir: pathlib.Path, n_windows: int = 100
    ) -> List[Dataset]:
        """
        Creates n_windows new Dataset objects as overlapping sliding windows over self.
        Let M = number of SNPs in self and N = n_windows.
        Each dataset will have a window size of int(M / N + (M / 2 * N)) SNPs.
        The stride is int(M / N) and the overlap between windows is thus int(M / 2 * N) SNPs.
        Note that the last Dataset will contain fewer SNPs as there is no following window to overlap with.
        However, due to rounding, the number of SNPs in the final Dataset will not simply be overlap fewer.

        Args:
            result_dir (pathlib.Path): Directory where to store the resulting Dataset files in.
                Each window Dataset will be named "window_{i}" for i in range(n_windows)
            n_windows (int): Number of windowed datasets to generate. Default is 100.

        Returns:
            List[Dataset]: List of n_windows new Dataset objects as overlapping windows over self.

        """
        n_snps = self.get_sequence_length()
        overlap = int((n_snps / n_windows) / 2)
        stride = int(n_snps / n_windows)
        window_size = overlap + stride

        windows = []

        for i in range(n_windows):
            window_prefix = pathlib.Path(f"{result_dir}/window_{i}")
            window_start = i * stride
            window_end = i * stride + window_size
            windows.append(
                self._generate_windowed_dataset(window_start, window_end, window_prefix)
            )

        return windows
