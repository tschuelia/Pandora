from __future__ import (  # allows type hint EigenDataset inside EigenDataset class
    annotations,
)

import pathlib
import random
import re
import shutil
import subprocess
import tempfile
import textwrap
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from numpy import typing as npt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.manifold import MDS as sklearnMDS

from pandora.custom_errors import PandoraConfigException, PandoraException
from pandora.custom_types import Executable
from pandora.distance_metrics import euclidean_sample_distance
from pandora.embedding import MDS, PCA, from_sklearn_mds, from_smartpca
from pandora.imputation import impute_data


def _check_geno_file(geno_file: pathlib.Path):
    # should only contain int values
    allowed_genos = {0, 1, 2, 9}
    seen_genos = set()

    try:
        geno_content = geno_file.open().readlines()
    except UnicodeDecodeError:
        raise PandoraException(
            "The provided dataset does not seem to be in EIGENSTRAT format. "
            "Make sure to convert it using convertf or the provided method in conversion.py."
        )

    line_lengths = set()
    for line in geno_content:
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


def _parse_smartpca_fst_results(
    result_prefix: pathlib.Path,
) -> Tuple[npt.NDArray, pd.Series]:
    fst_file = pathlib.Path(f"{result_prefix}.fst")
    smartpca_log = pathlib.Path(f"{result_prefix}.smartpca.log")

    fst_content = fst_file.open().read()
    # The smartpca output does not delimite negative numbers with a space, so we have to manually fix it
    # But we don't want to potentially split population names with dashes, so we match for a digit
    fst_content = re.sub(r"-(\d)", r" -\g<1>", fst_content)
    fst_file.open("w").write(fst_content)

    with fst_file.open() as f:
        shape = int(f.readline())
        cols = [f"dist{i}" for i in range(shape)]
        fst_results = pd.read_table(
            f, delimiter=" ", skipinitialspace=True, names=["_"] + cols
        )
        fst_results = fst_results[cols].to_numpy()

    # extract correct population names as smartpca truncates them in the output if they are too long
    populations = []
    for line in smartpca_log.open():
        line = line.strip()
        if not line.startswith("population:"):
            continue
        # population:   0   population   11
        _, _, population, *_ = line.split()
        populations.append(population.strip())

    return fst_results, pd.Series(populations)


def _smartpca_finished(n_components: int, result_prefix: pathlib.Path) -> bool:
    """Checks whether existing smartPCA results are correct.

    We consider them to be correct if\n
    1. the smartPCA run finished properly as indicated by the respective log file\n
    2. the number of principal components of the existing PCA matches the requested number n_components

    Parameters
    ----------
    n_components : int
        Number of principal components expected in the result files.
    result_prefix : pathlib.Path
        File path prefix pointing to the output of the smartpca run to check.

    Returns
    -------
    bool
        Whether the respective smartpca run finished properly.
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


def _smartpca_fst_finished(result_prefix: pathlib.Path) -> bool:
    """Checks whether existing smartpca FST distance matrix computation results are correct.

    We consider them to be correct if the smartPCA run finished properly as indicated by the respective log file.

    Parameters
    ----------
    result_prefix : pathlib.Path
        File path prefix pointing to the output of the smartpca FST computation to check.

    Returns
    -------
    bool
        Whether the respective smartpca run finished properly.
    """
    fst_file = pathlib.Path(f"{result_prefix}.fst")
    smartpca_log = pathlib.Path(f"{result_prefix}.smartpca.log")

    if not fst_file.exists() or not smartpca_log.exists():
        return False

    # 1. check if the run finished properly, that is indicated by a line "##end of smartpca:"
    # in the smartpca_log file

    run_finished = any(["##end of smartpca:" in line for line in smartpca_log.open()])
    if not run_finished:
        return False


def _get_embedding_populations(
    embedding_populations: Optional[pathlib.Path],
) -> pd.Series:
    """Return a pandas series of populations to use for the embedding computation.

    Parameters
    ----------
    embedding_populations : pathlib.Path
        Filepath containing the newline-separated list of populations.

    Returns
    -------
    pd.Series
        Pandas series of populations to use for the embedding computation.
    """
    if embedding_populations is None:
        return pd.Series(dtype=object)
    return pd.Series([pop.strip() for pop in embedding_populations.open()])


def _deduplicate_snp_id(snp_id: str, seen_snps: Set[str]):
    deduplicate = snp_id

    snp_id_counter = 0

    while deduplicate in seen_snps:
        snp_id_counter += 1
        deduplicate = f"{snp_id}_r{snp_id_counter}"

    return deduplicate


def _safe_cast(data: npt.NDArray, dtype: np.dtype) -> npt.NDArray:
    """Helper method to make sure that the numpy dtype casts do not yield unexpected values."""
    cast_data = data.astype(dtype)
    diffs = cast_data != data
    if np.any(diffs) and not np.isnan(data[diffs]).all():
        raise PandoraException(
            f"Could not safely cast to {dtype}. Incompatible values: {data[diffs]}."
        )
    return cast_data


def _process_input_data(
    input_data: npt.NDArray,
    missing_value: Union[float, int],
    dtype: np.dtype,
):
    # Depending on the dtype, we need to have a different nan value representation.
    # For float types we can use np.nan
    # For signed int types we use -1
    # For unsigned int types we use the respective max value
    # Note that we only allow integer and floating types
    if np.issubdtype(dtype, np.floating):
        nan_value = np.nan
    elif np.issubdtype(dtype, np.unsignedinteger):
        nan_value = np.iinfo(dtype).max
    elif np.issubdtype(dtype, np.signedinteger):
        # signed int
        nan_value = -1
    else:
        raise PandoraException(
            f"Unsupported numpy dtype passed: {dtype}. Only int and float dtypes are supported."
        )

    input_data = _safe_cast(input_data, np.float64)
    if np.isnan(missing_value):
        input_data[np.isnan(input_data)] = nan_value
    else:
        input_data[input_data == missing_value] = nan_value
    input_data = _safe_cast(input_data, dtype)

    return input_data, nan_value


class EigenDataset:
    """Class structure to represent a population genetics dataset in Eigenstrat format.

    This class provides methods to perform PCA and MDS analyses using the Eigensoft smartpca tool.
    It further provides methods to generate a bootstrap replicate dataset (SNPs resampled with replacement) and
    to generate overlapping sliding windows of sub-datasets.
    Note that in order for the bootstrap and windowing methods to work, the respective geno, ind, and snp files need to
    be in EIGENSTRAT format with a similar file prefix and need to have file endings ``.geno``, ``.ind``, and ``.snp``.

    Note that Pandora does not check on init whether all input files are present, as you are allowed to init
    a dataset despite the files missing. This is useful for saving storage when drawing lots of bootstrap
    replicates. In case the input files are missing, you need to pass sample_ids, populations and n_snps.
    Of course, you need the files if you want to run ``self.bootstrap``, ``self.run_pca``,
    ``self.run_mds``, or ``self.get_windows``.

    Parameters
    ----------
    file_prefix : pathlib.Path
        File path prefix pointing to the ind, geno, and snp files in EIGENSTRAT format.
        All methods assume that all three files have the same prefix and have the file endings
        ``.geno``, ``.ind``, and ``.snp``.
    embedding_populations : pathlib.Path, default=None
        Path pointing to a file containing a new-line separated list
        containing population names. Only samples belonging to these populations will be used for PCA analyses.
        If not set, all samples will be used.
    sample_ids :  pd.Series, default=None
        pandas Series containing a sample ID for each sequence in the dataset.
        If not set, Pandora will set this based on the content of ``self._ind_file``
    populations : pd.Series, default=None
        pandas Series containing the population for each sample in the dataset.
        If not set, Pandora will set this based on the content of ``self._ind_file``
    n_snps : int, default=None
        Number of SNPs in the dataset. If not set, Pandora will set this based on the content of ``self._geno_file``

    Attributes
    ----------
    file_prefix : pathlib.Path
        File path prefix pointing to the ind, geno, and snp files in EIGENSTRAT format.
        All methods assume that all three files have the same prefix and have the file endings
        ``.geno``, ``.ind``, and ``.snp``.
    name : str
        Name of the dataset. Inferred as name of the provided ``file_prefix``.
    embedding_populations : List[str]
        List of population used for PCA analysis with smartpca.
    sample_ids : pd.Series
        Pandas series containing the sample IDs for the dataset.
    populations : pd.Series
        Pandas series containing the populations for all samples in the dataset.
    projected_samples : pd.DataFrame
        Subset of self.sample_ids, contains only sample_ids that do not belong to
        embedding_populations.
    n_snps : int
        Number of SNPs in the dataset.
    pca : PCA
        PCA object as result of a smartpca run on the provided dataset.
        This is ``None`` until ``self.run_pca()`` was called.
    mds : MDS
        MDS object as a result of an MDS computation. This is ``None`` until ``self.run_mds()`` was called.

    Raises
    ------
    PandoraException
        If not all input files (``.geno``, ``.ind``, and ``.snp``) exist, but ``sample_ids``, ``populations``
        and/or ``n_snp`s` is ``None``.
    """

    def __init__(
        self,
        file_prefix: pathlib.Path,
        embedding_populations: Optional[pathlib.Path] = None,
        sample_ids: pd.Series = None,
        populations: pd.Series = None,
        n_snps: int = None,
    ):
        self.file_prefix: pathlib.Path = file_prefix
        self._file_dir: pathlib.Path = self.file_prefix.parent
        self.name: str = self.file_prefix.name

        self._ind_file: pathlib.Path = pathlib.Path(f"{self.file_prefix}.ind")
        self._geno_file: pathlib.Path = pathlib.Path(f"{self.file_prefix}.geno")
        self._snp_file: pathlib.Path = pathlib.Path(f"{self.file_prefix}.snp")

        self._embedding_populations_file: pathlib.Path = embedding_populations
        self.embedding_populations: pd.Series = _get_embedding_populations(
            self._embedding_populations_file
        )

        if not self.files_exist() and any(
            [sample_ids is None, populations is None, n_snps is None]
        ):
            # in case of missing input files, sample_ids, populations, and n_snps need to be set
            # otherwise the following initialization will fail due to missing files
            raise PandoraException(
                "Not all input files (.ind, .geno, .snp) for file_prefix present, "
                "but sample_ids, populations, and/or n_snps is None."
            )

        if sample_ids is None:
            self.sample_ids = self.get_sample_info()
        else:
            self.sample_ids = sample_ids

        if populations is None:
            self.populations = self.get_population_info()
        else:
            self.populations = populations

        if self.sample_ids.shape[0] != self.populations.shape[0]:
            raise PandoraException(
                f"Number of sample IDs and populations need to be identical. "
                f"Got {self.sample_ids.shape[0]} sample IDs, but {self.populations.shape[0]} populations."
            )

        self.projected_samples = self.get_projected_samples()

        if n_snps is None:
            self.n_snps = self.get_sequence_length()
        else:
            self.n_snps = n_snps

        self.pca: Optional[PCA] = None
        self.mds: Optional[MDS] = None

    def get_sample_info(self) -> pd.Series:
        """Reads the sample IDs from ``self._ind_file``.

        Returns
        -------
        pd.Series
            Pandas series containing the sample IDs of the dataset in the order in the ind file.

        Raises
        ------
        PandoraException
            if the respective ind file does not exist
        """
        if not self._ind_file.exists():
            raise PandoraConfigException(
                f"The .ind file {self._ind_file} does not exist."
            )

        sample_ids = []
        for sample in self._ind_file.open():
            sample_id, _, _ = sample.split()
            sample_ids.append(sample_id.strip())

        return pd.Series(sample_ids)

    def get_population_info(self) -> pd.Series:
        """Reads the populations from ``self._ind_file``.

        Returns
        -------
        pd.Series
            Pandas series containing the population for each sample in the order in the ind file.

        Raises
        ------
        PandoraException
            if the respective .ind file does not exist
        """
        if not self._ind_file.exists():
            raise PandoraConfigException(
                f"The .ind file {self._ind_file} does not exist."
            )

        populations = []
        for sample in self._ind_file.open():
            _, _, population = sample.split()
            populations.append(population.strip())

        return pd.Series(populations)

    def get_projected_samples(self) -> pd.Series:
        """Returns a pandas series with sample IDs of projected samples.

        If a sample is projected or used to compute the
        embedding is decided based on the presence and content of ``self._embedding_populations_file``.

        Returns
        -------
        pd.Series
            Pandas series containing only sample IDs of projected samples.
        """
        populations_for_embedding = _get_embedding_populations(
            self._embedding_populations_file
        )

        if populations_for_embedding.empty:
            # no samples are projected
            return pd.Series(dtype=object)

        projected = []
        for sample_id, population in zip(self.sample_ids, self.populations):
            if population not in populations_for_embedding.values:
                projected.append(sample_id)

        return pd.Series(projected)

    def get_sequence_length(self) -> int:
        """Counts and returns the number of SNPs in ``self._geno_file``.

        Returns
        -------
        int
            Number of SNPs in ``self._geno_file``.

        Raises
        ------
        PandoraException
            if the respective geno file does not exist
        """
        if not self._geno_file.exists():
            raise PandoraConfigException(
                f"The .geno file {self._geno_file} does not exist."
            )

        return sum(1 for _ in self._geno_file.open(mode="rb"))

    def files_exist(self) -> bool:
        """Checks whether all required input files (geno, snp, ind) exist.

        Returns
        -------
        bool
            Whether all three input files are present.
        """
        return all(
            [self._ind_file.exists(), self._geno_file.exists(), self._snp_file.exists()]
        )

    def check_files(self) -> None:
        """Checks whether all input files (geno, snp, ind) are in correct format according to the EIGENSOFT
        specification.

        Returns
        -------
        None

        Raises
        ------
        PandoraException
            If any of the three files is malformatted.
        """
        _check_ind_file(self._ind_file)
        _check_geno_file(self._geno_file)
        _check_snp_file(self._snp_file)

    def remove_input_files(self) -> None:
        """Removes all three input files (``self._ind_file``, ``self._geno_file``, ``self._snp_file``).

        This is useful if you want to save storage space and don't need the input files anymore
        (e.g. for bootstrap replicates).

        Returns
        -------
        None
        """
        self._ind_file.unlink(missing_ok=True)
        self._geno_file.unlink(missing_ok=True)
        self._snp_file.unlink(missing_ok=True)

    def run_pca(
        self,
        smartpca: Executable = "smartpca",
        n_components: int = 10,
        result_dir: Optional[pathlib.Path] = None,
        redo: bool = False,
        smartpca_optional_settings: Optional[Dict] = None,
    ) -> None:
        """Runs the EIGENSOFT smartpca on the dataset and assigns its PCA result to self.pca.

        Additional smartpca options can be passed as dictionary in smartpca_optional_settings, e.g.
        ``smartpca_optional_settings = dict(numoutlieriter=0, shrinkmode=True)``.

        Parameters
        ----------
        smartpca : Executable, default="smartpca"
            Path pointing to an executable of the EIGENSOFT smartpca tool.
            Default is 'smartpca'. This will only work if smartpca is installed systemwide.
        n_components : int, default=10
            Number of principal components to output.
        result_dir : pathlib.Path, default=self._file_dir
            File path pointing where to write the results to.
        redo : bool, default=False
            Whether to redo the analysis, if all outfiles are already present and correct.
        smartpca_optional_settings : Dict, default=None
            Additional smartpca settings.
            Not allowed are the following options: ``genotypename``, ``snpname``, ``indivname``,
            ``evecoutname``, ``evaloutname``, ``numoutevec``, ``maxpops``.
            If not set, the default settings of your smartpca executable are used.

        Returns
        -------
        None

        Raises
        ------
        PandoraException
            - If the number of principal components is >= the number of SNPs in ``self.input_data``.
            - If not all input files ein EIGENSTRAT format are present (geno, snp, ind files).
        RuntimeError
            If the ``smartpca`` run failed.
        """
        if n_components > self.n_snps:
            raise PandoraException(
                "Number of Principal Components needs to be smaller or equal than the number of SNPs. "
                f"Got {self.n_snps} SNPs, but asked for {n_components}."
            )

        if result_dir is None:
            result_dir = self._file_dir

        evec_out = result_dir / (self.name + ".evec")
        eval_out = result_dir / (self.name + ".eval")
        smartpca_log = result_dir / (self.name + ".smartpca.log")

        # check whether the all required output files are already present
        # and whether the smartPCA run finished properly and the number of PCs matches the requested number of PCs

        if _smartpca_finished(n_components, result_dir / self.name) and not redo:
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
                genotypename: {self._geno_file}
                snpname: {self._snp_file}
                indivname: {self._ind_file}
                evecoutname: {evec_out}
                evaloutname: {eval_out}
                numoutevec: {n_components}
                maxpops: {self.populations.unique().shape[0]}
                """

            smartpca_params = textwrap.dedent(smartpca_params)

            if smartpca_optional_settings is not None:
                for k, v in smartpca_optional_settings.items():
                    if isinstance(v, bool):
                        v = "YES" if v else "NO"
                    smartpca_params += f"{k}: {v}\n"

            if self._embedding_populations_file is not None:
                smartpca_params += "lsqproject: YES\n"
                smartpca_params += f"poplistname: {self._embedding_populations_file}\n"

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
                    # in case of an error, we remove the (probably) incorrect/broken evec and eval files
                    evec_out.unlink(missing_ok=True)
                    eval_out.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Error running smartPCA. "
                        f"Check the smartPCA logfile {smartpca_log.absolute()} for details."
                    )

            self.pca = from_smartpca(evec_out, eval_out)

    def fst_population_distance(
        self, smartpca: Executable, result_prefix: pathlib.Path, redo: bool = False
    ) -> Tuple[npt.NDArray, pd.Series]:
        """Computes the FST genetic distance matrix using EIGENSOFT's smartpca tool.

        The resulting FST matrix will be stored in fst_file and returned as numpy array.

        Parameters
        ----------
        smartpca : Executable
            Path pointing to an executable of the EIGENSOFT smartpca tool.
        result_prefix : pathlib.Path
            Prefix where to store the results of the smartpca FST computation. On successfull execution, two files will
            be created: the FST result (``{prefix}.fst``) and a smartpca log file (``{prefix}.smartpca.log``).
        redo : bool
            Whether to recompute the FST matrix in case the result file is already present.

        Returns
        -------
        distance_matrix : npt.NDArray
            Distance matrix of pairwise FST distances between all unique populations.
            The shape of this matrix is ``(n_unique_populations, n_unique_populations)``.
        populations : pd.Series
            Pandas Series containing a population name for each row in the distance matrix. This values of this series
            are the unique populations.

        Raises
        ------
        RuntimeError
            If the ``smartpca`` FST distance matrix computation failed.
        """
        fst_file = pathlib.Path(f"{result_prefix}.fst")
        smartpca_log = pathlib.Path(f"{result_prefix}.smartpca.log")

        if _smartpca_fst_finished(result_prefix) and not redo:
            return _parse_smartpca_fst_results(result_prefix)

        result_prefix.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(mode="w") as tmpfile:
            smartpca_fst_params = f"""
                genotypename: {self._geno_file}
                snpname: {self._snp_file}
                indivname: {self._ind_file}
                phylipoutname: {fst_file}
                fstonly: YES
                maxpops: {self.populations.unique().shape[0]}
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
                    # in case of an error, we remove the (probably) incorrect/broken fst file
                    fst_file.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Error computing FST matrix using smartPCA. "
                        f"Check the smartPCA logfile {smartpca_log.absolute()} for details."
                    )

            return _parse_smartpca_fst_results(result_prefix)

    def run_mds(
        self,
        smartpca: Executable,
        n_components: int = 2,
        result_dir: Optional[pathlib.Path] = None,
        redo: bool = False,
    ) -> None:
        """Performs MDS analysis using the data provided in this class.

        The FST matrix is generated using the EIGENSOFT smartpca tool.
        The subsequent MDS analysis is performed using the scikit-learn MDS implementation.

        Please note that since EIGENSOFT is used for the FST computation, all samples with the population set to
        ``Ignore`` will not be used for the MDS computation.

        Parameters
        ----------
        smartpca : Executable
            Path pointing to an executable of the EIGENSOFT smartpca tool.
        n_components : int, default=2
            Number of components to reduce the data to.
        result_dir : pathlib.Path, default=self._file_dir
            Directory where to store the data in. Calling this method will create two files:\n
            * ``result_dir / (self.name + ".fst")``: contains the FST distance matrix.\n
            * ``result_dir / (self.name + ".smartpca.log")``: contains the smartpca log.\n
            Default is ``self._file_dir``.
        redo : bool, default=False
            Whether to recompute the FST matrix in case the result file is already present.

        Returns
        -------
        None

        Raises
        ------
        PandoraException
            If the number of components is >= the number of SNPs in ``self.input_data``.
        RuntimeError
            If the ``smartpca`` FST distance matrix computation failed.
        """
        if n_components > self.n_snps:
            raise PandoraException(
                "Number of components needs to be smaller or equal than the number of SNPs. "
                f"Got {self.n_snps} SNPs, but asked for {n_components}."
            )

        if result_dir is None:
            result_dir = self._file_dir

        fst_matrix, populations = self.fst_population_distance(
            smartpca=smartpca, result_prefix=result_dir / self.name, redo=redo
        )

        mds = sklearnMDS(
            n_components=n_components,
            dissimilarity="precomputed",
            normalized_stress=False,
        )
        embedding = mds.fit_transform(fst_matrix)
        embedding = pd.DataFrame(
            data=embedding, columns=[f"D{i}" for i in range(n_components)]
        )
        embedding["population"] = populations

        # since we are using EIGENSOFT to compute the FST distance matrix, we remove all sample IDs with poopulation
        # set to ``Ignore`` as they are not present in the distance matrix
        sample_ids, populations = zip(
            *[
                (s, p)
                for s, p in zip(self.sample_ids, self.populations)
                if p.lower() != "ignore"
            ]
        )

        self.mds = from_sklearn_mds(
            embedding, pd.Series(sample_ids), pd.Series(populations), mds.stress_
        )

    def bootstrap(
        self, bootstrap_prefix: pathlib.Path, seed: int, redo: bool = False
    ) -> EigenDataset:
        """Creates a bootstrap dataset based on the content of self.

        Bootstraps the dataset by resampling SNPs with replacement.

        Parameters
        ----------
        bootstrap_prefix : pathlib.Path
            Prefix of the file path where to write the bootstrap dataset to.
                The resulting files will be  ``bootstrap_prefix.geno``, ``bootstrap_prefix.ind``, and ``bootstrap_prefix.snp``.
        seed : int
            Seed to initialize the random number generator before drawing the replicates.
        redo : bool, default=False
            Whether to redo the bootstrap if all output files are present.

        Returns
        -------
        EigenDataset
            A new dataset object containing the bootstrap replicate data.
        """
        bs_ind_file = pathlib.Path(f"{bootstrap_prefix}.ind")
        bs_geno_file = pathlib.Path(f"{bootstrap_prefix}.geno")
        bs_snp_file = pathlib.Path(f"{bootstrap_prefix}.snp")

        files_exist = all(
            [bs_ind_file.exists(), bs_geno_file.exists(), bs_snp_file.exists()]
        )

        if files_exist and not redo:
            return EigenDataset(
                bootstrap_prefix,
                self._embedding_populations_file,
                self.sample_ids,
                self.populations,
                self.n_snps,
            )

        bs_ind_file.unlink(missing_ok=True)
        bs_geno_file.unlink(missing_ok=True)
        bs_snp_file.unlink(missing_ok=True)

        # sample the SNPs using the snp file
        # each line in the snp file corresponds to one SNP
        num_snps = self.get_sequence_length()

        # create the parent directory to make sure we can write the files
        bootstrap_prefix.parent.mkdir(parents=True, exist_ok=True)

        # check if a checkpoint with SNP indices exists
        ckp_file = pathlib.Path(f"{bootstrap_prefix}.ckp")
        if ckp_file.exists() and not redo:
            seed, indices = ckp_file.open().readline().split(";")
            random.seed(int(seed))
            bootstrap_snp_indices = [int(v) for v in indices.split(",")]
        else:
            random.seed(seed)
            bootstrap_snp_indices = random.choices(range(num_snps), k=num_snps)

            # store random seed and selected SNP indices as checkpoint for reproducibility
            with ckp_file.open("w") as f:
                f.write(
                    str(seed) + ";" + ", ".join([str(i) for i in bootstrap_snp_indices])
                )

        # 1. Bootstrap the .snp file
        snps = self._snp_file.open().readlines()
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
        genos = self._geno_file.open(mode="rb").readlines()
        with bs_geno_file.open(mode="ab") as f:
            for bootstrap_idx in bootstrap_snp_indices:
                geno_line = genos[bootstrap_idx].strip()
                f.write(geno_line)
                f.write(b"\n")

        # when bootstrapping on SNP level, the .ind file does not change
        shutil.copy(self._ind_file, bs_ind_file)
        return EigenDataset(
            bootstrap_prefix,
            self._embedding_populations_file,
            self.sample_ids,
            self.populations,
            self.n_snps,
        )

    def _generate_windowed_dataset(
        self, window_start: int, window_end: int, result_prefix: pathlib.Path
    ) -> EigenDataset:
        """Extracts a section of SNPs from self and returns it as new ``EigenDataset``.

        Parameters
        ----------
        window_start : int
            Start index of the section to extract.
        window_end : int
            End index of the section to extract.
        result_prefix : pathlib.Path
            File prefix where to store the resulting EIGENSTRAT files.
            Will create three new files: ``{result_prefix}.ind``, ``{result_prefix}.geno``, ``{result_prefix}.snp`` containing
            the specified section of SNPs.

        Returns
        -------
        EigenDataset
            ``EigenDataset`` object containing the section of SNPs requested.
        """
        ind_file = pathlib.Path(f"{result_prefix}.ind")
        geno_file = pathlib.Path(f"{result_prefix}.geno")
        snp_file = pathlib.Path(f"{result_prefix}.snp")

        # .ind file does not change
        shutil.copy(self._ind_file, ind_file)

        # .snp file needs to be filtered
        snps = self._snp_file.open().readlines()
        window_snps = snps[window_start:window_end]
        with snp_file.open(mode="w") as f:
            f.write("".join(window_snps))

        # .geno file needs to be filtered as well
        genos = self._geno_file.open(mode="rb").readlines()
        window_genos = genos[window_start:window_end]
        with geno_file.open(mode="wb") as f:
            f.write(b"".join(window_genos))

        return EigenDataset(
            result_prefix,
            self._embedding_populations_file,
            self.sample_ids,
            self.populations,
        )

    def get_windows(
        self, result_dir: pathlib.Path, n_windows: int = 100
    ) -> List[EigenDataset]:
        """Creates n_windows new ``EigenDataset`` objects as overlapping sliding windows over self.

        Let :math:`K` = number of SNPs in self and :math:`N` = n_windows.
        Each dataset will have a window size of :math:`int(K / N + (K / 2 * N))` SNPs.
        The stride is :math:`int(K / N)` and the overlap between windows is thus :math:`int(K / 2 * N)` SNPs.
        Note that the last EigenDataset will contain fewer SNPs as there is no following window to overlap with.
        However, due to rounding, the number of SNPs in the final Dataset will not simply be overlap fewer.

        Parameters
        ----------
        result_dir : pathlib.Path
            Directory where to store the resulting Dataset files in.
            Each window Dataset will be named ``window_{i}`` for ``i`` in ``range(n_windows)``
        n_windows : int, default=100
            Number of windowed datasets to generate.

        Returns
        -------
        windows : List[EigenDataset]
            List of ``n_windows`` new ``EigenDataset`` objects as overlapping windows over self.
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


class NumpyDataset:
    """Class structure to represent a population genetics dataset in numerical format.

    This class provides methods to perform PCA and MDS analyses on the provided numerical data using scikit-learn.
    It further provides methods to generate a bootstrap replicate dataset (SNPs resampled with replacement) and
    to generate overlapping sliding windows of sub-datasets.

    Parameters
    ----------
    input_data : npt.NDArray
        Numpy Array containing the input data to use.
    sample_ids : pd.Series[str]
        Pandas Series containing the sample IDs of the sequences contained in ``input_data``.
        Expects the number of ``sample_ids`` to match the first dimension of ``input_data``.
    populations : pd.Series[str]
        Pandas Series containing the populations of the sequences contained in ``input_data``.
        Expects the number of ``populations`` to match the first dimension of ``input_data``.
    missing_value : Union[float, int], default=np.nan
        Value to treat as missing value. All missing values in ``input_data`` will be replaced with a special value
        depending on the specified dtype:
        For floating point types: np.nan
        For signed integer types: -1
        For unsigned integer types: the highest possible integer for the respective type
    dtype: np.dtype, default=np.uint8
        Numpy datatype to use to represent the input data. The more complex the data type, the higher it's memory footprint.
        The default is the datatype with the smallest possible memory footprint: np.uint8 (unsigned 8Bit integer).
        For most genotype data with only ``0``, ``1``, ``2``, and missing as genotype calls this is the best option.

    Attributes
    ----------
    input_data : npt.NDArray
        Numpy Array containing the input data to use.
    sample_ids : pd.Series[str]
        Pandas Series containing a sample ID for each row in ``input_data``.
    populations : pd.Series[str]
        Pandas Series containing a population name for each row in ``input_data``.
    pca : PCA
        PCA object as result of a PCA analysis run on the provided dataset.
        This is ``None`` until ``self.run_pca()`` was called.
    mds : MDS
        MDS object as a result of an MDS computation. This is ``None`` until ``self.run_mds()`` was called.

    Raises
    ------
    PandoraException
        If the number of samples and number of populations differ (need to provide one population per sample).
    """

    def __init__(
        self,
        input_data: npt.NDArray,
        sample_ids: pd.Series,
        populations: pd.Series,
        missing_value: Union[float, int] = np.nan,
        dtype: np.dtype = np.uint8,
    ):
        if sample_ids.shape[0] != populations.shape[0]:
            raise PandoraException(
                f"Provide a population for each sample. "
                f"Got {sample_ids.shape[0]} sample_ids but {populations.shape[0]} populations."
            )
        if sample_ids.shape[0] != input_data.shape[0]:
            raise PandoraException(
                f"Provide a sample ID for each sample in input_data. "
                f"Got {sample_ids.shape[0]} sample_ids but input_data has shape {input_data.shape}"
            )

        self.input_data, self._missing_value = _process_input_data(
            input_data, missing_value, dtype
        )
        self.sample_ids = sample_ids
        self.populations = populations

        self.pca: Optional[PCA] = None
        self.mds: Optional[MDS] = None

    def run_pca(
        self,
        n_components: int = 10,
        imputation: Optional[str] = "mean",
    ) -> None:
        """Performs PCA analysis on ``self.input_data`` reducing the data to ``n_components`` dimensions.

        Uses the scikit-learn PCA implementation. The result of the PCA analysis is a PCA object assigned to ``self.pca``.

        Parameters
        ----------
        n_components : int, default=10
            Number of components to reduce the data to. Default is 10.
        imputation : Optional[str], default="mean"
            Imputation method to use. Available options for PCA are:\n
            - mean: Imputes missing values with the average of the respective SNP\n
            - remove: Removes all SNPs with at least one missing value.
            - None: Does not impute missing data. Note that this option is only valid if ``self.input_data`` does not contain NaN values.

        Returns
        -------
        None

        Raises
        ------
        PandoraException
            - If the number of principal components is >= the number of SNPs in ``self.input_data``.
            - If imputation is ``None`` but ``self.input_data`` contains NaN values.
        """
        n_snps = self.input_data.shape[1]
        if n_components > n_snps:
            raise PandoraException(
                "Number of Principal Components needs to be smaller or equal than the number of SNPs. "
                f"Got {n_snps} SNPs, but asked for {n_components}."
            )

        has_missing_values = (
            np.any(np.isnan(self.input_data))
            if np.isnan(self._missing_value)
            else np.any(self.input_data == self._missing_value)
        )

        if imputation is None and has_missing_values:
            raise PandoraException(
                "Imputation method cannot be None if self.input_data contains missing data."
            )

        # For the imputation, we first transform the input data into a float numpy array and set the
        # missing value to np.nan
        input_data = self.input_data.astype("float")
        # this will work for every value other than np.nan, but no need to modify the data in that case anyway
        input_data[input_data == self._missing_value] = np.nan
        pca_input = impute_data(input_data, imputation)

        pca = sklearnPCA(n_components)
        embedding = pca.fit_transform(pca_input)
        embedding = pd.DataFrame(
            data=embedding, columns=[f"D{i}" for i in range(n_components)]
        )
        embedding["sample_id"] = self.sample_ids
        embedding["population"] = self.populations
        self.pca = PCA(embedding, n_components, pca.explained_variance_ratio_)

    def run_mds(
        self,
        n_components: int = 2,
        distance_metric: Callable[
            [npt.NDArray, pd.Series, str], Tuple[npt.NDArray, pd.Series]
        ] = euclidean_sample_distance,
        imputation: Optional[str] = "mean",
    ) -> None:
        """Performs MDS analysis using the data provided in this class.

        The distance matrix is generated using the
        provided distance_metric callable. The subsequent MDS analysis is performed using the scikit-learn MDS
        implementation. The result of the MDS analysis is an MDS object assigned to ``self.mds``.

        Parameters
        ----------
        n_components : int, default=2
            Number of components to reduce the data to.
        distance_metric : Callable[[npt.NDArray, pd.Series, str], Tuple[npt.NDArray, pd.Series]], default=euclidean_sample_distance
            Distance metric to use for computing the distance matrix input for MDS. This is expected to be a
            function that receives the numpy array of sequences, the population for each sequence, and the imputation method
            as input and should output the distance matrix and the respective populations for each row.
            The resulting distance matrix is of size :math:`(n, m)` and the resulting populations is expected to be
            of size :math:`(n, 1)`.
            Default is ``distance_metrics::eculidean_sample_distance`` (the pairwise Euclidean distance of all samples).
        imputation : Optional[str], default="mean"
            Imputation method to use. Available options are:\n
            - mean: Imputes missing values with the average of the respective SNP\n
            - remove: Removes all SNPs with at least one missing value.\n
            - None: Does not impute missing data.
            Note that depending on the ``distance_metric``, not all imputation methods are supported. See the respective
            documentations in the ``distance_metrics`` module.

        Returns
        -------
        None

        Raises
        ------
        PandoraException
            - If the ``distance_metric`` did not return one population per row in the returned distance matrix.
            - If the number of components is >= the number of rows in the distance matrix.
        """
        # Before applying the distance metric, transform the input data to a floating point array
        # and set the missing value to nan
        input_data = self.input_data.astype("float")
        input_data[input_data == self._missing_value] = np.nan

        distance_matrix, populations = distance_metric(
            input_data, self.populations, imputation
        )
        if distance_matrix.shape[0] != populations.shape[0]:
            raise PandoraException(
                "The distance matrix computation did not yield the expected array and series shapes: "
                "The number of populations needs to be identical to the number of rows in the distance matrix. "
                f"Got {populations.shape[0]} populations but {distance_matrix.shape[0]} rows in the distance matrix."
            )

        n_dims = distance_matrix.shape[1]
        if n_components > n_dims:
            raise PandoraException(
                "Number of components needs to be smaller or equal than the number of dimensions. "
                f"Got {n_dims} dimensions in the distance matrix, but asked for {n_components}."
            )

        mds = sklearnMDS(
            n_components, dissimilarity="precomputed", normalized_stress=False
        )
        embedding = mds.fit_transform(distance_matrix)
        embedding = pd.DataFrame(
            data=embedding, columns=[f"D{i}" for i in range(n_components)]
        )
        embedding["population"] = populations

        self.mds = from_sklearn_mds(
            embedding, self.sample_ids, self.populations, mds.stress_
        )

    def bootstrap(self, seed: Optional[int] = None) -> NumpyDataset:
        """Creates a bootstrap dataset based on the content of self.

        Bootstraps the dataset by resampling SNPs with replacement.

        Parameters
        ----------
        seed : Optional[int], default=None
            Optional seed to initialize the random number generator before drawing the replicates.

        Returns
        -------
        NumpyDataset
            A new dataset object containing the bootstrapped ``input_data``.
        """
        if seed is not None:
            random.seed(seed)
        num_snps = self.input_data.shape[1]
        bootstrap_data = self.input_data[
            :, np.random.choice(range(num_snps), size=num_snps)
        ]
        return NumpyDataset(bootstrap_data, self.sample_ids, self.populations)

    def get_windows(self, n_windows: int = 100) -> List[NumpyDataset]:
        """Creates ``n_windows`` new ``NumpyDataset`` objects as overlapping sliding windows over self.

        Let :math:`K` = number of SNPs in self and :math:`N` = n_windows.
        Each dataset will have a window size of :math:`int(K / N + (K / 2 * N))` SNPs.
        The stride is :math:`int(K / N)` and the overlap between windows is thus :math:`int(K / 2 * N)` SNPs.
        Note that the last NumpyDataset will contain fewer SNPs as there is no following window to overlap with.
        However, due to rounding, the number of SNPs in the final Dataset will not simply be overlap fewer.

        Parameters
        ----------
        n_windows : int, default=100
            Number of windowed datasets to generate.

        Returns
        -------
        windows : List[NumpyDataset]
            List of ``n_windows`` new ``NumpyDataset`` objects as overlapping windows over self.
        """
        num_snps = self.input_data.shape[1]
        overlap = int((num_snps / n_windows) / 2)
        stride = int(num_snps / n_windows)
        window_size = overlap + stride

        windows = []

        for i in range(n_windows):
            window_start = i * stride
            window_end = i * stride + window_size
            window_data = self.input_data[:, window_start:window_end]
            windows.append(NumpyDataset(window_data, self.sample_ids, self.populations))

        return windows


def numpy_dataset_from_eigenfiles(eigen_prefix: pathlib.Path) -> NumpyDataset:
    """Loads a genotype dataset in EIGENSTRAT format as ``NumpyDataset``.

    This method only requires the genotype and individual files as the metadata of the SNPs present in the ``.snp`` file is
    not used. Note that the dataset needs to be in EIGENSTRAT format with a similar file prefix and need to have file
    endings ``.geno`` and ``.ind`` for the respective file type.

    Parameters
    ----------
    eigen_prefix : pathlib.Path
        File path prefix pointing to the ind and geno genotype files in EIGENSTRAT format.
        This method assumes that all EIGEn files have the same prefix and have the file endings
        ``.geno`` and ``.ind``. (Note that the ``.snp`` file is not required.)

    Returns
    -------
    NumpyDataset
        ``NumpyDataset`` containing the genotype data provided in the EIGEN files located at ``eigen_prefix``.

    Raises
    ------
    PandoraException
        If not all input files (``.geno``, ``.ind``, and ``.snp``) for the given ``eigen_prefix`` exist.
    """
    ind_file = pathlib.Path(f"{eigen_prefix}.ind")
    geno_file = pathlib.Path(f"{eigen_prefix}.geno")

    if not ind_file.exists() or not geno_file.exists():
        raise PandoraException(
            f"Not all required input files (.ind, .geno) for eigen_prefix {eigen_prefix} present."
        )

    _check_ind_file(ind_file)
    _check_geno_file(geno_file)

    geno_data = []

    for line in geno_file.open():
        geno_data.append([int(c) for c in line.strip()])

    geno_data = np.asarray(geno_data).T
    geno_data = geno_data.astype(float)
    geno_data[geno_data == 9] = np.nan

    sample_ids = []
    populations = []

    for line in ind_file.open():
        sid, _, pop = line.strip().split()
        sample_ids.append(sid.strip())
        populations.append(pop.strip())

    return NumpyDataset(geno_data, pd.Series(sample_ids), pd.Series(populations))
