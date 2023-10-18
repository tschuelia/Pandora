import dataclasses
import datetime
import logging
import multiprocessing
import pathlib
import textwrap
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from plotly import graph_objects as go
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt, ValidationError
from pydantic.dataclasses import dataclass

from pandora import __version__
from pandora.bootstrap import bootstrap_and_embed_multiple
from pandora.converter import run_convertf
from pandora.custom_errors import PandoraConfigException, PandoraException
from pandora.custom_types import (
    AnalysisMode,
    EmbeddingAlgorithm,
    Executable,
    FileFormat,
)
from pandora.dataset import EigenDataset
from pandora.embedding_comparison import BatchEmbeddingComparison
from pandora.logger import fmt_message, logger
from pandora.plotting import (
    plot_clusters,
    plot_populations,
    plot_projections,
    plot_support_values,
)
from pandora.sliding_window import sliding_window_embedding


@dataclass
class PandoraConfig(BaseModel):
    """Pydantic dataclass encapsulating the settings required to run Pandora.

    Parameters
    ----------
    dataset_prefix : pathlib.Path
        File path prefix pointing to the dataset to use for the Pandora analyses.
        Pandora will look for files called <input>.* so make sure all files have the same prefix.
    result_dir : pathlib.Path
        Directory where to store all (intermediate) results to.
    file_format : FileFormat, default=FileFormat.EIGENSTRAT
        Format of the input dataset.
        Can be ANCESTRYMAP, EIGENSTRAT, PED, PACKEDPED, PACKEDANCESTRYMAP. Default is EIGENSTRAT.
    convertf : Executable, default="convertf"
        File path pointing to an executable of Eigensoft's convertf tool. Convertf is used
        if the provided dataset is not in EIGENSTRAT format. Default is 'convertf'. This will only work if
        convertf is installed systemwide.
    n_replicates : PositiveInt, default=100
        Number of bootstrap replicates or sliding windows to compute.
        In case of bootstrapping, make sure to also set the `bootstrap_convergence_check` parameter as desired.
    keep_replicates : bool, default=False
        Whether to store all intermediate datasets files (.geno, .snp, .ind). Note that this will
        result in a substantial storage consumption. Default is False. Note that the bootstrapped indicies are
        stored as checkpoints for full reproducibility in any case.
    bootstrap_convergence_check : bool, default=True
        Whether to heuristically determine convergence of the bootstrapping procedure.
        If true, instead of computing `n_replicates` bootstraps and embeddings, Pandora will check for convergence once
        every 10 bootstrap embeddings are computed. If according to our heuristic (see `bootstrap.py` for more details)
        the bootstrap procedure converged, all remaining tasks are cancelled and the stability is determined uisng
        only the number of replicates computed when convergence is determined.
        Note that this parameter is only relevant if `analysis_mode` is `AnalysisMode.BOOTSTRAP`.
    n_components : PositiveInt, default=10
        Number of dimensions to output and compare for PCA and MDS analyses.
        The recommended number is 10 for PCA and 2 for MDS. Default is 10 in correspondance to the default PCA embedding.
    embedding_algorithm : EmbeddingAlgorithm, default=EmbeddingAlgorithm.PCA
        Embedding to compute during the stability analysis. Can be either EmbeddingAlgorithm.PCA or EmbeddingAlgorithm.MDS.
    smartpca : Executable, default="smartpca"
        File path pointing to an executable of Eigensoft's smartpca tool. Smartpca is used
        for PCA analyses on the provided dataset. Default is 'smartpca'. This will only work if smartpca is
        installed systemwide.
    smartpca_optional_settings : Dict[str, Any], default=None
        Optional additional settings to use when performing PCA with
        smartpca. Pandora has full support for all smartpca options. Not allowed are the following options:
        genotypename, snpname, indivname, evecoutname, evaloutname, numoutevec, maxpops.
        Use the following schema to set the options: dict(shrinkmode=True, numoutlieriter=1)
    embedding_populations : pathlib.Path, default=None
        File containing a new-line separated list of population names.
        Only these populations will be used for the dimensionality reduction. In case of PCA analyses, all remaining
        samples in the dataset will be projected onto the PCA results.
    support_value_rogue_cutoff : float, default=0.5
        When plotting the support values, only samples with a support value lower
        than the support_value_rogue_cutoff  will be annotated with their sample IDs.
        Note that all samples in the respective plot are color-coded according to their support value in any case.
    kmeans_k : PositiveInt, default=None
        Number of clusters k to use for K-Means clustering of the dimensionality reduction embeddings.
        If not set, the optimal number of clusters will be automatically determined according to the
        Bayesian Information Criterion (BIC).
    analysis_mode : AnalysisMode, default=AnalysisMode.BOOTSTRAP
        Whether Pandora should do bootstrap analysis or sliding-window analysis.
    redo : bool, default=False
        Whether to rerun all analyses in case the results files from a previous run are already present.
        Careful: this will overwrite existing results!
    seed : int, default=None
        Seed to initialize the random number generator. This setting is recommended for reproducible
        analyses. Default is the current unix timestamp.
    threads : NonNegativeInt, default=None
        Number of threads to use for the analysis. Default is the number of CPUs available.
    result_decimals : NonNegativeInt, default=2
        Number of decimals to round the stability scores and support values in the output.
        Default is two decimals.
    verbosity : int, default=1
        Verbosity of the output logging of Pandora.
        - 0 = quiet, prints only errors and the results (loglevel = ERROR)
        - 1 = verbose, prints all intermediate infos (loglevel = INFO)
        - 2 = debug, prints intermediate infos and debug messages (loglevel = DEBUG)
    plot_results : bool, default=False,
        Whether to plot all dimensionality reduction results and sample support values.
    plot_dim_x : NonNegativeInt, default=0
        Dimension to plot on the x-axis. Note that the dimensions are zero-indexed. To plot the first
        dimension set plot_dim_x = 0
    plot_dim_y : NonNegativeInt, default=1
        Dimension to plot on the y-axis. Note that the dimensions are zero-indexed. To plot the second
        dimension set plot_dim_y = 1
    """

    model_config = ConfigDict(extra="forbid")

    # EigenDataset related
    dataset_prefix: pathlib.Path
    result_dir: pathlib.Path
    file_format: FileFormat = FileFormat.EIGENSTRAT
    convertf: Executable = "convertf"

    # Repliacates related settings
    n_replicates: NonNegativeInt = 100
    keep_replicates: bool = False

    # Bootstrap specific setting (convergence check)
    bootstrap_convergence_check: bool = True

    # Embedding related
    n_components: NonNegativeInt = 10
    embedding_algorithm: EmbeddingAlgorithm = EmbeddingAlgorithm.PCA
    smartpca: Executable = "smartpca"
    smartpca_optional_settings: Optional[Dict[str, Any]] = None
    embedding_populations: Optional[pathlib.Path] = None

    # sample support values
    support_value_rogue_cutoff: float = 0.5

    # Cluster settings
    kmeans_k: Optional[int] = None

    # Pandora execution mode settings
    analysis_mode: AnalysisMode = AnalysisMode.BOOTSTRAP
    redo: bool = False
    seed: int = int(datetime.datetime.now().timestamp())
    threads: PositiveInt = multiprocessing.cpu_count()
    result_decimals: NonNegativeInt = 2
    verbosity: int = 1

    # Plot settings
    plot_results: bool = False
    plot_dim_x: NonNegativeInt = 0
    plot_dim_y: NonNegativeInt = 1

    def __post_init__(self):
        self.result_dir.mkdir(exist_ok=True, parents=True)

    @property
    def pandora_logfile(self) -> pathlib.Path:
        """Returns a path to the Pandora logfile where all results should be logged to.

        Returns
        -------
        pathlib.Path
            Filepath to the Pandora logfile.
        """
        return self.result_dir / "pandora.log"

    @property
    def configfile(self) -> pathlib.Path:
        """Returns a path to the pandora config yaml.

        self.save_config will save all PandoraConfig options in this file

        Returns
        -------
        pathlib.Path
            Filepath to the config file.
        """
        return self.result_dir / "pandora.yaml"

    @property
    def result_file(self) -> pathlib.Path:
        """Returns a path to the Pandora results file where all final stability results should we written to.

        Returns
        -------
        pathlib.Path
            Filepath to the Pandora results file.
        """
        return self.result_dir / "pandora.txt"

    @property
    def bootstrap_result_dir(self) -> pathlib.Path:
        """Path where to store all bootstrap (intermediate) results in.

        Returns
        -------
        pathlib.Path
            Filepath to the bootstrap results directory.
        """
        return self.result_dir / "bootstrap"

    @property
    def sliding_window_result_dir(self) -> pathlib.Path:
        """Path where to store all sliding-window (intermediate) results in.

        Returns
        -------
        pathlib.Path
            Filepath to the sliding-window results directory.
        """
        return self.result_dir / "windows"

    @property
    def convertf_result_dir(self) -> pathlib.Path:
        """Path where to store converted input files.

        Returns
        -------
        pathlib.Path
            Filepath to the converted input files directory.
        """
        return self.result_dir / "converted"

    @property
    def pairwise_stability_result_file(self) -> pathlib.Path:
        """Returns a path to a csv file where all pairwise stability results should be written to.

        Returns
        -------
        pathlib.Path
            Filepath to a csv file for pairwise stability results.
        """
        return self.result_dir / "pandora.replicates.csv"

    @property
    def sample_support_values_csv(self) -> pathlib.Path:
        """Returns a path to a csv file where all sample support values should be written to.

        Returns
        -------
        pathlib.Path
            Filepath to a csv file for support value results for all samples.
        """
        return self.result_dir / "pandora.supportValues.csv"

    @property
    def projected_sample_support_values_csv(self) -> pathlib.Path:
        """Returns a path to a csv file where all sample support values for projected samples should be written to.

        Returns
        -------
        pathlib.Path
            Filepath to a csv file for support value results for projected samples.
        """
        return self.result_dir / "pandora.supportValues.projected.csv"

    @property
    def plot_dir(self) -> pathlib.Path:
        """Path where to store all plots in.

        Returns
        -------
        pathlib.Path
            Filepath to the plots directory.
        """
        return self.result_dir / "plots"

    @property
    def loglevel(self) -> int:
        """Converts the int log-level to the respective logging module constant.

        Returns
        -------
        int
            logging module loglevel based on the verbosity specified in self.
        """
        if self.verbosity == 0:
            return logging.ERROR
        elif self.verbosity == 1:
            return logging.INFO
        elif self.verbosity == 2:
            return logging.DEBUG
        else:
            raise ValueError(
                f"verbosity needs to be 0 (ERROR), 1 (INFO), or 2 (DEBUG). Instead got value {self.verbosity}."
            )

    def get_configuration(self) -> Dict[str, Any]:
        """Creates a dictionary mapping of all settings in self.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of all settings in self. Filepaths are translated to absolute path strings,
            enums are represted by their value.
        """
        config = dataclasses.asdict(self)

        # pathlib Paths cannot be dumped in yaml directly
        # so we have to manually replace them with their string representation
        for k, v in config.items():
            if isinstance(v, pathlib.Path):
                config[k] = str(v.absolute())
            elif isinstance(v, Enum):
                config[k] = v.value

        return config

    def save_config(self) -> None:
        """Saves the configurations of self in yaml format in self.configfile.

        Will additionally log the Pandora version used for reproducibility.
        The resulting config file can be used as input for a subsequent Pandora execution.

        Returns
        -------
        None
        """
        config_yaml = yaml.safe_dump(self.get_configuration())
        # additionally save the Pandora version
        self.configfile.open(mode="w").write(f"# PANDORA VERSION {__version__}\n\n")
        self.configfile.open(mode="a").write(config_yaml)

    def log_results_files(self) -> None:
        """Logs the absolute file paths of all files written during an execution of Pandora.

        Returns
        -------
        None
        """
        logger.info(
            textwrap.dedent(
                """
                ------------------
                Result Files
                ------------------"""
            )
        )
        logger.info(f"> Pandora results: {self.result_file.absolute()}")
        logger.info(
            f"> Pairwise stabilities: {self.pairwise_stability_result_file.absolute()}"
        )
        logger.info(
            f"> Sample Support values: {self.sample_support_values_csv.absolute()}"
        )

        if self.embedding_populations is not None:
            logger.info(
                f"> Projected Sample Support values: {self.projected_sample_support_values_csv.absolute()}"
            )
        if self.plot_results:
            logger.info(f"> All plots saved in directory: {self.plot_dir.absolute()}")


class Pandora:
    """Pandora class for encapsulating a pandora run an it's results.

    Parameters
    ----------
    pandora_config : PandoraConfig
        PandoraConfig object used to determine the analyses to run

    Attributes
    ----------
    pandora_config : PandoraConfig
        PandoraConfig object used to determine the analyses to run
    dataset : EigenDataset
        EigenDataset object that contains the input data provided by the user
    replicates : List[EigenDataset]
        List of bootstrap replicates / sliding-windows of self.dataset.
        This is empty until self.bootstrap_embeddings() or self.sliding_window() was called.
    pairwise_stabilities : pd.DataFrame
        Pandas dataframe containing the Pandora stability scores for all pairwise replicate comparisons.
         This is empty until self.bootstrap_embeddings() or self.sliding_window() was called.
    pandora_stability : float
        Overall Pandora stability of the dataset under bootstrapping or sliding-window analysis.
        This is None until self.bootstrap_embeddings() or self.sliding_window() was called.
    pairwise_stabilities : pd.DataFrame
        Pandas dataframe containing the Pandora cluster stability scores for all pairwise replicate comparisons.
         This is empty until self.bootstrap_embeddings() or self.sliding_window() was called.
    pandora_cluster_stability : float
        Overall Pandora cluster stability of the dataset under bootstrapping or sliding-window analysis.
        This is None until self.bootstrap_embeddings() or self.sliding_window() was called.
    sample_support_values : pd.DataFrame
        Pandas dataframe containing the support values for all samples of self.dataset for all pairwise
        replicate comparisons.
        This is empty until self.bootstrap_embeddings() or self.sliding_window() was called.
    """

    def __init__(self, pandora_config: PandoraConfig):
        self.pandora_config: PandoraConfig = pandora_config
        self.dataset: EigenDataset = EigenDataset(
            file_prefix=pandora_config.dataset_prefix,
            embedding_populations=pandora_config.embedding_populations,
        )
        self.replicates: List[EigenDataset] = []

        self.pairwise_stabilities: pd.DataFrame = pd.DataFrame()
        self.pandora_stability: float = None

        self.pairwise_cluster_stabilities: pd.DataFrame = pd.DataFrame()
        self.pandora_cluster_stability: float = None

        self.sample_support_values: pd.Series = pd.Series(dtype=float)

    def embed_dataset(self) -> None:
        """Perfoms dimensionality reduction on self.dataset.

        The parameters (e.g. what method to use) is determined based on the configured settings in self.pandora_config.

        Returns
        -------
        None

        Raises
        ------
        PandoraConfigException
            - If `self.pandora_config.embedding_algorithm` is not a valid EmbeddingAlgorithm.
        """
        self.pandora_config.result_dir.mkdir(exist_ok=True, parents=True)
        if self.pandora_config.embedding_algorithm == EmbeddingAlgorithm.PCA:
            logger.info(fmt_message("Running SmartPCA on the input dataset."))
            self.dataset.run_pca(
                smartpca=self.pandora_config.smartpca,
                n_components=self.pandora_config.n_components,
                result_dir=self.pandora_config.result_dir,
                redo=self.pandora_config.redo,
                smartpca_optional_settings=self.pandora_config.smartpca_optional_settings,
            )
        elif self.pandora_config.embedding_algorithm == EmbeddingAlgorithm.MDS:
            logger.info(fmt_message("Performing MDS analysis on the input dataset."))
            self.dataset.run_mds(
                smartpca=self.pandora_config.smartpca,
                n_components=self.pandora_config.n_components,
                result_dir=self.pandora_config.result_dir,
                redo=self.pandora_config.redo,
            )
        else:
            raise PandoraConfigException(
                f"Unrecognized embedding algorithm: {self.pandora_config.embedding_algorithm}."
            )

        # determine the optimal number of clusters if not manually set by user
        if self.pandora_config.kmeans_k is None:
            if self.pandora_config.embedding_algorithm == EmbeddingAlgorithm.PCA:
                self.pandora_config.kmeans_k = self.dataset.pca.get_optimal_kmeans_k()
            else:
                self.pandora_config.kmeans_k = self.dataset.mds.get_optimal_kmeans_k()

        if self.pandora_config.plot_results:
            self.pandora_config.plot_dir.mkdir(exist_ok=True, parents=True)
            logger.info(
                fmt_message("Plotting embedding results for the input dataset.")
            )
            self._plot_dataset(self.dataset, self.dataset.name)

    def _plot_dataset(self, dataset: EigenDataset, plot_prefix: str) -> None:
        if self.pandora_config.embedding_algorithm == EmbeddingAlgorithm.PCA:
            embedding = dataset.pca
        elif self.pandora_config.embedding_algorithm == EmbeddingAlgorithm.MDS:
            embedding = dataset.mds
        else:
            raise PandoraConfigException(
                f"Unrecognized embedding algorithm: {self.pandora_config.embedding_algorithm}."
            )

        if embedding is None:
            raise PandoraException(
                "Embedding not yet run for dataset. Nothing to plot."
            )

        pcx = self.pandora_config.plot_dim_x
        pcy = self.pandora_config.plot_dim_y

        # plot with annotated populations
        fig = plot_populations(embedding, pcx, pcy)
        fig.write_image(
            self.pandora_config.plot_dir / f"{plot_prefix}_with_populations.pdf"
        )

        # plot with annotated clusters
        fig = plot_clusters(
            embedding,
            dim_x=pcx,
            dim_y=pcy,
            kmeans_k=self.pandora_config.kmeans_k,
        )
        fig.write_image(
            self.pandora_config.plot_dir / f"{plot_prefix}_with_clusters.pdf"
        )

        if len(self.dataset.embedding_populations) > 0:
            fig = plot_projections(
                embedding,
                embedding_populations=self.dataset.embedding_populations,
                dim_x=pcx,
                dim_y=pcy,
            )
            fig.write_image(
                self.pandora_config.plot_dir / f"{plot_prefix}_projections.pdf"
            )

    def bootstrap_embeddings(self) -> None:
        """Draws bootstrap replicates of self.dataset and computes and compares the respective embedding for all
        bootstrap replicates.

        The parameters (e.g. what method to use) is determined based on the configured settings in self.pandora_config.
        On successfull run, the following parameters of self will be set:

            - self.replicates
            - self.pairwise_stabilities
            - self.pandora_stability
            - self.pairwise_cluster_stabilities
            - self.pandora_cluster_stability
            - self.sample_support_values

        Returns
        -------
        None
        """
        logger.info(
            fmt_message(
                f"Drawing {self.pandora_config.n_replicates} bootstrapped datasets and "
                f"running {self.pandora_config.embedding_algorithm.value}."
            )
        )
        if self.pandora_config.bootstrap_convergence_check:
            logger.info(
                fmt_message(
                    "NOTE: Bootstrap convergence check is enabled. "
                    "Will terminate bootstrap computation once convergence is determined."
                )
            )
        self.replicates = bootstrap_and_embed_multiple(
            self.dataset,
            self.pandora_config.n_replicates,
            self.pandora_config.bootstrap_result_dir,
            self.pandora_config.smartpca,
            self.pandora_config.embedding_algorithm,
            self.pandora_config.n_components,
            self.pandora_config.seed,
            self.pandora_config.threads,
            self.pandora_config.redo,
            self.pandora_config.keep_replicates,
            self.pandora_config.bootstrap_convergence_check,
            self.pandora_config.smartpca_optional_settings,
        )
        logger.info(
            fmt_message(
                f"Bootstrapping done. Number of replicates computed: {len(self.replicates)}"
            )
        )
        self._compare_and_plot_replicates()

    def sliding_window(self) -> None:
        """Separates self.dataset into self.pandora_config.n_replicates overlapping windows and and computes and
        compares the respective embedding for all of these windows.

        The parameters (e.g. what method to use) is determined based on the configured settings in self.pandora_config.
        On successfull run, the following parameters of self will be set:

            - self.replicates
            - self.pairwise_stabilities
            - self.pandora_stability
            - self.pairwise_cluster_stabilities
            - self.pandora_cluster_stability
            - self.sample_support_values

        Returns
        -------
        None
        """
        logger.info(
            fmt_message(
                f"Separating the dataset into {self.pandora_config.n_replicates} sliding-windows "
                f"running {self.pandora_config.embedding_algorithm.value}."
            )
        )

        self.replicates = sliding_window_embedding(
            self.dataset,
            self.pandora_config.n_replicates,
            self.pandora_config.sliding_window_result_dir,
            self.pandora_config.smartpca,
            self.pandora_config.embedding_algorithm,
            self.pandora_config.n_components,
            self.pandora_config.threads,
            self.pandora_config.redo,
            self.pandora_config.keep_replicates,
            self.pandora_config.smartpca_optional_settings,
        )
        self._compare_and_plot_replicates()

    def _compare_and_plot_replicates(self) -> None:
        # =======================================
        # Compare and plot results
        # pairwise comparison between all replicates
        # =======================================
        analysis_string = (
            "bootstrapping"
            if self.pandora_config.analysis_mode == AnalysisMode.BOOTSTRAP
            else "sliding window"
        )
        logger.info(fmt_message(f"Comparing {analysis_string} embedding results."))
        self._compare_replicates_similarity()

        if self.pandora_config.plot_results:
            self.pandora_config.plot_dir.mkdir(exist_ok=True, parents=True)
            logger.info(fmt_message(f"Plotting {analysis_string} embedding results."))
            self._plot_replicates()
            self._plot_sample_support_values()

            if self.pandora_config.embedding_populations is not None:
                self._plot_sample_support_values(projected_samples_only=True)

    def _plot_replicates(self) -> None:
        for i, replicate in enumerate(self.replicates):
            self._plot_dataset(replicate, f"replicate_{i}")

    def _plot_sample_support_values(
        self, projected_samples_only: bool = False
    ) -> go.Figure:
        if self.pandora_config.embedding_algorithm == EmbeddingAlgorithm.PCA:
            embedding = self.dataset.pca
        elif self.pandora_config.embedding_algorithm == EmbeddingAlgorithm.MDS:
            embedding = self.dataset.mds
        else:
            raise PandoraConfigException(
                f"Unrecognized embedding algorithm: {self.pandora_config.embedding_algorithm}."
            )

        if embedding is None:
            raise PandoraException(
                "Support values are plotted using self.dataset.embedding, but dimensionality reduction was not performed"
                "for self.dataset. Make sure to run self.embed_dataset() prior to plotting."
            )
        pcx = self.pandora_config.plot_dim_x
        pcy = self.pandora_config.plot_dim_y

        projected_samples = (
            self.dataset.projected_samples if projected_samples_only else None
        )

        fig = plot_support_values(
            embedding,
            self.sample_support_values,
            self.pandora_config.support_value_rogue_cutoff,
            pcx,
            pcy,
            projected_samples,
        )

        if projected_samples_only:
            fig_name = "projected_sample_support_values.pdf"
        else:
            fig_name = "sample_support_values.pdf"

        fig.write_image(self.pandora_config.plot_dir / fig_name)
        return fig

    def _compare_replicates_similarity(self) -> None:
        # Compare all replicates pairwise
        if self.pandora_config.embedding_algorithm == EmbeddingAlgorithm.PCA:
            embedding = self.dataset.pca
            batch_comparison = BatchEmbeddingComparison(
                [b.pca for b in self.replicates]
            )
        elif self.pandora_config.embedding_algorithm == EmbeddingAlgorithm.MDS:
            embedding = self.dataset.mds
            batch_comparison = BatchEmbeddingComparison(
                [b.mds for b in self.replicates]
            )
        else:
            raise PandoraConfigException(
                f"Unrecognized embedding algorithm: {self.pandora_config.embedding_algorithm}."
            )

        if self.pandora_config.kmeans_k is not None:
            kmeans_k = self.pandora_config.kmeans_k
        else:
            kmeans_k = embedding.get_optimal_kmeans_k()

        if len(self.replicates) == 0:
            raise PandoraException("No replicates to compare!")

        self.pairwise_stabilities = batch_comparison.get_pairwise_stabilities(
            self.pandora_config.threads
        )
        self.pandora_stability = self.pairwise_stabilities.mean()
        self.pairwise_cluster_stabilities = (
            batch_comparison.get_pairwise_cluster_stabilities(
                kmeans_k, self.pandora_config.threads
            )
        )
        self.pandora_cluster_stability = self.pairwise_cluster_stabilities.mean()
        self.sample_support_values = batch_comparison.get_sample_support_values(
            self.pandora_config.threads
        )

    def log_and_save_replicates_results(self) -> None:
        """Logs the results of the bootstrap/sliding-window analyses using pandora.logging.logger and also saves the
        results of the analyses to the respective files as specified by self.pandora_config.

        Returns
        -------
        None

        Raises
        ------
        PandoraException
            If the results were not computed yet and thus there are not results to log.
        """
        if self.pandora_stability is None or self.pandora_cluster_stability is None:
            raise PandoraException("No results to log!")

        # store the pairwise results in a file
        _rd = self.pandora_config.result_decimals

        pairwise_stability_results = pd.concat(
            [self.pairwise_stabilities, self.pairwise_cluster_stabilities], axis=1
        )
        pairwise_stability_results.to_csv(
            self.pandora_config.pairwise_stability_result_file
        )

        # log the summary and save it in a file
        results_string = textwrap.dedent(
            f"""
            > Performed Analysis: {self.pandora_config.analysis_mode.value}
            > Number of replicates computed: {len(self.replicates)}
            > Number of Kmeans clusters: {self.pandora_config.kmeans_k}

            ------------------
            Results
            ------------------
            Pandora Stability: {round(self.pandora_stability, _rd)}
            Pandora Cluster Stability: {round(self.pandora_cluster_stability, _rd)}"""
        )

        self.pandora_config.result_file.open(mode="w").write(results_string)
        logger.info(results_string)

        self._log_and_save_sample_support_values()

    def _log_support_values(self, title: str, support_values: pd.Series) -> None:
        _rd = self.pandora_config.result_decimals
        _min = round(support_values.min(), _rd)
        _max = round(support_values.max(), _rd)
        _mean = round(support_values.mean(), _rd)
        _median = round(support_values.median(), _rd)
        _stdev = round(support_values.std(), _rd)

        support_values_result_string = textwrap.dedent(
            f"""
            ------------------
            {title}: Support values
            ------------------
            > average ± standard deviation: {_mean} ± {_stdev}
            > median: {_median}
            > lowest support value: {_min}
            > highest support value: {_max}
            """
        )
        logger.info(support_values_result_string)

    def _log_and_save_sample_support_values(self) -> None:
        if self.sample_support_values.empty:
            raise PandoraException("No results to log!")

        self.sample_support_values.to_csv(self.pandora_config.sample_support_values_csv)

        self._log_support_values("All Samples", self.sample_support_values)

        if self.dataset.projected_samples.empty:
            return

        projected_support_values = self.sample_support_values.loc[
            lambda x: x.index.isin(self.dataset.projected_samples)
        ]

        projected_support_values.to_csv(
            self.pandora_config.projected_sample_support_values_csv
        )

        self._log_support_values("Projected Samples", projected_support_values)


def pandora_config_from_configfile(configfile: pathlib.Path) -> PandoraConfig:
    """Creates a new PandoraConfig object using the provided yaml configuration file.

    Parameters
    ----------
    configfile : pathlib.Path
        Configuration file in yaml file_format

    Returns
    -------
    PandoraConfig
        PandoraConfig object with the settings according to the given yaml file.
        Uses the default settings as specified in the PandoraConfig class for optional options not explictly
        specified in the configfile.

    Raises
    ------
    PandoraConfigException
        - If the config file does not specify a `dataset_prefix`.
        - If the config file does not specify a `result_dir`.
        - If the PandoraConfig object could not be initialized. This is most likely due to misspecified config options.
    """
    config_data = yaml.safe_load(configfile.open())

    dataset_prefix = config_data.get("dataset_prefix")
    if dataset_prefix is None:
        raise PandoraConfigException("No dataset_prefix set.")
    else:
        config_data["dataset_prefix"] = pathlib.Path(dataset_prefix)

    result_dir = config_data.get("result_dir")
    if result_dir is None:
        raise PandoraConfigException("No result_dir set.")
    else:
        config_data["result_dir"] = pathlib.Path(result_dir)

    embedding_populations = config_data.get("embedding_populations")
    if embedding_populations is not None:
        config_data["embedding_populations"] = pathlib.Path(embedding_populations)

    file_format = config_data.get("file_format")
    if file_format is not None:
        config_data["file_format"] = FileFormat(file_format)

    embedding_algorithm = config_data.get("embedding_algorithm")
    if embedding_algorithm is not None:
        config_data["embedding_algorithm"] = EmbeddingAlgorithm[
            embedding_algorithm.upper()
        ]

    analysis_mode = config_data.get("analysis_mode")
    if analysis_mode is not None:
        config_data["analysis_mode"] = AnalysisMode[analysis_mode.upper()]

    try:
        return PandoraConfig.model_validate(config_data)
    except ValidationError as e:
        error_msg = ""
        for error in e.errors():
            error_msg += f"{error['loc']}: {error['msg']}; "
        raise PandoraConfigException(
            "Initializing Pandora from your config file failed! Got the following error(s): ",
            error_msg,
        )


def convert_to_eigenstrat_format(
    convertf: Executable,
    convertf_result_dir: pathlib.Path,
    dataset_prefix: pathlib.Path,
    file_format: FileFormat,
    redo: bool = False,
) -> pathlib.Path:
    """Converts the given dataset from the given file_format to EIGENSTRAT format and stores it in the
    convertf_result_dir.

    Results in three new files:\n
    - {convertf_result_dir}/{dataset_prefix.name}.geno\n
    - {convertf_result_dir}/{dataset_prefix.name}.snp\n
    - {convertf_result_dir}/{dataset_prefix.name}.ind

    Parameters
    ----------
    convertf : Executable
        Executable of the EIGENSOFT convertf program.
    convertf_result_dir : pathlib.Path
        Filepath where the output should be stored.
    dataset_prefix : pathlib.Path
        Prefix of the filepath pointing to the respective dataset files that should be converted.
    file_format : FileFormat
        Format of the input files.
     redo : bool, default=False
        Whether to rerun the conversion if the output files are already present.

    Returns
    -------
    convert_prefix : pathlib.Path
        Filepath prefix pointing to the converted genotype files in EIGENSTRAT format.
    """

    convertf_result_dir.mkdir(exist_ok=True, parents=True)
    convert_prefix = convertf_result_dir / dataset_prefix.name

    run_convertf(
        convertf=convertf,
        in_prefix=dataset_prefix,
        in_format=file_format,
        out_prefix=convert_prefix,
        out_format=FileFormat.EIGENSTRAT,
        redo=redo,
    )

    return convert_prefix
