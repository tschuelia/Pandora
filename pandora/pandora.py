import dataclasses
import datetime
import itertools
import logging
import multiprocessing
import pathlib
import random
import statistics
import textwrap

import pandas as pd
import yaml
from pydantic import NonNegativeInt, PositiveInt, ValidationError
from pydantic.dataclasses import dataclass

from pandora import __version__
from pandora.converter import run_convertf
from pandora.dataset import (
    EigenDataset,
    bootstrap_and_embed_multiple,
    smartpca_finished,
)
from pandora.embedding_comparison import BatchEmbeddingComparison
from pandora.logger import fmt_message, logger
from pandora.plotting import *


@dataclass
class PandoraConfig:
    """
    Pydantic dataclass encapsulating the settings required to run Pandora.

    Attributes:
        dataset_prefix (pathlib.Path): File path prefix pointing to the dataset to use for the Pandora analyses.
            Pandora will look for files called <input>.* so make sure all files have the same prefix.
        result_dir (pathlib.Path): Directory where to store all (intermediate) results to.
        file_format (FileFormat): Format of the input dataset.
            Can be ANCESTRYMAP, EIGENSTRAT, PED, PACKEDPED, PACKEDANCESTRYMAP. Default is EIGENSTRAT.
        convertf (Executable): File path pointing to an executable of Eigensoft's convertf tool. Convertf is used
            if the provided dataset is not in EIGENSTRAT format. Default is 'convertf'. This will only work if
            convertf is installed systemwide.
        n_bootstraps (PositiveInt): Number of bootstrap replicates to compute. Default is 100,
        keep_bootstraps (bool): Whether to store all bootstrap datasets files (.geno, .snp, .ind). Note that this will
            result in a substantial storage consumption. Default is False. Note that the bootstrapped indicies are
            stored as checkpoints for full reproducibility in any case.
        n_components (PositiveInt): Number of Principal Components to output and compare for PCA analyses. Default is 20.
        smartpca (Executable): File path pointing to an executable of Eigensoft's smartpca tool. Smartpca is used
            for PCA analyses on the provided dataset. Default is 'smartpca'. This will only work if smartpca is
            installed systemwide.
        smartpca_optional_settings (Dict[str, str]): Optional additional settings to use when performing PCA with
            smartpca. Pandora has full support for all smartpca options. Not allowed are the following options:
            genotypename, snpname, indivname, evecoutname, evaloutname, numoutevec, maxpops.
            Use the following schema to set the options: dict(shrinkmode=True, numoutlieriter=1)
        embedding_populations (pathlib.Path): File containing a new-line separated list of population names.
            Only these populations will be used for the dimensionality reduction. In case of PCA analyses, all remaining
            samples in the dataset will be projected onto the PCA results.
        support_value_rogue_cutoff (float): When plotting the support values, only samples with a support value lower
            than the support_value_rogue_cutoff  will be annotated with their sample IDs.
            Note that all samples in the respective plot are color-coded according to their support value in any case.
            Default is 0.5.
        kmeans_k (PositiveInt): Number of clusters k to use for K-Means clustering of the dimensionality reduction embeddings.
            If not set, the optimal number of clusters will be automatically determined according to the
            Bayesian Information Criterion (BIC).
        do_bootstrapping (bool): Whether to do the stability analysis using bootstrapping. Default is True.
        redo (bool): Whether to rerun all analyses in case the results files from a previous run are already present.
            Careful: this will overwrite existing results! Default is False.
        seed (int): Seed to initialize the random number generator. This setting is recommended for reproducible
            analyses. Default is the current unix timestamp.
        threads (NonNegativeInt): Number of threads to use for the analysis. Default is the number of CPUs available.
        result_decimals (NonNegativeInt): Number of decimals to round the stability scores and support values in the output.
            Default is two decimals.
        verbosity (int): Verbosity of the output logging of Pandora.
            0 = quiet, prints only errors and the results (loglevel = ERROR)
            1 = verbose, prints all intermediate infos (loglevel = INFO)
            2 = debug, prints intermediate infos and debug messages (loglevel = DEBUG)
        plot_results (bool): Whether to plot all dimensionality reduction results and sample support values.
            Default is False.
        plot_dim_x (NonNegativeInt): Dimension to plot on the x-axis. Note that the dimensions are zero-indexed. To plot the first
            dimension set plot_dim_x = 0. Default is 0.
        plot_dim_y (NonNegativeInt): Dimension to plot on the y-axis. Note that the dimensions are zero-indexed. To plot the second
            dimension set plot_dim_y = 1. Default is 1.
    """

    # EigenDataset related
    dataset_prefix: pathlib.Path
    result_dir: pathlib.Path
    file_format: FileFormat = FileFormat.EIGENSTRAT
    convertf: Executable = "convertf"

    # Bootstrap related settings
    n_bootstraps: NonNegativeInt = 100
    keep_bootstraps: bool = False

    # Embedding related
    n_components: NonNegativeInt = 20
    embedding_algorithm: EmbeddingAlgorithm = EmbeddingAlgorithm.PCA
    smartpca: Executable = "smartpca"
    smartpca_optional_settings: Optional[Dict[str, Any]] = None
    embedding_populations: Optional[
        pathlib.Path
    ] = None  # list of populations to use for Embedding and later project the remaining populations on the Embedding

    # sample support values
    support_value_rogue_cutoff: float = 0.5

    # Cluster settings
    kmeans_k: Optional[int] = None

    # Pandora execution mode settings
    do_bootstrapping: bool = True
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
        return self.result_dir / "pandora.log"

    @property
    def configfile(self) -> pathlib.Path:
        return self.result_dir / "pandora.yaml"

    @property
    def result_file(self) -> pathlib.Path:
        return self.result_dir / "pandora.txt"

    @property
    def bootstrap_result_dir(self) -> pathlib.Path:
        return self.result_dir / "bootstrap"

    @property
    def pairwise_bootstrap_result_file(self) -> pathlib.Path:
        return self.result_dir / "pandora.bootstrap.csv"

    @property
    def sample_support_values_csv(self) -> pathlib.Path:
        return self.result_dir / "pandora.supportValues.pairwise.csv"

    @property
    def sample_support_values_projected_samples_file(self) -> pathlib.Path:
        return self.result_dir / "pandora.supportValues.projected.txt"

    @property
    def plot_dir(self) -> pathlib.Path:
        return self.result_dir / "plots"

    @property
    def loglevel(self):
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

    def convert_to_eigenstrat_format(self):
        logger.info(
            fmt_message(
                f"Converting dataset from {self.file_format.value} to {FileFormat.EIGENSTRAT.value}"
            )
        )
        convertf_dir = self.result_dir / "convertf"
        convertf_dir.mkdir(exist_ok=True)
        convert_prefix = convertf_dir / self.dataset_prefix.name

        run_convertf(
            convertf=self.convertf,
            in_prefix=self.dataset_prefix,
            in_format=self.file_format,
            out_prefix=convert_prefix,
            out_format=FileFormat.EIGENSTRAT,
            redo=self.redo,
        )

        self.dataset_prefix = convert_prefix

    def get_configuration(self):
        config = dataclasses.asdict(self)

        # pathlib Paths cannot be dumped in yaml directly
        # so we have to manually replace them with their string representation
        for k, v in config.items():
            if isinstance(v, pathlib.Path):
                config[k] = str(v.absolute())
            elif isinstance(v, Enum):
                config[k] = v.value

        return config

    def save_config(self):
        config_yaml = yaml.safe_dump(self.get_configuration())
        # additionally save the Pandora version
        self.configfile.open(mode="w").write(f"# PANDORA VERSION {__version__}\n\n")
        self.configfile.open(mode="a").write(config_yaml)

    def log_results_files(self):
        logger.info(
            textwrap.dedent(
                """
                ------------------
                Result Files
                ------------------"""
            )
        )
        logger.info(f"> Pandora results: {self.result_file.absolute()}")
        if self.do_bootstrapping:
            logger.info(
                f"> Pairwise bootstrap similarities: {self.pairwise_bootstrap_result_file.absolute()}"
            )
            logger.info(
                f"> Sample Support values: {self.sample_support_values_csv.absolute()}"
            )

            if self.embedding_populations is not None:
                logger.info(
                    f"> Projected Sample Support values: {self.sample_support_values_projected_samples_file.absolute()}"
                )
        if self.plot_results:
            logger.info(f"> All plots saved in directory: {self.plot_dir.absolute()}")


class Pandora:
    def __init__(self, pandora_config: PandoraConfig):
        self.pandora_config: PandoraConfig = pandora_config
        self.dataset: EigenDataset = EigenDataset(
            file_prefix=pandora_config.dataset_prefix,
            embedding_populations=pandora_config.embedding_populations,
        )
        self.bootstraps: List[EigenDataset] = []

        self.bootstrap_stabilities: pd.DataFrame = pd.DataFrame()
        self.bootstrap_stability: float = None

        self.bootstrap_cluster_stabilities: pd.DataFrame = pd.DataFrame()
        self.bootstrap_cluster_stability: float = None

        self.sample_support_values: pd.DataFrame = pd.DataFrame()

    def embed_dataset(self):
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

        if self.pandora_config.plot_results:
            self.pandora_config.plot_dir.mkdir(exist_ok=True, parents=True)
            logger.info(
                fmt_message("Plotting embedding results for the input dataset.")
            )
            self._plot_dataset(self.dataset, self.dataset.name)

    def _plot_dataset(self, dataset: EigenDataset, plot_prefix: str):
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
                embedding_populations=list(self.dataset.embedding_populations),
                dim_x=pcx,
                dim_y=pcy,
            )
            fig.write_image(
                self.pandora_config.plot_dir / f"{plot_prefix}_projections.pdf"
            )

    # ===========================
    # BOOTSTRAP RELATED FUNCTIONS
    # ===========================
    def bootstrap_embeddings(self):
        """
        Create bootstrap datasets and run dimensionality reduction on each dataset
        """
        logger.info(
            fmt_message(
                f"Drawing {self.pandora_config.n_bootstraps} bootstrapped datasets and "
                f"running {self.pandora_config.embedding_algorithm.value}."
            )
        )
        self.bootstraps = bootstrap_and_embed_multiple(
            self.dataset,
            self.pandora_config.n_bootstraps,
            self.pandora_config.bootstrap_result_dir,
            self.pandora_config.smartpca,
            self.pandora_config.embedding_algorithm,
            self.pandora_config.n_components,
            self.pandora_config.seed,
            self.pandora_config.threads,
            self.pandora_config.redo,
            self.pandora_config.keep_bootstraps,
            self.pandora_config.smartpca_optional_settings,
        )

        # =======================================
        # Compare results
        # pairwise comparison between all bootstraps
        # =======================================
        logger.info(fmt_message(f"Comparing bootstrap embedding results."))
        self._compare_bootstrap_similarity()

        if self.pandora_config.plot_results:
            self.pandora_config.plot_dir.mkdir(exist_ok=True, parents=True)
            logger.info(fmt_message(f"Plotting bootstrap embedding results."))
            self._plot_bootstraps()
            self._plot_sample_support_values()

            if self.pandora_config.embedding_populations is not None:
                self._plot_sample_support_values(projected_samples_only=True)

    def _plot_bootstraps(self):
        for i, bootstrap in enumerate(self.bootstraps):
            self._plot_dataset(bootstrap, f"bootstrap_{i}")

    def _plot_sample_support_values(self, projected_samples_only: bool = False):
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
            self.sample_support_values.mean(axis=1),
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

    def _compare_bootstrap_similarity(self):
        # Compare all bootstraps pairwise
        if self.pandora_config.embedding_algorithm == EmbeddingAlgorithm.PCA:
            embedding = self.dataset.pca
            batch_comparison = BatchEmbeddingComparison(
                [b.pca for b in self.bootstraps]
            )
        elif self.pandora_config.embedding_algorithm == EmbeddingAlgorithm.MDS:
            embedding = self.dataset.mds
            batch_comparison = BatchEmbeddingComparison(
                [b.mds for b in self.bootstraps]
            )
        else:
            raise PandoraConfigException(
                f"Unrecognized embedding algorithm: {self.pandora_config.embedding_algorithm}."
            )

        if self.pandora_config.kmeans_k is not None:
            kmeans_k = self.pandora_config.kmeans_k
        else:
            kmeans_k = embedding.get_optimal_kmeans_k()

        if len(self.bootstraps) == 0:
            raise PandoraException("No bootstrap replicates to compare!")

        self.bootstrap_stability = batch_comparison.compare()
        self.bootstrap_stabilities = batch_comparison.get_pairwise_stabilities()
        self.bootstrap_cluster_stability = batch_comparison.compare_clustering(kmeans_k)
        self.bootstrap_cluster_stabilities = (
            batch_comparison.get_pairwise_cluster_stabilities(kmeans_k)
        )
        self.sample_support_values = batch_comparison.get_sample_support_values()

    def log_and_save_bootstrap_results(self):
        if self.bootstrap_stability is None or self.bootstrap_cluster_stability is None:
            raise PandoraException("No bootstrap results to log!")

        # store the pairwise results in a file
        _rd = self.pandora_config.result_decimals

        pairwise_stability_results = pd.concat(
            [self.bootstrap_stabilities, self.bootstrap_cluster_stabilities], axis=1
        )
        pairwise_stability_results.to_csv(
            self.pandora_config.pairwise_bootstrap_result_file
        )

        # log the summary and save it in a file
        bootstrap_results_string = textwrap.dedent(
            f"""
            > Number of Bootstrap replicates computed: {self.pandora_config.n_bootstraps}
            > Number of Kmeans clusters: {self.pandora_config.kmeans_k}

            ------------------
            Bootstrapping Results
            ------------------
            Pandora Stability: {round(self.bootstrap_stability, _rd)}
            Pandora Cluster Stability: {round(self.bootstrap_cluster_stability, _rd)}"""
        )

        self.pandora_config.result_file.open(mode="w").write(bootstrap_results_string)
        logger.info(bootstrap_results_string)

    def _log_support_values(self, title: str, support_values: pd.Series):
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

    def log_and_save_sample_support_values(self):
        if self.sample_support_values.empty:
            raise PandoraException("No bootstrap results to log!")

        support_values = self.sample_support_values.copy()
        support_values["mean"] = support_values.mean(axis=1)
        support_values["stdev"] = support_values.std(axis=1)

        support_values.to_csv(self.pandora_config.sample_support_values_csv)

        self._log_support_values("All Samples", self.sample_support_values.mean(axis=1))

        if self.dataset.projected_samples.empty:
            return

        projected_support_values = self.sample_support_values.loc[
            lambda x: x.index.isin(self.dataset.projected_samples)
        ]
        self._log_support_values(
            "Projected Samples", projected_sample_support_values.mean(axis=1)
        )


def pandora_config_from_configfile(configfile: pathlib.Path) -> PandoraConfig:
    """
    Creates a new PandoraConfig object using the provided yaml configuration file.

    Args:
        configfile (pathlib.Path): Configuration file in yaml file_format

    Returns:
        PandoraConfig: PandoraConfig object with the settings according to the given yaml file.
            Uses the default settings as specified in the PandoraConfig class for optional options not explictly
            specified in the configfile.
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

    try:
        return PandoraConfig(**config_data)
    except TypeError as e:
        raise PandoraConfigException(
            "Got an unrecognized init argument for PandoraConfig. Make sure you are only passing "
            "parameters specified in the Pandora wiki and don't have any spelling errors!",
            e.args,
        )
    except ValidationError as e:
        error_msg = ""
        for error in e.errors():
            error_msg += f"{error['loc']}: {error['msg']}; "
        raise PandoraConfigException(
            "Initialzing Pandora from your config file failed! Got the following error(s): ",
            error_msg,
        )
