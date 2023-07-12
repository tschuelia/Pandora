import dataclasses
import datetime
import itertools
import logging
import multiprocessing
import pathlib
import random
import textwrap
import statistics
from multiprocessing import Pool

import yaml

from pandora import __version__
from pandora.converter import run_convertf
from pandora.dataset import Dataset, smartpca_finished
from pandora.logger import logger, fmt_message
from pandora.plotting import *


@dataclasses.dataclass
class PandoraConfig:
    """
    Dataclass encapsulating the settings required to run Pandora.

    Attributes:
        dataset_prefix (pathlib.Path): File path prefix pointing to the dataset to use for the Pandora analyses.
        result_dir (pathlib.Path): File path where to store all (intermediate) results to.
        file_format (FileFormat): Format of the input dataset.
            Can be ANCESTRYMAP, EIGENSTRAT, PED, PACKEDPED, PACKEDANCESTRYMAP. Default is EIGENSTRAT.
        convertf (Executable): File path pointing to an executable of Eigensoft's convertf tool. Convertf is used
            if the provided dataset is not in EIGENSTRAT format. Default is 'convertf'. This will only work if
            convertf is installed systemwide.
        n_bootstraps (int): Number of bootstrap replicates to compute. Default is 100,
        keep_bootstraps (bool): Whether to store all bootstrap datasets files (.geno, .snp, .ind). Note that this will
            result in a substantial storage consumption. Default is False. Note that the bootstrapped indicies are
            stored as checkpoints for full reproducibility in any case.
        n_pcs (int): Number of Principal Components to output and compare for PCA analyses. Default is 20.
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
        kmeans_k (int): Number of clusters k to use for K-Means clustering of the dimensionality reduction embeddings.
            If not set, the optimal number of clusters will be automatically determined according to the
            Bayesian Information Criterion (BIC).
        do_bootstrapping (bool): Whether to do the stability analysis using bootstrapping. Default is True.
        redo (bool): Whether to rerun all analyses in case the results files from a previous run are already present.
            Default is False.
        seed (int): Seed to initialize the random number generator. This setting is recommended for reproducible
            analyses. Default is the current unix timestamp.
        threads (int): Number of threads to use for the analysis. Default is the number of CPUs available.
        result_decimals (int): Number of decimals to round the stability scores and support values in the output.
        verbosity (int): Verbosity of the output logging of Pandora.
            0 = quiet, prints only errors and the results (loglevel = ERROR)
            1 = verbose, prints all intermediate infos (loglevel = INFO)
            2 = debug, prints intermediate infos and debug messages (loglevel = DEBUG)
        plot_results (bool): Whether to plot all dimensionality reduction results and sample support values.
            Default is False.
        plot_dim_x (int): Dimension to plot on the x-axis. Note that the dimensions are zero-indexed. To plot the first
            dimension set plot_dim_x = 0.
        plot_dim_y (int): Dimension to plot on the y-axis. Note that the dimensions are zero-indexed. To plot the second
            dimension set plot_dim_y = 1.
    """

    # Dataset related
    dataset_prefix: pathlib.Path
    result_dir: pathlib.Path
    file_format: FileFormat = FileFormat.EIGENSTRAT
    convertf: Executable = "convertf"

    # Bootstrap related settings
    n_bootstraps: int = 100
    keep_bootstraps: bool = False

    # PCA related
    n_pcs: int = 20
    smartpca: Executable = "smartpca"
    smartpca_optional_settings: Optional[Dict[str, str]] = None
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
    threads: int = multiprocessing.cpu_count()
    result_decimals: int = 2
    verbosity: int = 2

    # Plot settings
    plot_results: bool = False
    plot_dim_x: int = 0
    plot_dim_y: int = 1

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
        return self.result_dir / "pandora.bootstrap.txt"

    @property
    def sample_support_values_file(self) -> pathlib.Path:
        return self.result_dir / "pandora.supportValues.txt"

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
            elif isinstance(v, FileFormat):
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
                f"> Sample Support values: {self.sample_support_values_file.absolute()}"
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
        self.dataset: Dataset = Dataset(
            file_prefix=pandora_config.dataset_prefix,
            embedding_populations=pandora_config.embedding_populations,
        )
        self.bootstrap_datasets: List[Dataset] = []
        self.bootstrap_similarities: Dict[Tuple[int, int], float] = {}
        self.bootstrap_cluster_similarities: Dict[Tuple[int, int], float] = {}
        self.sample_support_values: pd.DataFrame = pd.DataFrame()

    def do_pca(self):
        self.pandora_config.result_dir.mkdir(exist_ok=True, parents=True)
        logger.info(fmt_message("Running SmartPCA on the input dataset."))
        self.dataset.smartpca(
            result_dir=self.pandora_config.result_dir,
            smartpca=self.pandora_config.smartpca,
            n_pcs=self.pandora_config.n_pcs,
            redo=self.pandora_config.redo,
            smartpca_optional_settings=self.pandora_config.smartpca_optional_settings,
        )

        if self.pandora_config.plot_results:
            self.pandora_config.plot_dir.mkdir(exist_ok=True, parents=True)
            logger.info(fmt_message("Plotting SmartPCA results for the input dataset."))
            self._plot_dataset()

    def _plot_pca(self, dataset: Dataset, plot_prefix: str):
        if dataset.pca is None:
            raise PandoraException("No PCA run for dataset yet. Nothing to plot.")
        pcx = self.pandora_config.plot_dim_x
        pcy = self.pandora_config.plot_dim_y

        # plot with annotated populations
        fig = plot_populations(dataset.pca, pcx, pcy)
        fig.write_image(
            self.pandora_config.plot_dir / f"{plot_prefix}_with_populations.pdf"
        )

        # plot with annotated clusters
        fig = plot_clusters(
            dataset.pca,
            dim_x=pcx,
            dim_y=pcy,
            kmeans_k=self.pandora_config.kmeans_k,
        )
        fig.write_image(
            self.pandora_config.plot_dir / f"{plot_prefix}_with_clusters.pdf"
        )

        if len(self.dataset.embedding_populations) > 0:
            fig = plot_projections(
                dataset.pca,
                embedding_populations=list(self.dataset.embedding_populations),
                dim_x=pcx,
                dim_y=pcy,
            )
            fig.write_image(
                self.pandora_config.plot_dir / f"{plot_prefix}_projections.pdf"
            )

    def _plot_dataset(self):
        self._plot_pca(self.dataset, self.dataset.name)

    # ===========================
    # BOOTSTRAP RELATED FUNCTIONS
    # ===========================
    def bootstrap_pcas(self):
        """
        Create bootstrap datasets and run PCA on each dataset
        """
        logger.info(
            fmt_message(
                f"Drawing {self.pandora_config.n_bootstraps} bootstrapped datasets and running SmartPCA."
            )
        )
        self.pandora_config.bootstrap_result_dir.mkdir(exist_ok=True, parents=True)
        random.seed(self.pandora_config.seed)
        args = [
            (
                self.pandora_config.bootstrap_result_dir / f"bootstrap_{i}",
                random.randint(0, 1_000_000),
                self.pandora_config.redo,
            )
            for i in range(self.pandora_config.n_bootstraps)
        ]
        with Pool(self.pandora_config.threads) as p:
            self.bootstrap_datasets = list(p.map(self._bootstrap_pca, args))

        # =======================================
        # Compare results
        # pairwise comparison between all bootstraps
        # =======================================
        logger.info(fmt_message(f"Comparing bootstrap PCA results."))
        self._compare_bootstrap_similarity()

        if self.pandora_config.plot_results:
            self.pandora_config.plot_dir.mkdir(exist_ok=True, parents=True)
            logger.info(fmt_message(f"Plotting bootstrap PCA results."))
            self._plot_bootstraps()
            self._plot_sample_support_values()

            if self.pandora_config.embedding_populations is not None:
                self._plot_sample_support_values(projected_samples_only=True)

    def _bootstrap_pca(self, args):
        bootstrap_prefix, seed, redo = args
        if smartpca_finished(self.pandora_config.n_pcs, bootstrap_prefix):
            # SmartPCA results are present and correct
            # Thus we initialize a bootstrap dataset manually using the correct prefix
            # We still need to call .smartpca later on to make sure bootstrap_dataset.embedding is set properly
            bootstrap_dataset = Dataset(
                bootstrap_prefix,
                self.dataset.embedding_populations_file,
                self.dataset.samples,
            )
        else:
            # draw bootstrap dataset
            bootstrap_dataset = self.dataset.create_bootstrap(
                bootstrap_prefix, seed, redo
            )

        # run smartpca
        bootstrap_dataset.smartpca(
            smartpca=self.pandora_config.smartpca,
            n_pcs=self.pandora_config.n_pcs,
            redo=self.pandora_config.redo,
            smartpca_optional_settings=self.pandora_config.smartpca_optional_settings,
        )

        if not self.pandora_config.keep_bootstraps:
            bootstrap_dataset.remove_input_files()

        return bootstrap_dataset

    def _plot_bootstraps(self):
        for i, bootstrap in enumerate(self.bootstrap_datasets):
            self._plot_pca(bootstrap, f"bootstrap_{i}")

    def _plot_sample_support_values(self, projected_samples_only: bool = False):
        if self.dataset.pca is None:
            raise PandoraException(
                "Support values are plotted using self.dataset.embedding, but PCA was not performed for self.dataset. "
                "Make sure to run self.do_pca prior to plotting."
            )
        pcx = self.pandora_config.plot_dim_x
        pcy = self.pandora_config.plot_dim_y

        projected_samples = (
            self.dataset.projected_samples.sample_id.unique()
            if projected_samples_only
            else None
        )

        fig = plot_support_values(
            self.dataset.pca,
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
        if self.pandora_config.kmeans_k is not None:
            kmeans_k = self.pandora_config.kmeans_k
        else:
            kmeans_k = self.dataset.pca.get_optimal_kmeans_k()

        sample_supports = []

        for (i1, bootstrap1), (i2, bootstrap2) in itertools.combinations(
            enumerate(self.bootstrap_datasets), r=2
        ):
            pca_comparison = EmbeddingComparison(
                comparable=bootstrap1.pca, reference=bootstrap2.pca
            )
            self.bootstrap_similarities[(i1, i2)] = pca_comparison.compare()
            self.bootstrap_cluster_similarities[
                (i1, i2)
            ] = pca_comparison.compare_clustering(kmeans_k)

            support_values = pca_comparison.get_sample_support_values()
            support_values.name = f"({i1}, {i2})"
            sample_supports.append(support_values)

        self.sample_support_values = pd.concat(sample_supports, axis=1)

    def log_and_save_bootstrap_results(self):
        if (
            len(self.bootstrap_similarities) == 0
            or len(self.bootstrap_cluster_similarities) == 0
        ):
            raise PandoraException("No bootstrap results to log!")

        # store the pairwise results in a file
        _rd = self.pandora_config.result_decimals

        with self.pandora_config.pairwise_bootstrap_result_file.open(mode="a") as f:
            for (i1, i2), similarity in self.bootstrap_similarities.items():
                cluster_similarity = self.bootstrap_cluster_similarities[(i1, i2)]

                f.write(
                    f"{i1}\t{i2}\t{round(similarity, _rd)}\t{round(cluster_similarity, _rd)}\n"
                )

        # log the summary and save it in a file
        _mean_pca = round(statistics.mean(self.bootstrap_similarities.values()), _rd)
        _std_pca = round(statistics.stdev(self.bootstrap_similarities.values()), _rd)

        _mean_kmeans = round(
            statistics.mean(self.bootstrap_cluster_similarities.values()), _rd
        )
        _std_kmeans = round(
            statistics.stdev(self.bootstrap_cluster_similarities.values()), _rd
        )

        bootstrap_results_string = textwrap.dedent(
            f"""
            > Number of Bootstrap replicates computed: {self.pandora_config.n_bootstraps}
            > Number of Kmeans clusters: {self.pandora_config.kmeans_k}

            ------------------
            Bootstrapping Similarity
            ------------------
            PCA: {_mean_pca} ± {_std_pca}
            K-Means clustering: {_mean_kmeans} ± {_std_kmeans}"""
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
        if len(self.sample_support_values) == 0:
            raise PandoraException("No bootstrap results to log!")

        _rd = self.pandora_config.result_decimals

        all_samples_file = self.pandora_config.sample_support_values_file

        with all_samples_file.open("w") as _all:
            for sample_id, support in self.sample_support_values.mean(axis=1).items():
                _all.write(f"{sample_id}\t{round(support, _rd)}\n")

        self._log_support_values("All Samples", self.sample_support_values.mean(axis=1))

        # store all pairwise support values in a csv file in case someone want to explore it further
        self.sample_support_values.to_csv(self.pandora_config.sample_support_values_csv)

        if self.dataset.projected_samples.empty:
            return

        projected_samples_file = (
            self.pandora_config.sample_support_values_projected_samples_file
        )

        projected_sample_support_values = self.sample_support_values.loc[
            lambda x: x.index.isin(self.dataset.projected_samples.sample_id.tolist())
        ]
        with projected_samples_file.open("w") as _projected:
            for sample_id, support in projected_sample_support_values.mean(
                axis=1
            ).items():
                _projected.write(f"{sample_id}\t{round(support, _rd)}\n")

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

    return PandoraConfig(**config_data)
