import dataclasses
import itertools
import multiprocessing
import random
import textwrap
import statistics
from multiprocessing import Pool

import yaml

from pandora import __version__
from pandora.converter import run_convertf
from pandora.dataset import Dataset, smartpca_finished
from pandora.logger import *
from pandora.plotting import *


@dataclasses.dataclass
class PandoraConfig:
    dataset_prefix: pathlib.Path
    file_format: FileFormat
    result_dir: pathlib.Path

    # Bootstrap related settings
    n_pcs: int
    n_bootstraps: int
    keep_bootstraps: bool

    # PCA related
    smartpca: Executable
    convertf: Executable
    smartpca_optional_settings: dict[str, str]

    # sample support values
    support_value_rogue_cutoff: float
    pca_populations: Union[
        pathlib.Path, None
    ]  # list of populations to use for PCA and later project the remaining the populations on the PCA

    # Cluster settings
    kmeans_k: Union[int, None]

    # Pandora execution mode settings
    do_bootstrapping: bool
    plot_results: bool
    redo: bool
    seed: int
    threads: int
    result_decimals: int
    verbosity: int

    # Plot settings
    plot_pcx: int
    plot_pcy: int

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

    def create_result_dirs(self):
        if self.do_bootstrapping:
            self.bootstrap_result_dir.mkdir(exist_ok=True, parents=True)
        if self.plot_results:
            self.plot_dir.mkdir(exist_ok=True, parents=True)


class Pandora:
    def __init__(self, pandora_config: PandoraConfig):
        self.pandora_config: PandoraConfig = pandora_config
        self.dataset: Dataset = Dataset(
            file_prefix=pandora_config.dataset_prefix,
            pca_populations=pandora_config.pca_populations,
        )
        self.bootstrap_datasets: List[Dataset] = []
        # TODO: umbauen auf pd.Dataframes
        self.bootstrap_similarities: Dict[Tuple[int, int], float] = {}
        self.bootstrap_cluster_similarities: Dict[Tuple[int, int], float] = {}
        self.sample_support_values: pd.DataFrame = pd.DataFrame()

    def do_pca(self):
        self.dataset.smartpca(
            result_dir=self.pandora_config.result_dir,
            smartpca=self.pandora_config.smartpca,
            n_pcs=self.pandora_config.n_pcs,
            redo=self.pandora_config.redo,
            smartpca_optional_settings=self.pandora_config.smartpca_optional_settings,
        )

    def _plot_pca(self, dataset: Dataset, plot_prefix: str):
        pcx = self.pandora_config.plot_pcx
        pcy = self.pandora_config.plot_pcy

        # plot with annotated populations
        fig = plot_pca_populations(dataset.pca, pcx, pcy)
        fig.write_image(
            self.pandora_config.plot_dir / f"{plot_prefix}_with_populations.pdf"
        )

        # plot with annotated clusters
        fig = plot_pca_clusters(
            dataset.pca,
            pcx=pcx,
            pcy=pcy,
            kmeans_k=self.pandora_config.kmeans_k,
        )
        fig.write_image(
            self.pandora_config.plot_dir / f"{plot_prefix}_with_clusters.pdf"
        )

        if len(self.dataset.pca_populations) > 0:
            fig = plot_pca_projections(
                dataset.pca,
                pca_populations=list(self.dataset.pca_populations),
                pcx=pcx,
                pcy=pcy,
            )
            fig.write_image(
                self.pandora_config.plot_dir / f"{plot_prefix}_projections.pdf"
            )

    def plot_dataset(self):
        self._plot_pca(self.dataset, self.dataset.name)

    # ===========================
    # BOOTSTRAP RELATED FUNCTIONS
    # ===========================
    def _run_pca(self, bootstrap: Dataset):
        bootstrap.smartpca(
            smartpca=self.pandora_config.smartpca,
            n_pcs=self.pandora_config.n_pcs,
            redo=self.pandora_config.redo,
            smartpca_optional_settings=self.pandora_config.smartpca_optional_settings,
        )
        return bootstrap

    def bootstrap_pcas(self):
        """
        Create bootstrap datasets and run PCA on each dataset
        """
        # In order to save storage, we are deleting the bootstrap datasets (.ind, .geno, .snp) files.
        # So before we create bootstrap datasets, we have to check whether the subsequent PCA run finished already
        # we do this check for each of the expected Bootstrap outputs and only compute the missing ones
        random.seed(self.pandora_config.seed)
        random_seeds = [
            random.randint(0, 1_000_000)
            for _ in range(self.pandora_config.n_bootstraps)
        ]

        bootstraps_to_do = []
        for i in range(self.pandora_config.n_bootstraps):
            bs_prefix = self.pandora_config.bootstrap_result_dir / f"bootstrap_{i}"

            if smartpca_finished(self.pandora_config.n_pcs, bs_prefix):
                continue

            # we need to pass the seed like this to make sure we use the same seed in subsequent runs
            bootstraps_to_do.append((bs_prefix, random_seeds[i]))

        # 1. Create bootstrap dataset
        with Pool(self.pandora_config.threads) as p:
            args = [
                (prefix, seed, self.pandora_config.redo)
                for prefix, seed in bootstraps_to_do
            ]
            list(p.starmap(self.dataset.create_bootstrap, args))

        self.bootstrap_datasets = [
            Dataset(
                self.pandora_config.bootstrap_result_dir / f"bootstrap_{i}",
                self.dataset.pca_populations_file,
                self.dataset.samples,
            )
            for i in range(self.pandora_config.n_bootstraps)
        ]

        # 2. Run bootstrap PCAs
        with Pool(self.pandora_config.threads) as p:
            # the subprocesses in Pool do not modify the object but work on a copy
            # we thus replace the objects with their modified versions
            self.bootstrap_datasets = list(
                p.map(self._run_pca, self.bootstrap_datasets)
            )

        # 3. EIGEN files can be rather large, so we delete the bootstrapped datasets
        if not self.pandora_config.keep_bootstraps:
            [bs.remove_input_files() for bs in self.bootstrap_datasets]

        # TODO: die Schritte oben zusammen ziehen damit bootstrap -> PCA -> löschen nicht sequenziell für all N gemacht wird
        # sondern alles direkt parallel

    def plot_bootstraps(self):
        for i, bootstrap in enumerate(self.bootstrap_datasets):
            self._plot_pca(bootstrap, f"bootstrap_{i}")

    def compare_bootstrap_similarity(self):
        # Compare all bootstraps pairwise
        if self.pandora_config.kmeans_k is not None:
            kmeans_k = self.pandora_config.kmeans_k
        else:
            kmeans_k = self.dataset.pca.get_optimal_kmeans_k()

        sample_supports = []

        for (i1, bootstrap1), (i2, bootstrap2) in itertools.combinations(
            enumerate(self.bootstrap_datasets), r=2
        ):
            pca_comparison = PCAComparison(
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

        self.pandora_config.result_file.open(mode="a").write(bootstrap_results_string)
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

    def plot_sample_support_values(self, projected_samples_only: bool = False):
        pcx = self.pandora_config.plot_pcx
        pcy = self.pandora_config.plot_pcy

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


def pandora_config_from_configfile(configfile: pathlib.Path) -> PandoraConfig:
    config_data = yaml.safe_load(configfile.open())

    dataset_prefix = config_data.get("dataset_prefix")
    result_dir = config_data.get("result_dir")

    if dataset_prefix is None:
        raise PandoraConfigException("No dataset_prefix set.")
    if result_dir is None:
        raise PandoraConfigException("No result_dir set.")

    pca_populations = config_data.get("pca_populations")
    if pca_populations is not None:
        pca_populations = pathlib.Path(pca_populations)

    # fmt: off
    return PandoraConfig(
        dataset_prefix  = pathlib.Path(dataset_prefix),
        file_format     = FileFormat(config_data.get("file_format", "EIGENSTRAT")),
        result_dir      = pathlib.Path(result_dir),

        # Bootstrap related settings
        n_pcs           = config_data.get("n_pcs", 0.95),
        n_bootstraps    = config_data.get("n_bootstraps", 100),
        keep_bootstraps = config_data.get("keep_bootstraps", False),

        # PCA related
        smartpca = config_data.get("smartpca", "smartpca"),
        convertf = config_data.get("convertf", "convertf"),
        smartpca_optional_settings = config_data.get("smartpca_optional_settings", {}),

        # sample support values
        support_value_rogue_cutoff = config_data.get("support_value_rogue_cutoff", 0.5),
        pca_populations = pca_populations,

        # Cluster settings
        kmeans_k = config_data.get("kmeans_k", None),

        # Pandora execution mode settings
        do_bootstrapping    = config_data.get("bootstrap", True),
        plot_results        = config_data.get("plot_results", False),
        redo                = config_data.get("redo", False),
        seed                = config_data.get("seed", 0),
        threads             = config_data.get("threads", multiprocessing.cpu_count()),
        result_decimals     = config_data.get("result_decimals", 2),
        verbosity           = config_data.get("verbosity", 2),

        # Plot settings
        plot_pcx = config_data.get("plot_pcx", 0),
        plot_pcy = config_data.get("plot_pcy", 1),
    )
    # fmt: on


def convert_to_eigenstrat_format(pandora_config: PandoraConfig):
    convertf_dir = pandora_config.result_dir / "convertf"
    convertf_dir.mkdir(exist_ok=True)
    convert_prefix = convertf_dir / pandora_config.dataset_prefix.name

    run_convertf(
        convertf=pandora_config.convertf,
        in_prefix=pandora_config.dataset_prefix,
        in_format=pandora_config.file_format,
        out_prefix=convert_prefix,
        out_format=FileFormat.EIGENSTRAT,
        redo=pandora_config.redo,
    )

    pandora_config.dataset_prefix = convert_prefix
