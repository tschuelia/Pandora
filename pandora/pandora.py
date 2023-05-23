import dataclasses
import itertools
import math
import multiprocessing
import random
import statistics
import textwrap
from collections import defaultdict
from multiprocessing import Pool
from plotly import graph_objects as go

import yaml

from pandora import __version__
from pandora.custom_types import *
from pandora.custom_errors import *
from pandora.dataset import Dataset
from pandora.logger import *
from pandora.pca_comparison import PCAComparison


@dataclasses.dataclass
class PandoraConfig:
    dataset_prefix: FilePath
    result_dir: FilePath

    # Bootstrap related settings
    n_pcs: int
    variance_cutoff: float
    n_bootstraps: int
    smartpca: Executable
    rogueness_cutoff: float
    smartpca_optional_settings: dict[str, str]

    # Cluster settings
    kmeans_k: Union[int, None]

    # Pandora execution mode settings
    do_bootstrapping: bool
    sample_support_values: bool
    plot_results: bool
    redo: bool
    seed: int
    threads: int
    result_decimals: int
    verbosity: int

    # Plot settings
    plot_pcx: int
    plot_pcy: int

    def __post_init__(self):
        # TODO: validate inputs
        """
        - Check that all input files are present
        """
        if self.sample_support_values and not self.do_bootstrapping:
            raise PandoraConfigException("Cannot compute the sample support values without bootstrap replicates. "
                                         "Set `bootstrap: True` in the config file and restart the analysis.")

    @property
    def pandora_logfile(self) -> FilePath:
        return self.result_dir / "pandora.log"

    @property
    def configfile(self) -> FilePath:
        return self.result_dir / "pandora.yaml"

    @property
    def result_file(self) -> FilePath:
        return self.result_dir / "pandora.txt"

    @property
    def bootstrap_result_dir(self) -> FilePath:
        return self.result_dir / "bootstrap"

    @property
    def pairwise_bootstrap_result_file(self) -> FilePath:
        return self.result_dir / "pandora.bootstrap.txt"

    @property
    def sample_support_values_file(self) -> FilePath:
        return self.result_dir / "pandora.supportValues.txt"

    @property
    def plot_dir(self) -> FilePath:
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
                f"verbosity needs to be 0 (ERROR), 1 (INFO), or 2 (DEBUG). Instead got value {self.verbosity}.")

    def get_configuration(self):
        config = dataclasses.asdict(self)

        # pathlib Paths cannot be dumped in yaml directly
        # so we have to manually replace them with their string representation
        for k, v in config.items():
            if isinstance(v, pathlib.Path):
                config[k] = str(v)

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
        self.pandora_config = pandora_config
        self.dataset = Dataset(pandora_config.dataset_prefix)
        self.bootstrap_datasets = []  # type: List[Dataset]
        self.bootstrap_similarities = {}  # type: Dict[Tuple[int, int], float]
        self.bootstrap_cluster_similarities = {}  # type: Dict[Tuple[int, int], float]

        self.kmeans_k = self.pandora_config.kmeans_k

        self.sample_support_values = {}  # type: Dict[str, float]

    def do_pca(self):
        self.dataset.smartpca(
            result_dir=self.pandora_config.result_dir,
            smartpca=self.pandora_config.smartpca,
            n_pcs=self.pandora_config.n_pcs,
            redo=self.pandora_config.redo,
            smartpca_optional_settings=self.pandora_config.smartpca_optional_settings
        )

    def _plot_pca(self, dataset: Dataset, plot_prefix: str):
        pcx = self.pandora_config.plot_pcx
        pcy = self.pandora_config.plot_pcy
        # plot with annotated populations
        dataset.pca.plot(
            pcx=pcx,
            pcy=pcy,
            annotation="population",
            outfile=self.pandora_config.plot_dir / f"{plot_prefix}_with_populations.pdf",
            redo=self.pandora_config.redo
        )

        # plot with annotated clusters
        dataset.pca.plot(
            pcx=pcx,
            pcy=pcy,
            annotation="cluster",
            kmeans_k=self.kmeans_k,
            outfile=self.pandora_config.plot_dir / f"{plot_prefix}_with_clusters.pdf",
            redo=self.pandora_config.redo
        )

    def plot_dataset(self):
        self._plot_pca(self.dataset, self.dataset.name)

    # ===========================
    # BOOTSTRAP RELATED FUNCTIONS
    # ===========================
    def bootstrap_dataset(self):
        random.seed(self.pandora_config.seed)

        args = [
            (self.pandora_config.bootstrap_result_dir / f"bootstrap_{i}", random.randint(0, 1_000_000), self.pandora_config.redo)
            for i in range(self.pandora_config.n_bootstraps)
        ]

        with Pool(self.pandora_config.threads) as p:
            self.bootstrap_datasets = list(p.starmap(self.dataset.create_bootstrap, args))

    def _run_pca(self, bootstrap: Dataset):
        bootstrap.smartpca(
            smartpca=self.pandora_config.smartpca,
            n_pcs=self.pandora_config.n_pcs,
            redo=self.pandora_config.redo,
            smartpca_optional_settings=self.pandora_config.smartpca_optional_settings
        )
        return bootstrap

    def bootstrap_pcas(self):
        with Pool(self.pandora_config.threads) as p:
            # the subprocesses in Pool do not modify the object but work on a copy
            # we thus replace the objects with their modified versions
            self.bootstrap_datasets = list(p.map(self._run_pca, self.bootstrap_datasets))

    def plot_bootstraps(self):
        for i, bootstrap in enumerate(self.bootstrap_datasets):
            self._plot_pca(bootstrap, f"bootstrap_{i}")

    def compare_bootstrap_similarity(self):
        # Compare all bootstraps pairwise
        if self.kmeans_k is not None:
            kmeans_k = self.kmeans_k
        else:
            kmeans_k = self.dataset.pca.get_optimal_kmeans_k()

        for (i1, bootstrap1), (i2, bootstrap2) in itertools.combinations(enumerate(self.bootstrap_datasets), r=2):
            pca_comparison = PCAComparison(comparable=bootstrap1.pca, reference=bootstrap2.pca)
            self.bootstrap_similarities[(i1, i2)] = pca_comparison.compare()
            self.bootstrap_cluster_similarities[(i1, i2)] = pca_comparison.compare_clustering(kmeans_k)

    def log_and_save_bootstrap_results(self):
        # store the pairwise results in a file
        _rd = self.pandora_config.result_decimals

        with self.pandora_config.pairwise_bootstrap_result_file.open(mode="a") as f:
            for (i1, i2), similarity in self.bootstrap_similarities.items():
                cluster_similarity = self.bootstrap_cluster_similarities[(i1, i2)]

                f.write(f"{i1}\t{i2}\t{round(similarity, _rd)}\t{round(cluster_similarity, _rd)}\n")

        # log the summary and save it in a file
        _mean_pca = round(statistics.mean(self.bootstrap_similarities.values()), _rd)
        _std_pca = round(statistics.stdev(self.bootstrap_similarities.values()), _rd)

        _mean_kmeans = round(statistics.mean(self.bootstrap_cluster_similarities.values()), _rd)
        _std_kmeans = round(statistics.stdev(self.bootstrap_cluster_similarities.values()), _rd)

        bootstrap_results_string = textwrap.dedent(
            f"""
            > Number of Bootstrap replicates computed: {self.pandora_config.n_bootstraps}
            > Number of Kmeans clusters: {self.kmeans_k}

            ------------------
            Bootstrapping Similarity
            ------------------
            PCA: {_mean_pca} ± {_std_pca}
            K-Means clustering: {_mean_kmeans} ± {_std_kmeans}"""
        )

        self.pandora_config.result_file.open(mode="a").write(bootstrap_results_string)
        logger.info(bootstrap_results_string)

    def compute_sample_support_values(self):
        # compare all bootstrap results pairwise and determine the rogue samples for each comparison
        self.dataset.set_sample_ids_and_populations()
        rogue_counter = defaultdict(int)

        for bootstrap1, bootstrap2 in itertools.combinations(self.bootstrap_datasets, r=2):
            pca_comparison = PCAComparison(comparable=bootstrap1.pca, reference=bootstrap2.pca)
            rogue_samples = pca_comparison.detect_rogue_samples(rogue_cutoff=self.pandora_config.rogueness_cutoff)
            for sample in rogue_samples:
                rogue_counter[sample] += 1

        n_bootstrap_combinations = math.comb(self.pandora_config.n_bootstraps, 2)

        # compute the support value for each sample
        # to do so, we compute 1 - (#rogue / #combinations)
        for sample in self.dataset.samples:
            self.sample_support_values[sample] = 1 - (rogue_counter.get(sample, 0) / n_bootstrap_combinations)

    def save_sample_support_values(self):
        with self.pandora_config.sample_support_values_file.open(mode="w") as f:
            for sample, support_value in self.sample_support_values.items():
                f.write(f"{sample}\t{round(support_value, self.pandora_config.result_decimals)}\n")

    def plot_sample_support_values(self):
        # Annotate the sample support values for each sample in the empirical PCA
        pca_data = self.dataset.pca.pca_data

        # to make sure we are correctly matching the sample IDs with their support apply explicit sorting
        pca_data = pca_data.sort_values(by="sample_id").reset_index(drop=True)
        support_values = list(self.sample_support_values.items())
        support_values = sorted(support_values)

        pcx = self.pandora_config.plot_pcx
        pcy = self.pandora_config.plot_pcy

        fig = go.Figure(
            go.Scatter(
                x=pca_data[f"PC{pcx}"],
                y=pca_data[f"PC{pcy}"],
                mode="markers+text",
                # annotate only samples with a support value < 1
                text=[f"{round(support, 2)}" if support < 1 else "" for (_, support) in support_values],
                textposition="bottom center"
            )
        )
        fig.update_xaxes(title=f"PC {pcx + 1}")
        fig.update_yaxes(title=f"PC {pcy + 1}")
        fig.update_layout(template="plotly_white", height=1000, width=1000)
        fig.write_image(self.pandora_config.plot_dir / f"sample_support_values.pdf", )
        return fig


def pandora_config_from_configfile(configfile: FilePath) -> PandoraConfig:
    config_data = yaml.safe_load(configfile.open())

    # fmt: off
    return PandoraConfig(
        dataset_prefix  = pathlib.Path(config_data["input"]),
        result_dir      = pathlib.Path(config_data["output"]),

        # Bootstrap related settings
        n_pcs           = 20,  # TODO: implement automatic detection
        variance_cutoff = config_data.get("varianceCutoff", 0.95),
        n_bootstraps    = config_data.get("bootstraps", 100),
        smartpca        = config_data.get("smartpca", "smartpca"),
        smartpca_optional_settings  = config_data.get("smartpcaOptionalSettings", {}),
        rogueness_cutoff = config_data.get("roguenessCutoff", 0.95),

        # Cluster settings
        kmeans_k = config_data.get("kClusters", None),

        # Pandora execution mode settings
        do_bootstrapping        = config_data.get("bootstrap", True),
        sample_support_values   = config_data.get("supportValues", False),
        plot_results            = config_data.get("plotResults", False),
        redo            = config_data.get("redo", False),
        seed            = config_data.get("seed", 0),
        threads         = config_data.get("threads", multiprocessing.cpu_count()),
        result_decimals = config_data.get("resultsDecimals", 2),
        verbosity   = config_data.get("verbosity", 2),

        # Plot settings
        plot_pcx    = config_data.get("plot_pcx", 0),
        plot_pcy    = config_data.get("plot_pcy", 0),
    )
    # fmt: on