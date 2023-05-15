import dataclasses
import itertools
import logging
import multiprocessing
import pathlib
import textwrap
import yaml

from pandora import __version__
from pandora.bootstrapping import create_bootstrap_pcas
from pandora.custom_types import *
from pandora.logger import logger, fmt_message
from pandora.pca import *
from pandora.pca_comparison import PCAComparison
from pandora.pca_runner import run_smartpca


@dataclasses.dataclass(repr=True)
class PandoraConfig:
    infile_prefix: FilePath
    outdir: FilePath

    n_bootstraps: int
    threads: int
    seed: int
    redo: bool
    variance_cutoff: int
    k_clusters: int

    smartpca: Executable
    convertf: Executable

    run_bootstrapping: bool
    sample_support_values: bool
    run_slidingWindow: bool
    run_plotting: bool

    plot_pcx: int
    plot_pcy: int

    verbosity: int
    results_decimals: int

    @property
    def dataset(self):
        return self.infile_prefix.name

    @property
    def outfile_prefix(self):
        return self.outdir / self.dataset

    @property
    def filtered_infile_prefix(self):
        return pathlib.Path(f"{self.outfile_prefix}.filtered")

    @property
    def logfile(self):
        return self.outdir / "pandora.log"

    @property
    def configfile(self):
        return self.outdir / "pandora.yaml"

    @property
    def bootstrap_dir(self):
        return self.outdir / "bootstrap"

    @property
    def pandora_results_file(self):
        return self.outdir / "pandora.results.txt"

    @property
    def pairwise_bootstrap_results_file(self):
        return self.outdir / "pandora.bootstrap.txt"

    @property
    def plot_dir(self):
        return self.outdir / "pcaPlots"

    @property
    def loglevel(self):
        if self.verbosity == 0:
            return logging.ERROR
        elif self.verbosity == 1:
            return logging.INFO
        elif self.verbosity == 2:
            return logging.DEBUG
        else:
            raise ValueError(f"verbosity needs to be 0 (ERROR), 1 (INFO), or 2 (DEBUG). Instead got value {self.verbosity}.")

    def get_configuration(self):
        config = dataclasses.asdict(self)

        # pathlib Paths cannot be dumped in yaml directly
        # so we have to manually replace them with their string representation
        for k, v in config.items():
            if isinstance(v, pathlib.Path):
                config[k] = str(v)

        return config

    def to_configfile(self):
        config_yaml = yaml.safe_dump(self.get_configuration())
        self.configfile.open(mode="w").write(f"# PANDORA VERSION {__version__}\n\n")
        self.configfile.open(mode="a").write(config_yaml)

    def create_outdirs(self):
        if self.run_bootstrapping:
            self.bootstrap_dir.mkdir(exist_ok=True, parents=True)
        if self.run_plotting:
            self.plot_dir.mkdir(exist_ok=True, parents=True)


@dataclasses.dataclass
class Pandora:
    pandora_config: PandoraConfig

    empirical_pca: PCA = None
    bootstrap_pcas: List[PCA] = None

    n_pcs: int = None
    kmeans_k: int = None

    bootstrap_similarities: List[float] = None
    bootstrap_cluster_similarities: List[float] = None

    def run_empirical_pca(self):
        # First, determine the number of PCs we are going to use
        # TODO: wie npcs bestimmen?
        # n_pcs = determine_number_of_pcs(
        #     infile_prefix=pandora_config.infile_prefix,
        #     outfile_prefix=pandora_config.outfile_prefix,
        #     smartpca=pandora_config.smartpca,
        #     # TODO: what is a meaningful variance cutoff?
        #     explained_variance_cutoff=pandora_config.variance_cutoff / 100,
        #     redo=pandora_config.redo,
        # )
        n_pcs = 20
        # Next, run SmartPCA with the determined number of PCs on the unmodified input dataset
        # fmt: off
        run_smartpca(
            infile_prefix   = self.pandora_config.infile_prefix,
            outfile_prefix  = self.pandora_config.outfile_prefix,
            smartpca        = self.pandora_config.smartpca,
            n_pcs           = n_pcs,
            redo            = self.pandora_config.redo,
        )
        # fmt: on
        # now run the empirical PCA again using the determined number of n_pcs
        evec_file = pathlib.Path(f"{self.pandora_config.outfile_prefix}.evec")
        eval_file = pathlib.Path(f"{self.pandora_config.outfile_prefix}.eval")
        self.empirical_pca = from_smartpca(evec_file, eval_file)
        self.n_pcs = n_pcs

    def run_bootstrap_pcas(self):
        # TODO: implement bootstopping
        self.bootstrap_pcas = create_bootstrap_pcas(
            infile_prefix=self.pandora_config.infile_prefix,
            bootstrap_outdir=self.pandora_config.bootstrap_dir,
            convertf=self.pandora_config.convertf,
            smartpca=self.pandora_config.smartpca,
            n_bootstraps=self.pandora_config.n_bootstraps,
            seed=self.pandora_config.seed,
            n_pcs=self.n_pcs,
            n_threads=self.pandora_config.threads,
            redo=self.pandora_config.redo
        )


    def _set_kmeans_k(self):
        kmeans_k_ckp = self.pandora_config.outdir / "kmeans.ckp"

        if kmeans_k_ckp.exists() and not self.pandora_config.redo:
            self.kmeans_k = int(open(kmeans_k_ckp).readline())
            return
        elif self.pandora_config.k_clusters is not None:
            kmeans_k = self.pandora_config.k_clusters
            logger.info(fmt_message(f"Using configured number of clusters: {kmeans_k}"))
        else:
            kmeans_k = self.empirical_pca.get_optimal_kmeans_k()
            logger.info(fmt_message(f"Optimal number of clusters determined to be: {kmeans_k}"))

        kmeans_k_ckp.open("w").write(str(kmeans_k))
        self.kmeans_k = kmeans_k

    def compare_bootstrap_results(self):
        # Compare all bootstraps pairwise
        bootstrap_similarities = []
        bootstrap_cluster_similarities = []

        pairwise_outfile = self.pandora_config.pairwise_bootstrap_results_file
        pairwise_outfile.unlink(missing_ok=True)

        # determine and set the number of clusters to use for K-Means comparison
        self._set_kmeans_k()

        with pairwise_outfile.open("a") as f:
            for (i1, bootstrap1), (i2, bootstrap2) in itertools.combinations(
                    enumerate(self.bootstrap_pcas, start=1), r=2):
                pca_comparison = PCAComparison(comparable=bootstrap1, reference=bootstrap2)
                similarity = pca_comparison.compare()
                bootstrap_similarities.append(similarity)

                clustering_score = pca_comparison.compare_clustering(kmeans_k=self.kmeans_k)
                bootstrap_cluster_similarities.append(clustering_score)

                f.write(f"{i1}\t{i2}\t{round(similarity, 4)}\t{round(clustering_score, 4)}\n")

        self.bootstrap_similarities = bootstrap_similarities
        self.bootstrap_cluster_similarities = bootstrap_cluster_similarities

    def log_and_save_bootstrap_results(self):
        _mean_pca = round(np.mean(self.bootstrap_similarities), self.pandora_config.results_decimals)
        _std_pca = round(np.std(self.bootstrap_similarities), self.pandora_config.results_decimals)

        _mean_kmeans = round(np.mean(self.bootstrap_cluster_similarities), self.pandora_config.results_decimals)
        _std_kmeans = round(np.std(self.bootstrap_cluster_similarities), self.pandora_config.results_decimals)

        bootstrap_results_string = textwrap.dedent(f"""
        > Number of Bootstrap replicates computed: {self.pandora_config.n_bootstraps}
        > Number of PCs required to explain at least {self.pandora_config.variance_cutoff}% variance: {self.n_pcs}
        > Optimal number of clusters: {self.kmeans_k}
        ------------------
        Bootstrapping Similarity
        ------------------
        PCA: {_mean_pca} ± {_std_pca}
        K-Means clustering: {_mean_kmeans} ± {_std_kmeans}
        
        > Pairwise bootstrap results written to {self.pandora_config.pairwise_bootstrap_results_file}
        """)

        logger.info(bootstrap_results_string)
        self.pandora_config.pandora_results_file.open(mode="a").write(bootstrap_results_string)

    def plot_pca(self, pca:PCA, plot_prefix: str):
        pcx = self.pandora_config.plot_pcx
        pcy = self.pandora_config.plot_pcy
        # plot with annotated populations
        pca.plot(
            pcx=pcx,
            pcy=pcy,
            annotation="population",
            outfile=self.pandora_config.plot_dir / f"{plot_prefix}_with_populations.pdf",
            redo=self.pandora_config.redo
        )

        # plot with annotated clusters
        pca.plot(
            pcx=pcx,
            pcy=pcy,
            annotation="cluster",
            kmeans_k=self.kmeans_k,
            outfile=self.pandora_config.plot_dir / f"{plot_prefix}_with_clusters.pdf",
            redo=self.pandora_config.redo
        )

        logger.info(fmt_message(f"Plotted bootstrap PCA {plot_prefix}"))

    def plot_results(self):
        self.plot_pca(self.empirical_pca, "empirical")
        for i, bootstrap_pca in enumerate(self.bootstrap_pcas):
            self.plot_pca(bootstrap_pca, f"bootstrap_{i + 1}")


def pandora_config_from_configfile(configfile: FilePath) -> PandoraConfig:
    config_data = yaml.load(open(configfile), yaml.Loader)

    # fmt: off
    return PandoraConfig(
        infile_prefix   = pathlib.Path(config_data["input"]),
        outdir          = pathlib.Path(config_data["output"]),
        n_bootstraps    = config_data.get("bootstraps", 100),
        threads         = config_data.get("threads", multiprocessing.cpu_count()),
        seed            = config_data.get("seed", 0),
        redo            = config_data.get("redo", False),
        variance_cutoff = config_data.get("varianceCutoff", 95),
        k_clusters      = config_data.get("kClusters", None),
        smartpca        = config_data.get("smartpca", "smartpca"),
        convertf        = config_data.get("convertf", "convertf"),
        run_bootstrapping       = config_data.get("bootstrap", True),
        sample_support_values   = config_data.get("supportValues", False),
        run_slidingWindow       = config_data.get("slidingWindow", False),
        run_plotting    = config_data.get("plotResults", False),
        plot_pcx        = config_data.get("plot_pcx", 0),
        plot_pcy        = config_data.get("plot_pcy", 0),
        verbosity       = config_data.get("verbosity", 2),
        results_decimals = config_data.get("resultsDecimals", 2)
    )
    # fmt: on




