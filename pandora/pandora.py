import dataclasses
import itertools
import multiprocessing
import textwrap
import yaml

from pandora.bootstrapping import create_bootstrap_pcas
from pandora.logger import *
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
    run_slidingWindow: bool
    run_plotting: bool

    plot_pcx: int
    plot_pcy: int

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
    def bootstrap_dir(self):
        return self.outdir / "bootstrap"

    @property
    def plot_dir(self):
        return self.outdir / "pcaPlots"

    def get_configuration(self):
        return dataclasses.asdict(self)

    def to_configfile(self, configfile: FilePath):
        # TODO: implementieren -> config verbose in ein yaml file schreiben
        # fuer reproducibility
        # auch pandora version als kommentar oben reinschreiben
        raise NotImplementedError

    def create_outdirs(self):
        if self.run_bootstrapping:
            self.bootstrap_dir.mkdir(exist_ok=True, parents=True)
        if self.run_plotting:
            self.plot_dir.mkdir(exist_ok=True, parents=True)


def from_config(configfile: FilePath) -> PandoraConfig:
    config_data = yaml.load(open(configfile), yaml.Loader)

    uncertainty_analyses = config_data.get("uncertainty", {})
    # TODO: implement support values
    sample_analyses = config_data.get("sampleProjection", {})

    return PandoraConfig(
        infile_prefix=pathlib.Path(config_data["input"]),
        outdir=pathlib.Path(config_data["output"]),
        n_bootstraps=config_data.get("bootstraps", 100),
        threads=config_data.get("threads", multiprocessing.cpu_count()),
        seed=config_data.get("seed", 0),
        redo=config_data.get("redo", False),
        variance_cutoff=config_data.get("varianceCutoff", 95),
        k_clusters=config_data.get("kClusters", None),
        smartpca=config_data.get("smartpca", "smartpca"),
        convertf=config_data.get("convertf", "convertf"),
        run_bootstrapping=uncertainty_analyses.get("bootstrap", True),
        run_slidingWindow=uncertainty_analyses.get("slidingWindow", False),
        run_plotting=config_data.get("plotResults", False),
        plot_pcx=config_data.get("plot_pcx", 0),
        plot_pcy=config_data.get("plot_pcy", 0),
    )


def run_empirical_pca(pandora_config: PandoraConfig):
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
    run_smartpca(
        infile_prefix=pandora_config.infile_prefix,
        outfile_prefix=pandora_config.outfile_prefix,
        smartpca=pandora_config.smartpca,
        n_pcs=n_pcs,
        redo=pandora_config.redo,
    )
    # now run the empirical PCA again using the determined number of n_pcs
    evec_file = pathlib.Path(f"{pandora_config.outfile_prefix}.evec")
    eval_file = pathlib.Path(f"{pandora_config.outfile_prefix}.eval")
    empirical_pca = from_smartpca(evec_file, eval_file)
    return empirical_pca


def run_bootstrap_pcas(pandora_config: PandoraConfig, n_pcs: int):
    logger.info(fmt_message(f"Drawing {pandora_config.n_bootstraps} bootstrapped datasets."))

    # TODO: implement bootstopping

    bootstrap_pcas = create_bootstrap_pcas(
        infile_prefix=pandora_config.infile_prefix,
        bootstrap_outdir=pandora_config.bootstrap_dir,
        convertf=pandora_config.convertf,
        smartpca=pandora_config.smartpca,
        n_bootstraps=pandora_config.n_bootstraps,
        seed=pandora_config.seed,
        n_pcs=n_pcs,
        n_threads=pandora_config.threads,
        redo=pandora_config.redo
    )

    return bootstrap_pcas


def get_kmeans_k(pandora_config: PandoraConfig, empirical_pca: PCA):
    kmeans_k_ckp = pandora_config.outdir / "kmeans.ckp"

    if kmeans_k_ckp.exists() and not pandora_config.redo:
        kmeans_k = int(open(kmeans_k_ckp).readline())
    elif pandora_config.k_clusters is not None:
        kmeans_k = pandora_config.k_clusters
        kmeans_k_ckp.open("w").write(str(kmeans_k))
        logger.info(fmt_message(f"Using configured number of clusters: {kmeans_k}"))
    else:
        kmeans_k = empirical_pca.get_optimal_kmeans_k()
        kmeans_k_ckp.open("w").write(str(kmeans_k))
        logger.info(fmt_message(f"Optimal number of clusters determined to be: {kmeans_k}"))

    return kmeans_k


def compare_bootstrap_results(pandora_config: PandoraConfig, bootstrap_pcas: List[PCA], kmeans_k: int):
    # Compare all bootstraps pairwise
    bootstrap_similarities = []
    bootstrap_cluster_similarities = []

    pairwise_outfile = pathlib.Path(f"{pandora_config.outfile_prefix}.pandora.bootstrap.txt")
    pairwise_outfile.unlink(missing_ok=True)

    with pairwise_outfile.open("a") as f:
        for (i1, bootstrap1), (i2, bootstrap2) in itertools.combinations(enumerate(bootstrap_pcas, start=1), r=2):
            pca_comparison = PCAComparison(comparable=bootstrap1, reference=bootstrap2)
            similarity = pca_comparison.compare()
            bootstrap_similarities.append(similarity)

            clustering_score = pca_comparison.compare_clustering(kmeans_k=kmeans_k)
            bootstrap_cluster_similarities.append(clustering_score)

            f.write(f"{i1}\t{i2}\t{round(similarity, 4)}\t{round(clustering_score, 4)}\n")

    return bootstrap_similarities, bootstrap_cluster_similarities


def plot_pca(pandora_config: PandoraConfig, pca: PCA, kmeans_k: int, plot_prefix: str):
    # TODO: make plotted PCs command line settable
    pcx = 0
    pcy = 1

    # plot with annotated populations
    pca.plot(
        pcx=pcx,
        pcy=pcy,
        annotation="population",
        outfile=pandora_config.plot_dir / f"{plot_prefix}_with_populations.pdf",
        redo=pandora_config.redo
    )

    # plot with annotated clusters
    pca.plot(
        pcx=pcx,
        pcy=pcy,
        annotation="cluster",
        kmeans_k=kmeans_k,
        outfile=pandora_config.plot_dir / f"{plot_prefix}_with_clusters.pdf",
        redo=pandora_config.redo
    )

    logger.info(fmt_message(f"Plotted bootstrap PCA {plot_prefix}"))


def plot_bootstraps(pandora_config: PandoraConfig, bootstrap_pcas: List[PCA], kmeans_k: int):
    # TODO: make plotted PCs command line settable
    # TODO: paralleles plotten
    pcx = 0
    pcy = 1

    for i, bootstrap_pca in enumerate(bootstrap_pcas):
        # TODO: populations so zu setzen funktioniert nicht mehr wenn man die outlier in smartpca raus schmeisst
        # -> muss einmal am anfang das mapping sampleID -> population speichern
        # bootstrap_pca.set_populations(empirical_pca.pca_data.population)

        # plot Bootstrap with annotated populations
        bootstrap_pca.plot(
            pcx=pcx,
            pcy=pcy,
            annotation="population",
            outfile=pandora_config.plot_dir / f"bootstrap_{i + 1}_with_populations.pca.pdf",
            redo=pandora_config.redo
        )

        # plot Bootstrap only with annotated clusters
        bootstrap_pca.plot(
            pcx=pcx,
            pcy=pcy,
            annotation="cluster",
            kmeans_k=kmeans_k,
            outfile=pandora_config.plot_dir / f"bootstrap_{i + 1}_with_clusters.pca.pdf",
            redo=pandora_config.redo
        )

        logger.info(fmt_message(f"Plotted bootstrap PCA #{i + 1}"))


def plot_results(pandora_config: PandoraConfig, empirical_pca: PCA, bootstrap_pcas: List[PCA], kmeans_k: int):
    plot_pca(pandora_config, empirical_pca, kmeans_k, "empirical")
    for i, bootstrap_pca in enumerate(bootstrap_pcas):
        plot_pca(pandora_config, bootstrap_pca, kmeans_k, f"bootstrap_{i + 1}")
