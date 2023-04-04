import dataclasses
import itertools
import multiprocessing
import yaml

from pandora.bootstrapping import create_bootstrap_pcas
from pandora.converter import *
from pandora.logger import *
from pandora.pca import *
from pandora.pca_comparison import PCAComparison
from pandora.pca_runner import *


@dataclasses.dataclass(repr=True)
class PandoraConfig:
    infile_prefix: FilePath
    outdir: FilePath

    n_bootstraps: int
    threads: int
    seed: int
    redo: bool
    variance_cutoff: int

    smartpca: Executable
    convertf: Executable
    plink2: Executable

    run_bootstrapping: bool
    run_alternative: bool
    run_slidingWindow: bool
    run_plotting: bool

    @property
    def dataset(self):
        return self.infile_prefix.name

    @property
    def outfile_prefix(self):
        return self.outdir / self.dataset

    @property
    def logfile(self):
        return self.outdir / "pandora.log"

    @property
    def convertf_dir(self):
        return self.outdir / "convertf"

    @property
    def convertf_prefix(self):
        return self.convertf_dir / self.dataset

    @property
    def convertf_dir_binary(self):
        return self.convertf_dir / "binary"

    @property
    def convertf_prefix_binary(self):
        return self.convertf_dir_binary / self.dataset

    @property
    def bootstrap_dir(self):
        return self.outdir / "bootstrap"

    @property
    def alternative_tools_dir(self):
        return self.outdir / "alternativeTools"

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
            self.convertf_dir.mkdir(exist_ok=True, parents=True)
            self.bootstrap_dir.mkdir(exist_ok=True, parents=True)
        if self.run_alternative:
            self.convertf_dir_binary.mkdir(exist_ok=True, parents=True)
            self.alternative_tools_dir.mkdir(exist_ok=True, parents=True)
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
        smartpca=config_data.get("smartpca", "smartpca"),
        convertf=config_data.get("convertf", "convertf"),
        plink2=config_data.get("plink2", "plink2"),
        run_bootstrapping=uncertainty_analyses.get("bootstrap", True),
        run_alternative=uncertainty_analyses.get("alternativeTools", True),
        run_slidingWindow=uncertainty_analyses.get("slidingWindow", False),
        run_plotting=config_data.get("plotResults", False),
    )


def run_empirical_pca(pandora_config: PandoraConfig):
    # First, determine the number of PCs we are going to use
    n_pcs = determine_number_of_pcs(
        infile_prefix=pandora_config.infile_prefix,
        outfile_prefix=pandora_config.outfile_prefix,
        smartpca=pandora_config.smartpca,
        # TODO: what is a meaningful variance cutoff?
        explained_variance_cutoff=pandora_config.variance_cutoff / 100,
        redo=pandora_config.redo,
    )
    # Next, run SmartPCA with the determined number of PCs on the unmodified input dataset
    run_smartpca(
        infile_prefix=pandora_config.infile_prefix,
        outfile_prefix=pandora_config.outfile_prefix,
        smartpca=pandora_config.smartpca,
        n_pcs=n_pcs,
        redo=pandora_config.redo,
    )
    # now run the empirical PCA again using the determined number of n_pcs
    empirical_pca = from_smartpca(pathlib.Path(f"{pandora_config.outfile_prefix}.evec"))
    return empirical_pca


def run_bootstrap_pcas(pandora_config: PandoraConfig, n_pcs: int):
    logger.info(fmt_message("Converting Input files to PLINK format for bootstrapping."))

    eigen_to_plink(
        eigen_prefix=pandora_config.infile_prefix,
        plink_prefix=pandora_config.convertf_prefix,
        convertf=pandora_config.convertf,
        redo=pandora_config.redo
    )

    logger.info(fmt_message(f"Drawing {pandora_config.n_bootstraps} bootstrapped datasets."))

    # TODO: implement bootstopping

    bootstrap_pcas = create_bootstrap_pcas(
        infile_prefix=pandora_config.convertf_prefix,
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


def run_plink_pca(pandora_config: PandoraConfig, n_pcs: int):
    plink_to_bplink(
        plink_prefix=pandora_config.convertf_prefix,
        bplink_prefix=pandora_config.convertf_prefix_binary,
        convertf=pandora_config.convertf,
        redo=pandora_config.redo
    )

    plink_prefix = pandora_config.alternative_tools_dir / "plink2"

    run_plink(
        infile_prefix=pandora_config.convertf_prefix_binary,
        outfile_prefix=plink_prefix,
        plink=pandora_config.plink2,
        n_pcs=n_pcs,
        redo=pandora_config.redo
    )

    plink_pca = from_plink(
        plink_evec_file=pathlib.Path(f"{plink_prefix}.eigenvec"),
        plink_eval_file=pathlib.Path(f"{plink_prefix}.eigenval"),
    )

    return plink_pca


def run_sklearn_pca(pandora_config: PandoraConfig, n_pcs: int):
    sklearn_prefix = pandora_config.alternative_tools_dir / "sklearn"

    # we use PLINK to generate us the input files for the PCA analysis
    bplink_to_datamatrix(
        bplink_prefix=pandora_config.convertf_prefix_binary,
        outfile_prefix=sklearn_prefix,
        plink=pandora_config.plink2,
        redo=pandora_config.redo
    )

    return run_sklearn(
        outfile_prefix=sklearn_prefix,
        n_pcs=n_pcs,
        redo=pandora_config.redo
    )


def run_alternative_pcas(pandora_config: PandoraConfig, n_pcs: int):
    # TODO: parallel ausrechnen
    alternatives = {}
    try:
        plink_pca = run_plink_pca(pandora_config, n_pcs)
        alternatives["plink2"] = plink_pca
    except subprocess.CalledProcessError as e:
        logger.warning(fmt_message("Failed to run PLINK: " + str(e)))

    try:
        sklearn_pca = run_sklearn_pca(pandora_config, n_pcs)
        alternatives["scikit-learn"] = sklearn_pca
    except subprocess.CalledProcessError as e:
        logger.warning(fmt_message("Failed to run scikit-learn: " + str(e)))

    return alternatives


def get_n_clusters(pandora_config: PandoraConfig, empirical_pca: PCA):
    n_clusters_ckp = pandora_config.outdir / "kmeans.ckp"

    if n_clusters_ckp.exists() and not pandora_config.redo:
        n_clusters = int(open(n_clusters_ckp).readline())
    else:
        n_clusters = empirical_pca.get_optimal_n_clusters()
        n_clusters_ckp.open("w").write(str(n_clusters))
    logger.info(fmt_message(f"Optimal number of clusters determined to be: {n_clusters}"))

    return n_clusters


def compare_bootstrap_results(pandora_config: PandoraConfig, empirical_pca: PCA, bootstrap_pcas: List[PCA], n_clusters: int):
    # Compare Empirical <> Bootstraps
    bootstrap_similarities = []
    bootstrap_cluster_similarities = []

    for i, bootstrap_pca in enumerate(bootstrap_pcas):
        pca_comparison = PCAComparison(comparable=bootstrap_pca, reference=empirical_pca)
        similarity = pca_comparison.compare()
        bootstrap_similarities.append(similarity)

        clustering_score = pca_comparison.compare_clustering(n_clusters=n_clusters, weighted=True)
        bootstrap_cluster_similarities.append(clustering_score)

    # write similarities of all bootstraps to file
    with open(f"{pandora_config.outfile_prefix}.pandora.bootstrap.txt", "w") as f:
        output = [f"{i + 1}\t{sim}" for i, sim in enumerate(bootstrap_similarities)]
        f.write("\n".join(output))

    return bootstrap_similarities, bootstrap_cluster_similarities


def compare_alternative_tool_results(empirical_pca: PCA, alternative_pcas: Dict[str, PCA], n_clusters: int):
    alternative_pcas["smartPCA"] = empirical_pca

    tool_similarities = []
    tool_cluster_similarities = []

    pairwise = {}

    for pca1, pca2 in itertools.combinations(alternative_pcas.items(), r=2):
        # TODO: hier stimmt bei sklearn noch was mit den sample_ids nicht glaube ich
        name1, pca1 = pca1
        name2, pca2 = pca2

        pca_comparison = PCAComparison(comparable=pca1, reference=pca2)
        similarity = pca_comparison.compare()
        tool_similarities.append(similarity)

        cluster_similarity = pca_comparison.compare_clustering(n_clusters=n_clusters, weighted=True)
        tool_cluster_similarities.append(cluster_similarity)

        pairwise[f"{name1} <> {name2}"] = (similarity, cluster_similarity)

    return tool_similarities, tool_cluster_similarities, pairwise


def plot_empirical(pandora_config: PandoraConfig, empirical_pca: PCA, n_clusters: int):
    # TODO: make plotted PCs command line settable
    pc1 = 0
    pc2 = 1

    # plot with annotated populations
    empirical_pca.plot(
        pc1=pc1,
        pc2=pc2,
        annotation="population",
        outfile=pandora_config.plot_dir / "empirical_with_populations.pdf",
        redo=pandora_config.redo
    )

    # plot with annotated clusters
    empirical_pca.plot(
        pc1=pc1,
        pc2=pc2,
        annotation="cluster",
        n_clusters=n_clusters,
        outfile=pandora_config.plot_dir / "empirical_with_clusters.pdf",
        redo=pandora_config.redo
    )

    logger.info(fmt_message(f"Plotted empirical PCA."))


def plot_bootstraps(pandora_config: PandoraConfig, empirical_pca: PCA, bootstrap_pcas: List[PCA], n_clusters: int):
    # TODO: make plotted PCs command line settable
    # TODO: paralleles plotten
    pc1 = 0
    pc2 = 1

    for i, bootstrap_pca in enumerate(bootstrap_pcas):
        # TODO: populations so zu setzen funktioniert nicht mehr wenn man die outlier in smartpca raus schmeisst
        # -> muss einmal am anfang das mapping sampleID -> population speichern
        # bootstrap_pca.set_populations(empirical_pca.pca_data.population)

        # plot Bootstrap with annotated populations
        bootstrap_pca.plot(
            pc1=pc1,
            pc2=pc2,
            annotation="population",
            outfile=pandora_config.plot_dir / f"bootstrap_{i + 1}_with_populations.pca.pdf",
            redo=pandora_config.redo
        )

        # plot Bootstrap only with annotated clusters
        bootstrap_pca.plot(
            pc1=pc1,
            pc2=pc2,
            annotation="cluster",
            n_clusters=n_clusters,
            outfile=pandora_config.plot_dir / f"bootstrap_{i + 1}_with_clusters.pca.pdf",
            redo=pandora_config.redo
        )

        # Plot transformed bootstrap and empirical data jointly
        # for this, we first need to transform the empirical and bootstrap data
        standardized_empirical, transformed_bootstrap = transform_pca_to_reference(bootstrap_pca, empirical_pca)
        fig = standardized_empirical.plot(
            pc1=pc1,
            pc2=pc2,
            name="empirical (standardized)",
            marker_color="darkblue",
            marker_symbol="circle",
        )

        transformed_bootstrap.plot(
            pc1=pc1,
            pc2=pc2,
            name="bootstrap (transformed)",
            fig=fig,
            marker_color="orange",
            marker_symbol="star",
            outfile=pandora_config.plot_dir / f"bootstrap_{i + 1}_with_empirical.pca.pdf",
            redo=pandora_config.redo
        )

        logger.info(fmt_message(f"Plotted bootstrap PCA #{i + 1}"))


def plot_alternative_tools(pandora_config: PandoraConfig, empirical_pca: PCA, alternative_pcas: Dict[str, PCA], n_clusters: int):
    # TODO: make plotted PCs command line settable
    # TODO: paralleles plotten
    pc1 = 0
    pc2 = 1

    # Plot transformed alternative Tools and smartPCA data jointly
    # for this, we first need to transform the empirical and bootstrap data
    for name, pca in alternative_pcas.items():
        pca.plot(
            pc1=pc1,
            pc2=pc2,
            name=f"Transformed {name}",
            outfile=pandora_config.plot_dir / f"{name}.pca.pdf"
        )

        standardized_reference, transformed_alternative = transform_pca_to_reference(pca, empirical_pca)
        fig = standardized_reference.plot(
            pc1=pc1,
            pc2=pc2,
            name="SmartPCA (standardized)",
            marker_color="darkblue",
            marker_symbol="circle",
        )

        transformed_alternative.plot(
            pc1=pc1,
            pc2=pc2,
            name=f"{name} (transformed)",
            fig=fig,
            marker_color="orange",
            marker_symbol="star",
            outfile=pandora_config.plot_dir / f"{name}_with_smartpca.pca.pdf"
        )


def plot_results(pandora_config: PandoraConfig, empirical_pca: PCA, bootstrap_pcas: List[PCA], n_clusters: int):
    plot_empirical(pandora_config, empirical_pca, n_clusters)
    plot_bootstraps(pandora_config, empirical_pca, bootstrap_pcas, n_clusters)
