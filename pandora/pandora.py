import dataclasses
import itertools
import multiprocessing
import pathlib
import shutil

import yaml

from pandora.bootstrapping import create_bootstrap_pcas
from pandora.converter import *
from pandora.logger import *
from pandora.pca import *
from pandora.pca_comparison import PCAComparison, match_and_transform
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
    k_clusters: int

    smartpca: Executable
    convertf: Executable
    plink2: Executable

    run_bootstrapping: bool
    run_alternative: bool
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
        k_clusters=config_data.get("kClusters", None),
        smartpca=config_data.get("smartpca", "smartpca"),
        convertf=config_data.get("convertf", "convertf"),
        plink2=config_data.get("plink2", "plink2"),
        run_bootstrapping=uncertainty_analyses.get("bootstrap", True),
        run_alternative=uncertainty_analyses.get("alternativeTools", True),
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


def run_plink_pca(pandora_config: PandoraConfig, n_pcs: int):
    # eigen_to_plink(
    #     eigen_prefix=pandora_config.filtered_infile_prefix,
    #     plink_prefix=
    # )

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


def compare_alternative_tool_results(alternative_tools: Dict[str, PCA], kmeans_k: int):
    tool_similarities = []
    tool_cluster_similarities = []

    pairwise = {}

    for pca1, pca2 in itertools.combinations(alternative_tools.items(), r=2):
        # TODO: hier stimmt bei sklearn noch was mit den sample_ids nicht glaube ich
        name1, pca1 = pca1
        name2, pca2 = pca2

        pca_comparison = PCAComparison(comparable=pca1, reference=pca2)
        similarity = pca_comparison.compare()
        tool_similarities.append(similarity)

        cluster_similarity = pca_comparison.compare_clustering(kmeans_k=kmeans_k)
        tool_cluster_similarities.append(cluster_similarity)

        pairwise[f"{name1} <> {name2}"] = (similarity, cluster_similarity)

    return tool_similarities, tool_cluster_similarities, pairwise


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


def plot_alternative_tools(pandora_config: PandoraConfig, alternative_tools: Dict[str, PCA], kmeans_k: int):
    for name, pca in alternative_tools.items():
        plot_pca(pandora_config, pca, kmeans_k, name)


def plot_results(pandora_config: PandoraConfig, empirical_pca: PCA, bootstrap_pcas: List[PCA], kmeans_k: int):
    plot_pca(pandora_config, empirical_pca, kmeans_k, "empirical")
    for i, bootstrap_pca in enumerate(bootstrap_pcas):
        plot_pca(pandora_config, bootstrap_pca, kmeans_k, f"bootstrap_{i + 1}")


def filter_outliers(pandora_config: PandoraConfig):
    # TODO: hier erstmal das empirical PCA anschauen und die outlier rausfiltern für einen besseren vergleich
    smartpca_outlier_file = pathlib.Path(f"{pandora_config.outfile_prefix}.outlier")

    if not smartpca_outlier_file.exists():
        raise RuntimeError("SmartPCA outlier file does not exist. Run smartPCA first to detect outliers.")

    smartpca_outlier = []
    for line in smartpca_outlier_file.open():
        # REMOVED outlier I0114 iter 4 evec 8 sigmage -6.697 pop: Control
        _, _, outlier_id, *_ = line.split()
        smartpca_outlier.append(outlier_id.strip())

    # filter .ind file
    ind_file = pathlib.Path(f"{pandora_config.infile_prefix}.ind")
    new_ind_file = pathlib.Path(f"{pandora_config.filtered_infile_prefix}.ind")

    new_ind_data = []
    inlier_indices = []

    for i, ind_line in enumerate(ind_file.open()):
        sample_id, *_ = ind_line.strip().split()
        if sample_id.strip() not in smartpca_outlier:
            new_ind_data.append(ind_line.strip())
            inlier_indices.append(i)

    new_ind_file.open(mode="w").write("\n".join(new_ind_data))

    # filter .geno file
    geno_file = pathlib.Path(f"{pandora_config.infile_prefix}.geno")
    new_geno_file = pathlib.Path(f"{pandora_config.filtered_infile_prefix}.geno")

    new_geno_data = []
    for i, line in enumerate(geno_file.open()):
        snps = list(line.strip())
        # columns = individuals
        # => use only inlier_indices
        snps = [snps[idx] for idx in inlier_indices]
        new_geno_data.append("".join(snps))

    new_geno_file.open(mode="w").write("\n".join(new_geno_data))

    # copy .snp file
    snp_file = pathlib.Path(f"{pandora_config.infile_prefix}.snp")
    new_snp_file = pathlib.Path(f"{pandora_config.filtered_infile_prefix}.snp")

    shutil.copy(snp_file, new_snp_file)
