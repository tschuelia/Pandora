import dataclasses
import itertools

from pandora.bootstrapping import create_bootstrap_pcas
from pandora.converter import eigen_to_plink, plink_to_bplink
from pandora.pca import *


@dataclasses.dataclass
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

    def create_outdirs(self):
        if self.run_bootstrapping:
            self.convertf.mkdir(exist_ok=True, parents=True)
            self.bootstrap_dir.mkdir(exist_ok=True, parents=True)
        if self.run_alternative:
            self.convertf_dir_binary.mkdir(exist_ok=True, parents=True)
            self.alternative_tools_dir.mkdir(exist_ok=True, parents=True)
        if self.run_plotting:
            self.plot_dir.mkdir(exist_ok=True, parents=True)


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
        n_pcs=n_pcs
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

    run_sklearn(
        infile_prefix=pandora_config.convertf_prefix,
        outfile_prefix=sklearn_prefix,
        n_pcs=n_pcs,
        redo=pandora_config.redo
    )
    sklearn_pca = from_sklearn(
        fitted_pca_model=pathlib.Path(f"{sklearn_prefix}.pca.sklearn.model"),
        pca_data=pathlib.Path(f"{sklearn_prefix}.pca.output.npy"),
        sample_ids=pathlib.Path(f"{sklearn_prefix}.pca.sample.ids")
    )

    return sklearn_pca


def run_alternative_pcas(pandora_config: PandoraConfig, n_pcs: int):
    plink_pca = run_plink_pca(pandora_config, n_pcs)
    sklearn_pca = run_sklearn_pca(pandora_config, n_pcs)

    return {
        "plink2": plink_pca,
        "scikit-learn": sklearn_pca
    }


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
        similarity = bootstrap_pca.compare(other=empirical_pca)
        bootstrap_similarities.append(similarity)

        clustering_score = bootstrap_pca.compare_clustering(other=empirical_pca, n_clusters=n_clusters, weighted=False)
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
        name1, pca1 = pca1
        name2, pca2 = pca2

        similarity = pca1.compare(other=pca2)
        tool_similarities.append(similarity)

        cluster_similarity = pca1.compare_clustering(other=pca2, n_clusters=n_clusters, weighted=False)
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
    pc1 = 0
    pc2 = 1

    for i, bootstrap_pca in enumerate(bootstrap_pcas):
        bootstrap_pca.set_populations(empirical_pca.pca_data.population)

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
        transformed_bootstrap, scaled_empirical = transform_pca_to_reference(bootstrap_pca, empirical_pca)
        fig = scaled_empirical.plot(
            pc1=pc1,
            pc2=pc2,
            name="Scaled empirical",
            marker_color="darkblue",
            marker_symbol="circle",
        )

        transformed_bootstrap.plot(
            pc1=pc1,
            pc2=pc2,
            name="Transformed Bootstrap",
            fig=fig,
            marker_color="orange",
            marker_symbol="star",
            outfile=pandora_config.plot_dir / f"bootstrap_{i + 1}_with_empirical.pca.pdf",
            redo=pandora_config.redo
        )

        logger.info(fmt_message(f"Plotted bootstrap PCA #{i + 1}"))


def plot_alternative_tools(pandora_config: PandoraConfig, empirical_pca: PCA, alternative_pcas: Dict[str, PCA], n_clusters: int):
    # TODO: make plotted PCs command line settable
    pc1 = 0
    pc2 = 1

    # Plot transformed alternative Tools and smartPCA data jointly
    # for this, we first need to transform the empirical and bootstrap data
    for name, pca in alternative_pcas.items():
        transformed_alternative, scaled_empirical = transform_pca_to_reference(pca, empirical_pca)
        fig = scaled_empirical.plot(
            pc1=pc1,
            pc2=pc2,
            name="Scaled SmartPCA",
            marker_color="darkblue",
            marker_symbol="circle",
        )

        transformed_alternative.plot(
            pc1=pc1,
            pc2=pc2,
            name=f"Transformed {name}",
            fig=fig,
            marker_color="orange",
            marker_symbol="star",
            outfile=pandora_config.plot_dir / f"{name}_with_smartpca.pca.pdf"
        )


def plot_results(pandora_config: PandoraConfig, empirical_pca: PCA, bootstrap_pcas: List[PCA], n_clusters: int):
    plot_empirical(pandora_config, empirical_pca, n_clusters)
    plot_bootstraps(pandora_config, empirical_pca, bootstrap_pcas, n_clusters)
