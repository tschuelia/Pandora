import os

from pandora import __version__
from pandora.bootstrapping import create_bootstrap_pcas
from pandora.converter import eigen_to_plink
from pandora.custom_types import *
from pandora.logger import logger, fmt_message
from pandora.pca import (
    determine_number_of_pcs,
)


def print_header():
    logger.info(
        f"Pandora version {__version__} released by The Exelixis Lab\n"
        f"Developed by: Julia Haag\n"
        f"Latest version: https://github.com/tschuelia/Pandora\n"
        f"Questions/problems/suggestions? Please open an issue on GitHub.\n",
    )


def main():
    print_header()
    """
    TODO: CL arguments
    - type of analysis:
        - bootstrapping on SNP level (bool)
        - bootstrapping on individual level (bool)
        - clustering uncertainty (bool)
        - all (= all 4 analysis)
        - n bootstrap samples (default = 100 / bootstopping criterion) (int)
        - n threads (int)
        - save results (bool)
        - data format (string) -> implement conversions of the most common data formats
        - seed (int)
    """

    """
    begin config
    """
    dataset = "EastAsian_HO"
    n_bootstraps = 3
    n_threads = 2
    seed = 0
    redo = False

    if "/Users/julia" in os.getcwd():
        infile_prefix = (
            pathlib.Path("/Users/julia/Desktop/Promotion/ADNA_Popgen/input")
            / dataset
            / dataset
        )

        outfile_base = (
            pathlib.Path("/Users/julia/Desktop/Promotion/ADNA_Popgen/Pandora/results")
            / dataset
        )

        smartpca = "/Users/julia/micromamba/envs/pca_micromamba/bin/smartpca"
        convertf = "/Users/julia/micromamba/envs/pca_micromamba/bin/convertf"
    else:
        infile_prefix = (
                pathlib.Path("/hits/fast/cme/schmidja/popgen_adna/input")
                / dataset
                / dataset
        )

        outfile_base = pathlib.Path("/hits/fast/cme/schmidja/popgen_adna/pca_uncertainty/results") / dataset

        smartpca = "/home/schmidja/miniconda3/envs/pca/bin/smartpca"
        convertf = "/home/schmidja/miniconda3/envs/pca/bin/convertf"
    """
    End config
    """

    outfile_base.mkdir(exist_ok=True, parents=True)
    outfile_prefix = outfile_base / dataset

    empirical_pca = determine_number_of_pcs(
        infile_prefix=infile_prefix,
        outfile_prefix=outfile_prefix,
        smartpca=smartpca,
        # TODO: what is a meaningful variance cutoff?
        explained_variance_cutoff=0.95,
        redo=redo,
    )

    logger.info(fmt_message("Converting Input files to PLINK format for bootstrapping."))

    convertf_dir = outfile_base / "convertf"
    convertf_dir.mkdir(exist_ok=True)
    convert_prefix = convertf_dir / dataset

    eigen_to_plink(
        eigen_prefix=infile_prefix, plink_prefix=convert_prefix, convertf=convertf
    )

    logger.info(fmt_message(f"Drawing {n_bootstraps} bootstrapped datasets, converting them to EIGEN format and running smartpca."))

    bootstrap_dir = outfile_base / "bootstrap"
    bootstrap_dir.mkdir(exist_ok=True)

    bootstrap_pcas = create_bootstrap_pcas(
        infile_prefix=convert_prefix,
        bootstrap_outdir=bootstrap_dir,
        convertf=convertf,
        smartpca=smartpca,
        n_bootstraps=n_bootstraps,
        seed=seed,
        n_pcs=empirical_pca.n_pcs,
        n_threads=n_threads,
        redo=redo
    )

    logger.info(fmt_message(f"Plotting and comparing PCA results."))

    pc1 = 0
    pc2 = 1

    fig = empirical_pca.plot(
        pc1=pc1,
        pc2=pc2,
        plot_populations=True,
    )
    fp = f"{outfile_prefix}.pdf"
    fig.write_image(fp)
    logger.info(fmt_message(f"Plotted empirical PCA: {fp}"))

    n_clusters = empirical_pca.get_optimal_n_clusters()
    logger.info(fmt_message(f"Optimal number of clusters determined to be: {n_clusters}"))
    similarities = []
    clustering_scores = None

    for i, bootstrap_pca in enumerate(bootstrap_pcas):
        bootstrap_pca.set_populations(empirical_pca.pca_data.population)

        fig = bootstrap_pca.plot(
            pc1=pc1,
            pc2=pc2,
            plot_populations=True,
        )

        fp = bootstrap_dir / f"bootstrap_{i + 1}.pca.pdf"
        fig.write_image(fp)
        logger.info(fmt_message(f"Plotted bootstrap PCA #{i + 1}: {fp}"))

        similarity = bootstrap_pca.compare(other=empirical_pca, normalize=True)
        similarities.append(similarity)

        scores = bootstrap_pca.compare_clustering(other=empirical_pca, n_clusters=n_clusters)

        if clustering_scores is None:
            clustering_scores = dict([(k, [v]) for k, v in scores.items()])
        else:
            for k, v in scores.items():
                clustering_scores[k].append(v)

    print(f"PCA similarity: {round(np.mean(similarities), 2)} ± {round(np.std(similarities), 2)}")
    print("Clustering measures ", [(k, round(np.mean(v), 2)) for k, v in clustering_scores.items()])


if __name__ == "__main__":
    main()
