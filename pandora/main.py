import argparse
import datetime
import multiprocessing
import time
import sys

from pandora import __version__
from pandora.bootstrapping import create_bootstrap_pcas
from pandora.converter import eigen_to_plink
from pandora.custom_types import *
from pandora.logger import logger, fmt_message, SCRIPT_START
from pandora.pca import determine_number_of_pcs, transform_pca_to_reference


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
    Command line parser 
    """
    parser = argparse.ArgumentParser(prog="Pandora", description="Command line parser for Pandora options.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Prefix of the input files. Pandora will look for files called <input>.* so make sure all files have the same prefix."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Directory where to write the output to."
    )
    parser.add_argument(
        "-bs",
        "--bootstraps",
        type=int,
        required=False,
        default=100,
        help="Maximum number of bootstrap replicates to compute (default = TODO)."
    )
    parser.add_argument(
        "--disableBootstopping",
        action="store_true",
        help="If set, disables the automatic bootstopping check and infers the maximum number of bootstrap replicates."
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        required=False,
        default=multiprocessing.cpu_count(),
        help="Number of threads to use for computation (default is number of cores)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=int(time.time()),
        help="Seed for random number generator (default is current time)."
    )
    parser.add_argument(
        "--smartpca",
        type=str,
        required=False,
        default="smartpca",
        help="Optional path to the smartPCA executable. Per default, Pandora will use 'smartpca'. "
             "Specify a path if smartPCA is not installed system-wide on your machine."
    )
    parser.add_argument(
        "--convertf",
        type=str,
        required=False,
        default="convertf",
        help="Optional path to the convertf executable. Per default, Pandora will use 'convertf'. "
             "Specify a path if convertf is not installed system-wide on your machine."
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="If set, reruns the analysis. Careful: this will overwrite existing results!"
    )
    parser.add_argument(
        "--varianceCutoff",
        type=int,
        default=95,
        help="Cutoff to use when determining the number of PCs required. "
             "Pandora will automatically find the number of PCs required to explain at least <varianceCutoff>%% variance in the data."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, plots the individual PCAs with the resulting K-Means clusters annotated."
    )

    args = parser.parse_args()

    """
    Pandora main program
    """
    # initialize options and directories
    infile_prefix = pathlib.Path(args.input)
    dataset = infile_prefix.name

    outfile_base = pathlib.Path(args.output)
    outfile_prefix = outfile_base / dataset
    convertf_dir = outfile_base / "convertf"
    bootstrap_dir = outfile_base / "bootstrap"

    seed = args.seed
    n_bootstraps = args.bootstraps if args.bootstraps else 100  # TODO: what number should we use as default?
    disable_bootstopping = args.disableBootstopping  # TODO: implement bootstopping
    n_threads = args.threads
    redo = args.redo
    variance_cutoff = args.varianceCutoff
    plot_pcas = args.plot

    smartpca = args.smartpca
    convertf = args.convertf

    _arguments_str = [f"{k}: {v}" for k, v in vars(args).items()]
    _command_line = " ".join(sys.argv)

    logger.info("--------- PANDORA CONFIGURATION ---------")
    logger.info("\n".join(_arguments_str))
    logger.info(f"\nCommand line: {_command_line}")

    # start computation
    logger.info("\n--------- STARTING COMPUTATION ---------")

    outfile_base.mkdir(exist_ok=True, parents=True)

    empirical_pca = determine_number_of_pcs(
        infile_prefix=infile_prefix,
        outfile_prefix=outfile_prefix,
        smartpca=smartpca,
        # TODO: what is a meaningful variance cutoff?
        explained_variance_cutoff=variance_cutoff / 100,
        redo=redo,
    )

    logger.info(fmt_message("Converting Input files to PLINK format for bootstrapping."))

    convertf_dir.mkdir(exist_ok=True)
    convert_prefix = convertf_dir / dataset

    eigen_to_plink(
        eigen_prefix=infile_prefix, plink_prefix=convert_prefix, convertf=convertf
    )

    # TODO: implement bootstopping
    logger.info(fmt_message(f"Drawing {n_bootstraps} bootstrapped datasets, converting them to EIGEN format and running smartpca."))

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

    logger.info(fmt_message(f"Comparing PCA results."))
    # TODO: make plotted PCs command line settable
    pc1 = 0
    pc2 = 1

    if plot_pcas:

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

        if plot_pcas:
            # plot Bootstrap with annotated populations
            bootstrap_pca.set_populations(empirical_pca.pca_data.population)

            fig = bootstrap_pca.plot(
                pc1=pc1,
                pc2=pc2,
                annotation="population",
            )

            fp = bootstrap_dir / f"bootstrap_{i + 1}_with_populations.pca.pdf"
            fig.write_image(fp)

            # plot Bootstrap only with annotated clusters
            fig = bootstrap_pca.plot(
                pc1=pc1,
                pc2=pc2,
                annotation="cluster",
                n_clusters=n_clusters
            )
            fp = bootstrap_dir / f"bootstrap_{i + 1}_with_clusters.pca.pdf"
            fig.write_image(fp)

            # Plot transformed bootstrap and empirical data jointly
            # for this, we first need to transform the empirical and bootstrap data
            transformed_bootstrap, scaled_empirical = transform_pca_to_reference(bootstrap_pca, empirical_pca)
            fig = scaled_empirical.plot(
                pc1=pc1,
                pc2=pc2,
                name="Scaled empirical"
            )

            fig = transformed_bootstrap.plot(
                pc1=pc1,
                pc2=pc2,
                name="Transformed Bootstrap",
                fig=fig
            )

            fp = bootstrap_dir / f"bootstrap_{i + 1}_with_empirical.pca.pdf"
            fig.write_image(fp)

            logger.info(fmt_message(f"Plotted bootstrap PCA #{i + 1}"))

        similarity = bootstrap_pca.compare(other=empirical_pca)
        similarities.append(similarity)

        scores = bootstrap_pca.compare_clustering(other=empirical_pca, n_clusters=n_clusters)

        if clustering_scores is None:
            clustering_scores = dict([(k, [v]) for k, v in scores.items()])
        else:
            for k, v in scores.items():
                clustering_scores[k].append(v)

    # write similarities of all bootstraps to file
    with open(f"{outfile_prefix}.pandora.txt", "w") as f:
        output = [f"{i + 1}\t{sim}" for i, sim in enumerate(similarities)]
        f.write("\n".join(output))

    # TODO: also log the output into a log file

    logger.info("--------- PANDORA RESULTS ---------")
    logger.info("PCA:")
    logger.info(f"- number of PCs required to explain at least {variance_cutoff}% variance: {empirical_pca.n_pcs}")
    logger.info(f"=> uncertainty: {round(np.mean(similarities), 2)} ± {round(np.std(similarities), 2)}")
    logger.info("K-Means clustering:")
    logger.info("- uncertainty: TODO: select the best measure")

    SCRIPT_END = time.perf_counter()
    total_runtime = SCRIPT_END - SCRIPT_START
    logger.info(f"\nTotal runtime: {datetime.timedelta(seconds=total_runtime)} ({total_runtime} seconds)")

    print("Clustering measures ", [(k, round(np.mean(v), 2)) for k, v in clustering_scores.items()])


if __name__ == "__main__":
    main()
