import argparse
import datetime
import logging
import multiprocessing
import time
import sys

from pandora import __version__
from pandora.logger import *
from pandora.pandora import *

"""
TODO: 
- bootstopping
- check bei jedem PCA das geskippt wird ob die Anzahl PCs auch mit der bestimmten Anzahl PCs die verwendet werden sollten 
    übereinstimmt -> falls nein, neu berechnen 
- % explained variance sinnlos wenn man 2 PCs gegeneinander plottet
- config file statt tausend command line optionen erlauben (und bei einem run das config file immer in den results ordner packen für reproduzierbarkeit)
"""


def print_header():
    logger.info(
        f"Pandora version {__version__} released by The Exelixis Lab\n"
        f"Developed by: Julia Haag\n"
        f"Latest version: https://github.com/tschuelia/Pandora\n"
        f"Questions/problems/suggestions? Please open an issue on GitHub.\n",
    )


def argument_parser():
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
        "--plink2",
        type=str,
        required=False,
        default="plink2",
        help="Optional path to the plink2 executable. Per default, Pandora will use 'plink2'. "
             "Specify a path if plink2 is not installed system-wide on your machine."
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
             "Pandora will automatically find the number of PCs required to explain at least "
             "<varianceCutoff>%% variance in the data. Default is 95%."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, plots the individual PCAs with the resulting K-Means clusters annotated."
    )

    return parser.parse_args()


def main():
    print_header()
    """
    Command line parser 
    """
    args = argument_parser()

    """
    Pandora main program
    """
    # =======================================
    # initialize options and directories
    # =======================================

    pandora_config = PandoraConfig(
        infile_prefix=pathlib.Path(args.input),
        outdir=pathlib.Path(args.output),
        n_bootstraps=args.bootstraps,
        threads=args.threads,
        seed=args.seed,
        redo=args.redo,
        variance_cutoff=args.varianceCutoff,
        smartpca=args.smartpca,
        convertf=args.convertf,
        plink2=args.plink2,
        run_bootstrapping=True,
        run_alternative=True,
        run_plotting=args.plot,
    )

    # TODO: manuell noch den heaeder in das logfile schreiben (wegen Version etc.)
    pandora_config.create_outdirs()
    logger.addHandler(logging.FileHandler(pandora_config.logfile))

    _arguments_str = [f"{k}: {v}" for k, v in pandora_config.get_configuration().items()]
    _command_line = " ".join(sys.argv)

    logger.info("--------- PANDORA CONFIGURATION ---------")
    logger.info("\n".join(_arguments_str))
    logger.info(f"\nCommand line: {_command_line}")

    # =======================================
    # start computation
    # =======================================
    logger.info("\n--------- STARTING COMPUTATION ---------")

    # Empirical PCA using smartPCA and no bootstrapping
    empirical_pca = run_empirical_pca(pandora_config)

    # Bootstrapped PCA
    bootstrap_pcas = run_bootstrap_pcas(pandora_config, n_pcs=empirical_pca.n_pcs)

    # PCA with alternative tools
    logger.info(fmt_message("Running PCA using Plink and scikit-learn"))
    alternative_tool_pcas = run_alternative_pcas(pandora_config, n_pcs=empirical_pca.n_pcs)

    # =======================================
    # Compare results
    # =======================================
    logger.info(fmt_message(f"Comparing PCA results."))

    n_clusters = get_n_clusters(pandora_config, empirical_pca)

    bootstrap_similarities, bootstrap_cluster_similarities = compare_bootstrap_results(
        pandora_config, empirical_pca, bootstrap_pcas, n_clusters
    )

    # Compare Empirical <> alternative tools
    tool_similarities, tool_cluster_similarities, pairwise_similarities = compare_alternative_tool_results(
        empirical_pca, alternative_tool_pcas, n_clusters)

    # =======================================
    # Plot results
    # =======================================
    if pandora_config.run_plotting:
        plot_results(
            pandora_config,
            empirical_pca,
            bootstrap_pcas,
            n_clusters
        )

        plot_alternative_tools(
            pandora_config,
            empirical_pca,
            alternative_tool_pcas,
            n_clusters
        )

    logger.info("\n\n========= PANDORA RESULTS =========")
    logger.info(f"> Input dataset: {pandora_config.infile_prefix}")
    logger.info(f"> Number of Bootstrap replicates computed: {pandora_config.n_bootstraps}")
    logger.info(f"> Number of PCs required to explain at least {pandora_config.variance_cutoff}% variance: {empirical_pca.n_pcs}")
    logger.info(f"> Optimal number of clusters: {n_clusters}")
    logger.info("\n------------------")
    logger.info("Bootstrapping Similarity")
    logger.info("------------------")
    logger.info(f"PCA: {round(np.mean(bootstrap_similarities), 2)} ± {round(np.std(bootstrap_similarities), 2)}")
    logger.info(f"K-Means clustering: {round(np.mean(bootstrap_cluster_similarities), 2)} ± {round(np.std(bootstrap_cluster_similarities), 2)}")

    logger.info("\n------------------")
    logger.info("Alternative Tools Similarity")
    logger.info("------------------")
    logger.info(f"PCA: {round(np.mean(tool_similarities), 2)} ± {round(np.std(tool_similarities), 2)}")
    logger.info(f"K-Means clustering: {round(np.mean(tool_cluster_similarities), 2)} ± {round(np.std(tool_cluster_similarities), 2)}")

    total_runtime = math.ceil(time.perf_counter() - SCRIPT_START)
    logger.info(f"\nTotal runtime: {datetime.timedelta(seconds=total_runtime)} ({total_runtime} seconds)")

    # For debugging now:
    for tool, (pca, cluster) in pairwise_similarities:
        print(tool, round(pca, 2), round(cluster, 2))


if __name__ == "__main__":
    main()
