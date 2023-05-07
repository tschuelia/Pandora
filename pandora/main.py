import argparse
import datetime
import math
import sys

from pandora import __version__
from pandora.pandora import *

"""
TODO: 
- bootstopping
- check bei jedem PCA das geskippt wird ob die Anzahl PCs auch mit der bestimmten Anzahl PCs die verwendet werden sollten 
    übereinstimmt -> falls nein, neu berechnen 
"""


def get_header():
    return textwrap.dedent(
        f"Pandora version {__version__} released by The Exelixis Lab\n"
        f"Developed by: Julia Haag\n"
        f"Latest version: https://github.com/tschuelia/Pandora\n"
        f"Questions/problems/suggestions? Please open an issue on GitHub.\n",
    )


def argument_parser():
    parser = argparse.ArgumentParser(prog="Pandora", description="Command line parser for Pandora options.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the yaml config file to use for the Pandora analyses. "
             "Needs to be in valid yaml format. See the example file for guidance on how to configure your run."
    )

    return parser.parse_args()


def main():
    print(get_header())
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

    pandora_config = from_config(args.config)

    pandora_config.create_outdirs()
    open(pandora_config.logfile, "w").write(get_header())
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

    # Bootstrapped PCAs
    bootstrap_pcas = run_bootstrap_pcas(pandora_config, n_pcs=empirical_pca.n_pcs)

    # PCAs using alternative tools (= alternative PCA implementations, different numerics, ...)
    # To do so, we first filter all outliers that smartPCA detected
    if pandora_config.run_alternative:
        filter_outliers(pandora_config)
        logger.info(fmt_message("Running PCA using Plink and scikit-learn"))
        # first, convert the EIGENSOFT files to PLINK format
        eigen_to_plink(
            eigen_prefix=pandora_config.filtered_infile_prefix,
            plink_prefix=pandora_config.convertf_prefix,
            convertf=pandora_config.convertf
        )
        alternative_tool_pcas = run_alternative_pcas(pandora_config, n_pcs=empirical_pca.n_pcs)
        alternative_tool_pcas["smartPCA"] = empirical_pca

    # =======================================
    # Compare results
    # pairwise comparison between all bootstraps
    # =======================================
    logger.info(fmt_message(f"Comparing PCA results."))

    kmeans_k = get_kmeans_k(pandora_config, empirical_pca)

    bootstrap_similarities, bootstrap_cluster_similarities = compare_bootstrap_results(
        pandora_config, bootstrap_pcas, kmeans_k
    )

    # Compare Empirical <> alternative tools
    if pandora_config.run_alternative:
        tool_similarities, tool_cluster_similarities, pairwise_similarities = compare_alternative_tool_results(
            alternative_tool_pcas, kmeans_k)

    # =======================================
    # Plot results
    # =======================================
    if pandora_config.run_plotting:
        plot_results(
            pandora_config,
            empirical_pca,
            bootstrap_pcas,
            kmeans_k
        )

        if pandora_config.run_alternative:
            plot_alternative_tools(
                pandora_config,
                alternative_tool_pcas,
                kmeans_k
            )

    logger.info("\n\n========= PANDORA RESULTS =========")
    logger.info(f"> Input dataset: {pandora_config.infile_prefix}")
    logger.info(f"> Number of Bootstrap replicates computed: {pandora_config.n_bootstraps}")
    logger.info(f"> Number of PCs required to explain at least {pandora_config.variance_cutoff}% variance: {empirical_pca.n_pcs}")
    logger.info(f"> Optimal number of clusters: {kmeans_k}")
    logger.info("\n------------------")
    logger.info("Bootstrapping Similarity")
    logger.info("------------------")
    logger.info(f"PCA: {round(np.mean(bootstrap_similarities), 2)} ± {round(np.std(bootstrap_similarities), 2)}")
    logger.info(f"K-Means clustering: {round(np.mean(bootstrap_cluster_similarities), 2)} ± {round(np.std(bootstrap_cluster_similarities), 2)}")

    if pandora_config.run_alternative:
        logger.info("\n------------------")
        logger.info("Alternative Tools Similarity")
        logger.info("------------------")
        logger.info(f"PCA: {round(np.mean(tool_similarities), 2)} ± {round(np.std(tool_similarities), 2)}")
        logger.info(f"K-Means clustering: {round(np.mean(tool_cluster_similarities), 2)} ± {round(np.std(tool_cluster_similarities), 2)}")

    total_runtime = math.ceil(time.perf_counter() - SCRIPT_START)
    logger.info(f"\nTotal runtime: {datetime.timedelta(seconds=total_runtime)} ({total_runtime} seconds)")

    # For debugging now:
    if pandora_config.run_alternative:
        for tool, (pca, cluster) in pairwise_similarities.items():
            print(tool, round(pca, 2), round(cluster, 2))


if __name__ == "__main__":
    main()
