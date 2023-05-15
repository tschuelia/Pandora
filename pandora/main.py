import argparse
import datetime
import sys

from pandora import __version__
from pandora.pandora import *
from pandora.logger import *

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
    pandora_config = pandora_config_from_configfile(args.config)

    # store pandora config in a verbose config file for reproducibility
    pandora_config.to_configfile()

    # next, create the required output directories based on the analysis specified
    pandora_config.create_outdirs()

    # set the log verbosity according to the pandora config
    logger.setLevel(pandora_config.loglevel)

    # hook up the logfile to the logger to also store the output
    pandora_config.logfile.open(mode="w").write(get_header())
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

    # initialize empty PandoraResults object that keeps track of all results
    pandora_results = Pandora(
        pandora_config=pandora_config
    )

    if pandora_config.run_bootstrapping:
        # Empirical PCA using smartPCA and no bootstrapping
        pandora_results.run_empirical_pca()

        # Bootstrapped PCAs
        logger.info(fmt_message(f"Drawing {pandora_config.n_bootstraps} bootstrapped datasets."))
        pandora_results.run_bootstrap_pcas()

        # =======================================
        # Compare results
        # pairwise comparison between all bootstraps
        # =======================================
        logger.info(fmt_message(f"Comparing bootstrap PCA results."))
        pandora_results.compare_bootstrap_results()

        # =======================================
        # Plot results
        # =======================================
        if pandora_config.run_plotting:
            pandora_results.plot_results()

    logger.info("\n\n========= PANDORA RESULTS =========")
    logger.info(f"> Input dataset: {pandora_config.infile_prefix.absolute()}")

    pandora_results.results_file.unlink(missing_ok=True)

    if pandora_config.run_bootstrapping:
        pandora_results.log_and_save_bootstrap_results()

    logger.info(
        textwrap.dedent(
            """
            ------------------
            Result Files
            ------------------"""
        )
    )
    logger.info(f"> Pandora results: {pandora_results.results_file.absolute()}")

    if pandora_config.run_bootstrapping:
        logger.info(f"> Pairwise bootstrap: {pandora_results.pairwise_bootstrap_results_file.absolute()}")

    total_runtime = math.ceil(time.perf_counter() - SCRIPT_START)
    logger.info(f"\nTotal runtime: {datetime.timedelta(seconds=total_runtime)} ({total_runtime} seconds)")


if __name__ == "__main__":
    main()
