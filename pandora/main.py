"""
Numpy uses OMP and BLAS for its linear algebra under the hood. Per default, both libraries use all cores on the machine.
Since we are doing a pairwise comparison of all bootstrap replicates, for a substantial amount of time Pandora would
require all cores on the machine, which is not an option when working on shared servers/clusters.
Therefore, we set the OMP and BLAS threads manually to 1 to prevent this.
Setting these variables needs to happen before the first import of numpy.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import datetime
import sys
import math

from pandora import __version__
from pandora.pandora import *
from pandora.logger import *


def get_header():
    return textwrap.dedent(
        f"Pandora version {__version__} released by The Exelixis Lab\n"
        f"Developed by: Julia Haag\n"
        f"Latest version: https://github.com/tschuelia/Pandora\n"
        f"Questions/problems/suggestions? Please open an issue on GitHub.\n",
    )


def argument_parser():
    parser = argparse.ArgumentParser(
        prog="Pandora", description="Command line parser for Pandora options."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=FilePath,
        required=True,
        help="Path to the yaml config file to use for the Pandora analyses. "
        "Needs to be in valid yaml format. See the example file for guidance on how to configure your run.",
    )

    return parser.parse_args()


def main():
    print(get_header())
    args = argument_parser()

    """
    Pandora main program
    """
    # =======================================
    # initialize options and directories
    # =======================================
    pandora_config = pandora_config_from_configfile(args.config)

    # TODO: compare pandora_config with potentially existing config file and warn user if settings changed but redo was not set

    # create the required output directories based on the analysis specified
    pandora_config.create_result_dirs()
    pandora_config.result_file.unlink(missing_ok=True)

    # set the log verbosity according to the pandora config
    logger.setLevel(pandora_config.loglevel)

    # hook up the logfile to the logger to also store the output
    pandora_config.pandora_logfile.open(mode="w").write(get_header())
    logger.addHandler(logging.FileHandler(pandora_config.pandora_logfile))

    # log the run configuration
    _arguments_str = [
        f"{k}: {v}" for k, v in pandora_config.get_configuration().items()
    ]
    _command_line = " ".join(sys.argv)

    logger.info("--------- PANDORA CONFIGURATION ---------")
    logger.info("\n".join(_arguments_str))
    logger.info(f"\nCommand line: {_command_line}")

    # store pandora config in a verbose config file for reproducibility
    pandora_config.save_config()

    # =======================================
    # start computation
    # =======================================
    logger.info("\n--------- STARTING COMPUTATION ---------")

    # before starting any computation, make sure that we have the input data in EIGENSTRAT format
    # we need this format for bootstrapping and smartPCA
    if pandora_config.file_format != FileFormat.EIGENSTRAT:
        logger.info(
            fmt_message(
                f"Converting dataset from {pandora_config.file_format.value} to {FileFormat.EIGENSTRAT.value}"
            )
        )
        convert_to_eigenstrat_format(pandora_config)

    # initialize empty Pandora object that keeps track of all results
    pandora_results = Pandora(pandora_config)

    # TODO: implement alternative MDS analysis
    # Run PCA on the input dataset without any bootstrapping
    logger.info(fmt_message("Running SmartPCA on the input dataset."))
    pandora_results.do_pca()

    if pandora_config.plot_results:
        logger.info(fmt_message("Plotting SmartPCA results for the input dataset."))
        pandora_results.plot_dataset()

    if pandora_config.do_bootstrapping:
        # Bootstrapped PCAs
        logger.info(
            fmt_message(
                f"Drawing {pandora_config.n_bootstraps} bootstrapped datasets and running SmartPCA."
            )
        )
        pandora_results.bootstrap_pcas()

        # =======================================
        # Compare results
        # pairwise comparison between all bootstraps
        # =======================================
        logger.info(fmt_message(f"Comparing bootstrap PCA results."))
        pandora_results.compare_bootstrap_similarity()

        if pandora_config.plot_results:
            logger.info(fmt_message(f"Plotting bootstrap PCA results."))
            pandora_results.plot_bootstraps()
            pandora_results.plot_sample_support_values()

            if pandora_config.pca_populations is not None:
                pandora_results.plot_sample_support_values(projected_samples_only=True)

    logger.info("\n\n========= PANDORA RESULTS =========")
    logger.info(f"> Input dataset: {pandora_config.dataset_prefix.absolute()}")

    if pandora_config.do_bootstrapping:
        pandora_results.log_and_save_bootstrap_results()
        pandora_results.log_and_save_sample_support_values()

    logger.info(
        textwrap.dedent(
            """
            ------------------
            Result Files
            ------------------"""
        )
    )
    logger.info(f"> Pandora results: {pandora_config.result_file.absolute()}")

    if pandora_config.do_bootstrapping:
        logger.info(
            f"> Pairwise bootstrap similarities: {pandora_config.pairwise_bootstrap_result_file.absolute()}"
        )
        logger.info(
            f"> Sample Support values: {pandora_config.sample_support_values_file.absolute()}"
        )

    if pandora_config.pca_populations is not None:
        logger.info(
            f"> Projected Sample Support values: {pandora_config.sample_support_values_projected_samples_file.absolute()}"
        )

    if pandora_config.plot_results:
        logger.info(
            f"> All plots saved in directory: {pandora_config.plot_dir.absolute()}"
        )

    total_runtime = math.ceil(time.perf_counter() - SCRIPT_START)
    logger.info(
        f"\nTotal runtime: {datetime.timedelta(seconds=total_runtime)} ({total_runtime} seconds)"
    )


if __name__ == "__main__":
    main()
