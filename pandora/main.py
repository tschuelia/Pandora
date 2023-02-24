from pandora import __version__
from pandora.bootstrapping import create_bootstrap_datasets
from pandora.converter import eigen_to_plink, parallel_plink_to_eigen
from pandora.custom_types import *
from pandora.logger import logger, fmt_message
from pandora.pca import run_pca_original_data_and_get_number_of_pcs, run_pca_bootstapped_data


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
    dataset = "HumanOriginsPublic2068"
    n_bootstraps = 2
    n_threads = 2
    seed = 0

    infile_prefix = (
        pathlib.Path("/Users/julia/Desktop/Promotion/ADNA_Popgen/input")
        / dataset
        / dataset
    )

    outfile_base = (
        pathlib.Path("/Users/julia/Desktop/Promotion/ADNA_Popgen/Pandora/results")
        / dataset
    )
    outfile_base.mkdir(exist_ok=True, parents=True)

    outfile_prefix = outfile_base / dataset

    convertf_dir = outfile_base / "convertf"
    convertf_dir.mkdir(exist_ok=True)
    convert_prefix = convertf_dir / dataset

    bootstrap_dir = outfile_base / "bootstrap"
    bootstrap_dir.mkdir(exist_ok=True)

    smartpca = "/Users/julia/micromamba/envs/pca_micromamba/bin/smartpca"
    convertf = "/Users/julia/micromamba/envs/pca_micromamba/bin/convertf"

    n_pcs = run_pca_original_data_and_get_number_of_pcs(
        infile_prefix=infile_prefix,
        outfile_prefix=outfile_prefix,
        smartpca=smartpca,
        explained_variance_cutoff=0.01,
    )

    logger.info(fmt_message("Converting Input files to PLINK format for bootstrapping."))

    eigen_to_plink(
        eigen_prefix=infile_prefix, plink_prefix=convert_prefix, convertf=convertf
    )

    logger.info(fmt_message(f"Drawing {n_bootstraps} bootstrapped datasets."))

    create_bootstrap_datasets(
        infile_prefix=convert_prefix,
        bootstrap_outdir=bootstrap_dir,
        n_bootstraps=n_bootstraps,
        seed=seed,
        n_threads=n_threads
    )

    logger.info(fmt_message("Converting bootstrapped files to EIGEN format for PCA."))

    parallel_plink_to_eigen(
        bootstrap_dir=bootstrap_dir,
        convertf=convertf,
        n_bootstraps=n_bootstraps,
        n_threads=n_threads
    )

    logger.info(fmt_message(f"Running PCA for {n_bootstraps} bootstrapped datasets."))

    run_pca_bootstapped_data(
        bootstrap_dir=bootstrap_dir,
        smartpca=smartpca,
        n_bootstraps=n_bootstraps,
        n_pcs=n_pcs,
        n_threads=n_threads
    )


if __name__ == "__main__":
    main()
