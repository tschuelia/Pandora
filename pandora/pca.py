import logging
import tempfile
import pandas as pd
import subprocess
import textwrap

from multiprocessing import Pool

from pandora.custom_types import *
from pandora.postprocessing import read_smartpca_eigenvec
from pandora.logger import logger, fmt_message


def get_number_of_populations(indfile: FilePath):
    df = pd.read_table(indfile, delimiter=" ", skipinitialspace=True, header=None)
    return df[2].unique().shape[0]


def run_smartpca(
    infile_prefix: FilePath,
    outfile_prefix: FilePath,
    smartpca: Executable,
    n_pcs: int = 20,
    redo: bool = False,
):
    evec_out = pathlib.Path(f"{outfile_prefix}.evec")
    eval_out = pathlib.Path(f"{outfile_prefix}.eval")

    if eval_out.exists() and eval_out.exists() and not redo:
        logger.info(
            fmt_message(f"Skipping PCA. Files {outfile_prefix}.* already exist.")
        )
        return

    with tempfile.NamedTemporaryFile(mode="w") as tmpfile:
        num_populations = get_number_of_populations(f"{infile_prefix}.ind")

        conversion_content = f"""
        genotypename: {infile_prefix}.geno
        snpname: {infile_prefix}.snp
        indivname: {infile_prefix}.ind
        evecoutname: {evec_out}
        evaloutname: {eval_out}
        numoutevec: {n_pcs}
        numoutlieriter: 0
        maxpops: {num_populations}
        """

        tmpfile.write(textwrap.dedent(conversion_content))
        tmpfile.flush()

        cmd = [
            smartpca,
            "-p",
            tmpfile.name,
        ]
        subprocess.check_output(cmd)


def check_pcs_sufficient(
    explained_variances: pd.DataFrame, cutoff: float
) -> Union[int, None]:
    if min(explained_variances.explained_variance) < cutoff:
        # at least one PC explains less than <cutoff>% variance
        # -> find the index of the last PC explaining more than <cutoff>%
        # the optimal number of PCs is this index + 1 (pandas is zero-indexed)
        explained_variances.sort_values(by="explained_variance")
        max_idx = explained_variances.loc[
            explained_variances.explained_variance >= cutoff
        ].index.max()
        n_pcs = max_idx + 1
        logger.info(
            fmt_message(
                f"Optimal number of PCs for explained variance cutoff {cutoff}: {n_pcs}"
            )
        )
        return n_pcs


def determine_number_of_pcs(
    infile_prefix: FilePath,
    outfile_prefix: FilePath,
    smartpca: Executable,
    cutoff: float = 0.01,
    redo: bool = False,
):
    n_pcs = 20
    evec_file = pathlib.Path(f"{outfile_prefix}.evec")
    eval_file = pathlib.Path(f"{outfile_prefix}.eval")

    if evec_file.exists() and eval_file.exists() and not redo:
        # in case of a restarted analysis, this ensures that we correctly update the n_pcs variable below
        _, explained_variances = read_smartpca_eigenvec(f"{outfile_prefix}.evec")

        logger.info(
            fmt_message(
                f"Resuming from checkpoint: "
                f"Reading data from existing PCA outfiles {outfile_prefix}.*. "
                f"Delete files or set the redo flag in case you want to rerun the PCA."
            )
        )

        # check if the last run PCA already had the optimal number of PCs present
        # if yes, return it
        best_pcs = check_pcs_sufficient(explained_variances, cutoff)
        if best_pcs:
            return best_pcs
        # otherwise, resume the search for the optimal number of PCs from the last number of PCs
        n_pcs = explained_variances.shape[0]
        logger.info(
            fmt_message(
                f"Resuming the search for the optimal number of PCS. "
                f"Previously tested setting: {n_pcs}, new setting: {int(n_pcs * 1.5)} "
            )
        )
        n_pcs = int(1.5 * n_pcs)
    else:
        logger.info(
            fmt_message(
                f"Determining number of PCs. Now running PCA analysis with {n_pcs} PCs."
            )
        )

    while True:
        run_smartpca(
            infile_prefix=infile_prefix,
            outfile_prefix=outfile_prefix,
            smartpca=smartpca,
            n_pcs=n_pcs,
            redo=True,
        )
        # get the explained variances
        _, explained_variances = read_smartpca_eigenvec(f"{outfile_prefix}.evec")

        best_pcs = check_pcs_sufficient(explained_variances, cutoff)
        if best_pcs:
            return best_pcs

        # if all PCs explain >= <cutoff>% variance, rerun the PCA with an increased number of PCs
        # we increase the number by a factor of 1.5
        logger.info(
            fmt_message(
                f"{n_pcs} PCs not sufficient. Repeating analysis with {int(1.5 * n_pcs)} PCs."
            )
        )
        n_pcs = int(1.5 * n_pcs)


def run_pca_original_data_and_get_number_of_pcs(
    infile_prefix: FilePath,
    outfile_prefix: FilePath,
    smartpca: Executable,
    explained_variance_cutoff=0.01,
):
    n_pcs = determine_number_of_pcs(
        infile_prefix=infile_prefix,
        outfile_prefix=outfile_prefix,
        smartpca=smartpca,
        cutoff=explained_variance_cutoff,
        redo=False,
    )

    # The search for the optimal number of PCs already ran the PCA on the original data
    # we just have to trim the results for the actual number of PCs we will be considering in our analyses
    original_pca, original_explained_variances = read_smartpca_eigenvec(
        f"{outfile_prefix}.evec"
    )
    original_pca = original_pca[
        ["sample_id", "population"] + [f"PC{i}" for i in range(n_pcs)]
    ]
    original_explained_variances = original_explained_variances.head(n_pcs)

    # save the resulting dataframes
    original_pca.to_parquet(f"{outfile_prefix}.pca.parquet")
    original_explained_variances.to_parquet(f"{outfile_prefix}.variances.parquet")

    return n_pcs


def _run_bs_pca(args):
    bootstrap_dir, smartpca, n_pcs, i = args
    infile_prefix = bootstrap_dir / f"bootstrap_{i}"
    outfile_prefix = bootstrap_dir / f"bootstrap_{i}.pca"

    run_smartpca(
        infile_prefix=infile_prefix,
        outfile_prefix=outfile_prefix,
        smartpca=smartpca,
        n_pcs=n_pcs,
    )

    logger.info(fmt_message(f"Finished PCA for bootstrapped dataset #{i}"))


def run_pca_bootstapped_data(
    bootstrap_dir: FilePath,
    smartpca: Executable,
    n_bootstraps: int,
    n_pcs: int,
    n_threads: int,
):
    args = [
        (
            bootstrap_dir,
            smartpca,
            n_pcs,
            i + 1
        )
        for i in range(n_bootstraps)
    ]

    with Pool(n_threads) as p:
        list(p.map(_run_bs_pca, args))
