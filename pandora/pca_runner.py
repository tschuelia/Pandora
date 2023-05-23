import subprocess
import tempfile
import textwrap

from pandora.custom_types import *
from pandora.logger import *
from pandora.pca import PCA, from_smartpca


def _check_smartpca_results_correct(outfile_prefix: FilePath, n_pcs: int):
    """
    Checks whether existing smartPCA results are correct.
    We consider them to be correct if
    1. the smartPCA run finished properly as indicated by the respective log file
    2. the number of principal components of the existing PCA matches the requested number n_pcs
    """
    # 1. check if the run finished properly, that is indicated by a line "##end of smartpca:" in the smartpca_log file
    smartpca_log = pathlib.Path(f"{outfile_prefix}.smartpca.log")
    run_finished = any(["##end of smartpca:" in line for line in smartpca_log.open()])

    if not run_finished:
        # something must have gone wrong with the previous smartPCA run, it did not finish properly
        logger.debug(
            fmt_message(f"Previous smartPCA results {outfile_prefix}.* appear to be incomplete. Repeating PCA.")
        )
        return False

    # 2. check that the number of PCs of the existing results matches the number of PCs set in the function call here
    evec_out = pathlib.Path(f"{outfile_prefix}.evec")
    eval_out = pathlib.Path(f"{outfile_prefix}.eval")

    pca = from_smartpca(evec_out, eval_out)
    if pca.n_pcs != n_pcs:
        logger.debug(
            fmt_message(f"Previous smartPCA results have {pca.n_pcs} principal components. "
                        f"Requested n_pcs is {n_pcs}. Repeating PCA with {n_pcs} principal components.")
        )
        return False

    return True


def run_smartpca(
    infile_prefix: FilePath,
    outfile_prefix: FilePath,
    smartpca: Executable,
    n_pcs: int = 20,
    redo: bool = False,
    smartpca_optional_settings: Dict = None
) -> PCA:
    geno = pathlib.Path(f"{infile_prefix}.geno")
    snp = pathlib.Path(f"{infile_prefix}.snp")
    ind = pathlib.Path(f"{infile_prefix}.ind")

    # first check that all required input files are present
    files_exist = all([geno.exists(), snp.exists(), ind.exists()])
    if not files_exist:
        raise ValueError(
            f"Not all input files for file prefix {infile_prefix} present. "
            f"Looking for files in EIGEN format with endings .geno, .snp, and .ind"
        )

    evec_out = pathlib.Path(f"{outfile_prefix}.evec")
    eval_out = pathlib.Path(f"{outfile_prefix}.eval")
    smartpca_log = pathlib.Path(f"{outfile_prefix}.smartpca.log")

    # next, check whether the all required output files are already present
    # and whether the smartPCA run finished properly and the number of PCs matches the requested number of PCs
    files_exist = all([evec_out.exists(), eval_out.exists(), smartpca_log.exists()])

    if files_exist and _check_smartpca_results_correct(outfile_prefix, n_pcs) and not redo:
        logger.info(
            fmt_message(f"Skipping smartpca. Files {outfile_prefix}.* already exist.")
        )
        return from_smartpca(evec_out, eval_out)

    with tempfile.NamedTemporaryFile(mode="w") as tmpfile:
        _df = pd.read_table(
            f"{infile_prefix}.ind", delimiter=" ", skipinitialspace=True, header=None
        )
        num_populations = _df[2].unique().shape[0]

        conversion_content = f"""
            genotypename: {infile_prefix}.geno
            snpname: {infile_prefix}.snp
            indivname: {infile_prefix}.ind
            evecoutname: {evec_out}
            evaloutname: {eval_out}
            numoutevec: {n_pcs}
            maxpops: {num_populations}
            """

        conversion_content = textwrap.dedent(conversion_content)

        if smartpca_optional_settings is not None:
            for k, v in smartpca_optional_settings.items():
                if isinstance(v, bool):
                    v = "YES" if v else "NO"
                conversion_content += f"{k}: {v}\n"

        tmpfile.write(conversion_content)
        tmpfile.flush()

        cmd = [
            smartpca,
            "-p",
            tmpfile.name,
        ]
        with smartpca_log.open("w") as logfile:
            try:
                subprocess.run(cmd, stdout=logfile, stderr=logfile)
            except subprocess.CalledProcessError:
                raise RuntimeError(f"Error running smartPCA. "
                                   f"Check the smartPCA logfile {smartpca_log.absolute()} for details.")

    return from_smartpca(evec_out, eval_out)


def check_pcs_sufficient(explained_variances: List, cutoff: float) -> Union[int, None]:
    # if all PCs explain more than 1 - <cutoff>% variance we consider the number of PCs to be insufficient
    if all([e > (1 - cutoff) for e in explained_variances]):
        return

    # otherwise, find the index of the last PC explaining more than <cutoff>%
    sum_variances = 0
    for i, var in enumerate(explained_variances, start=1):
        sum_variances += var
        if sum_variances >= cutoff:
            logger.info(
                fmt_message(
                    f"Optimal number of PCs for explained variance cutoff {cutoff}: {i}"
                )
            )
            return i


def determine_number_of_pcs(
    infile_prefix: FilePath,
    outfile_prefix: FilePath,
    smartpca: Executable,
    explained_variance_cutoff: float = 0.95,
    redo: bool = False,
    **smartPCA_kwargs,
) -> int:
    n_pcs = 20
    pca_checkpoint = pathlib.Path(f"{outfile_prefix}.ckp")

    if pca_checkpoint.exists() and not redo:
        # checkpointing file contains three values: an int, a bool, and a float
        # the int is the number of PCs that was last tested
        # the bool says whether the analysis finished properly or not
        # the float (ignored here) is the amount of variance explained by the current number of PCs (used for debugging)
        n_pcs, finished = pca_checkpoint.open().readline().strip().split()
        n_pcs = int(n_pcs)
        finished = bool(int(finished))

        if finished:
            logger.info(
                fmt_message(
                    f"Resuming from checkpoint: determining number of PCs already finished."
                )
            )

            # check if the last smartpca run already had the optimal number of PCs present
            # if yes, create a new PCA object and truncate the data to the optimal number of PCs
            return n_pcs

        # otherwise, running smartPCA was aborted and did not finnish properly, resume from last tested n_pcs
        logger.info(
            fmt_message(
                f"Resuming from checkpoint: Previously tested setting {n_pcs} not sufficient. "
                f"Repeating with n_pcs = {int(1.5 * n_pcs)}"
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
        with tempfile.TemporaryDirectory() as tmp_outdir:
            tmp_outfile_prefix = pathlib.Path(tmp_outdir) / "determine_npcs"
            pca = run_smartpca(
                infile_prefix=infile_prefix,
                outfile_prefix=tmp_outfile_prefix,
                smartpca=smartpca,
                n_pcs=n_pcs,
                redo=True,
                **smartPCA_kwargs
            )
            best_pcs = check_pcs_sufficient(
                pca.explained_variances, explained_variance_cutoff
            )
            if best_pcs:
                pca_checkpoint.write_text(f"{best_pcs} 1")
                return best_pcs

        # if all PCs explain >= <cutoff>% variance, rerun the PCA with an increased number of PCs
        # we increase the number by a factor of 1.5
        logger.info(
            fmt_message(
                f"{n_pcs} PCs not sufficient. Repeating analysis with {int(1.5 * n_pcs)} PCs."
            )
        )
        # number of PCs not sufficient, write checkpoint and increase n_pcs
        pca_checkpoint.write_text(f"{n_pcs} 0 {sum(pca.explained_variances)}")
        n_pcs = int(1.5 * n_pcs)
