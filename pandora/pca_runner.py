import subprocess
import tempfile
import textwrap

from sklearn.decomposition import PCA as sklearnPCA

from pandora.custom_types import *
from pandora.logger import *
from pandora.pca import PCA, from_plink, from_sklearn, from_smartpca


def run_smartpca(
    infile_prefix: FilePath,
    outfile_prefix: FilePath,
    smartpca: Executable,
    n_pcs: int = 20,
    redo: bool = False,
) -> PCA:
    geno = pathlib.Path(f"{infile_prefix}.geno")
    snp = pathlib.Path(f"{infile_prefix}.snp")
    ind = pathlib.Path(f"{infile_prefix}.ind")

    files_exist = all([geno.exists(), snp.exists(), ind.exists()])
    if not files_exist:
        raise ValueError(
            f"Not all input files for file prefix {infile_prefix} present. "
            f"Looking for files in EIGEN format with endings .geno, .snp, and .ind"
        )

    evec_out = pathlib.Path(f"{outfile_prefix}.evec")
    eval_out = pathlib.Path(f"{outfile_prefix}.eval")
    smartpca_log = pathlib.Path(f"{outfile_prefix}.smartpca.log")

    files_exist = all([evec_out.exists(), eval_out.exists(), smartpca_log.exists()])

    if files_exist and not redo:
        # TODO: das reicht nicht als check, bei unfertigen runs sind die files einfach nicht vollständig aber
        #  leider noch vorhanden
        logger.info(
            fmt_message(f"Skipping smartpca. Files {outfile_prefix}.* already exist.")
        )
        return from_smartpca(evec_out)

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
            shrinkmode: YES 
            """
        # projection_file = pathlib.Path(f"{infile_prefix}.population")
        # if projection_file.exists():
        #     conversion_content += f"\npoplistname: {projection_file}"

        tmpfile.write(textwrap.dedent(conversion_content))
        tmpfile.flush()

        cmd = [
            smartpca,
            "-p",
            tmpfile.name,
        ]
        with smartpca_log.open("w") as logfile:
            subprocess.run(cmd, stdout=logfile, stderr=logfile)

    return from_smartpca(evec_out)


def run_plink(
    infile_prefix: FilePath,
    outfile_prefix: FilePath,
    plink: Executable,
    n_pcs: int = 20,
    redo: bool = False,
) -> PCA:
    evec_out = pathlib.Path(f"{outfile_prefix}.eigenvec")
    eval_out = pathlib.Path(f"{outfile_prefix}.eigenval")
    plink_log = pathlib.Path(f"{outfile_prefix}.plinkpca.log")

    files_exist = all([evec_out.exists(), eval_out.exists(), plink_log.exists()])

    if files_exist and not redo:
        # TODO: das reicht nicht als check, bei unfertigen runs sind die files einfach nicht vollständig aber leider noch vorhanden
        logger.info(
            fmt_message(f"Skipping plink PCA. Files {outfile_prefix}.* already exist.")
        )
        return from_plink(evec_out, eval_out)

    pca_cmd = [
        plink,
        "--pca",
        str(n_pcs),
        "--bfile",
        infile_prefix,
        "--out",
        outfile_prefix,
        "--no-fid"
    ]

    with plink_log.open("w") as logfile:
        subprocess.run(pca_cmd, stdout=logfile, stderr=logfile)

    return from_plink(evec_out, eval_out)


def run_sklearn(
    outfile_prefix: FilePath,
    n_pcs: int = 20,
    redo: bool = False,
) -> PCA:
    plink_snp_data = pathlib.Path(f"{outfile_prefix}.rel")
    plink_sample_data = pathlib.Path(f"{outfile_prefix}.rel.id")

    pc_vectors_file = pathlib.Path(f"{outfile_prefix}.sklearn.evec.npy")
    variances_file = pathlib.Path(f"{outfile_prefix}.sklearn.eval.npy")

    if redo or (not pc_vectors_file.exists() and not variances_file.exists()):
        snp_data = []
        for line in plink_snp_data.open():
            values = line.split()
            values = [float(v) for v in values]
            snp_data.append(values)

        snp_data = np.asarray(snp_data)

        pca = sklearnPCA(n_components=n_pcs)
        pca_data = pca.fit_transform(snp_data)

        np.save(pc_vectors_file, pca_data)
        np.save(variances_file, pca.explained_variance_ratio_)

    return from_sklearn(
        evec_file=pc_vectors_file,
        eval_file=variances_file,
        plink_id_file=plink_sample_data
    )


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
