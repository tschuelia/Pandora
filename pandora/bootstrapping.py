import os
import random
import shutil

from multiprocessing import Pool

from pandora.custom_types import *
from pandora.logger import logger, fmt_message
from pandora.pca import PCA
from pandora.pca_runner import run_smartpca


def bootstrap_snp_level(
    infile_prefix: FilePath, outfile_prefix: FilePath, seed: int, redo: bool = False
):
    random.seed(seed)
    snp_in = pathlib.Path(f"{infile_prefix}.snp")
    geno_in = pathlib.Path(f"{infile_prefix}.geno")
    ind_in = pathlib.Path(f"{infile_prefix}.ind")

    snp_out = pathlib.Path(f"{outfile_prefix}.snp")
    geno_out = pathlib.Path(f"{outfile_prefix}.geno")
    ind_out = pathlib.Path(f"{outfile_prefix}.ind")

    if snp_out.exists() and geno_out.exists() and ind_out.exists() and not redo:
        logger.info(
            fmt_message(
                f"Skipping bootstrapping. Files {outfile_prefix}.* already exist."
            )
        )
        return

    # when bootstrapping on SNP level, the .ind file does not change
    shutil.copy(ind_in, ind_out)

    # sample the SNPs using the snp file
    # each line in the snp file corresponds to one SNP
    num_samples = sum(1 for _ in open(snp_in))
    bootstrap_snp_indices = sorted(random.choices(range(num_samples), k=num_samples))

    # 1. Bootstrap the .snp file
    snps = snp_in.open().readlines()
    seen_snps = set()

    with snp_out.open(mode="a") as f:
        for bootstrap_idx in bootstrap_snp_indices:
            snp_line = snps[bootstrap_idx]

            # lines look like this:
            # 1   S_Adygei-1.DG 0 0 1 1
            snp_id, chrom_id, rest = snp_line.split(maxsplit=2)
            deduplicate = snp_id

            snp_id_counter = 0

            while deduplicate in seen_snps:
                snp_id_counter += 1
                deduplicate = f"{snp_id}_r{snp_id_counter}"

            seen_snps.add(deduplicate)
            f.write(f"{deduplicate} {chrom_id} {rest}")

    # 2. Bootstrap the .geno file using the bootstrap_snp_indices above
    # the .geno file contains one column for each individual sample
    genos = geno_in.open().readlines()
    with geno_out.open(mode="a") as f:
        for bootstrap_idx in bootstrap_snp_indices:
            geno_line = genos[bootstrap_idx]
            f.write(geno_line)


def _run(args):
    _i, _seed, _in, _out, _redo, _convertf, _smartpca, _npcs, _smartpca_optional_settings = args
    bootstrap_prefix = _out / f"bootstrap_{_i}"

    # first check if the final EIGEN files already exist
    # if so, skip the bootstrapping and the conversion
    geno_out = pathlib.Path(f"{bootstrap_prefix}.geno")
    snp_out = pathlib.Path(f"{bootstrap_prefix}.snp")
    ind_out = pathlib.Path(f"{bootstrap_prefix}.ind")

    files_missing = not all([geno_out.exists(), snp_out.exists(), ind_out.exists()])

    if _redo or files_missing:
        bootstrap_snp_level(
            infile_prefix=_in, outfile_prefix=bootstrap_prefix, seed=_seed, redo=_redo
        )

        logger.debug(fmt_message(f"Finished drawing bootstrap dataset #{_i}"))
    else:
        logger.debug(fmt_message(f"Bootstrapped dataset #{_i} already exists."))

    # to not waste disk space, we remove the input files once the conversion is done
    # we don't want to save the data redundantly in different file types
    for f in [
        pathlib.Path(f"{bootstrap_prefix}.ped"),
        pathlib.Path(f"{bootstrap_prefix}.map"),
        pathlib.Path(f"{bootstrap_prefix}.fam"),
    ]:
        if f.exists():
            os.remove(f)

    # next, run the PCA and return the resulting PCA object
    pca_prefix = _out / f"bootstrap_{_i}.pca"

    pca = run_smartpca(
        infile_prefix=bootstrap_prefix,
        outfile_prefix=pca_prefix,
        smartpca=_smartpca,
        n_pcs=_npcs,
        redo=_redo,
        smartpca_optional_settings = _smartpca_optional_settings
    )

    logger.info(fmt_message(f"Finished PCA for bootstrapped dataset #{_i}"))
    return pca


def create_bootstrap_pcas(
    infile_prefix: FilePath,
    bootstrap_outdir: FilePath,
    convertf: Executable,
    smartpca: Executable,
    n_bootstraps: int,
    seed: int,
    n_pcs: int,
    n_threads: int,
    redo: bool = False,
    smartpca_optional_settings: Dict = None
) -> List[PCA]:
    """
    Draws n_bootstraps bootstrapped datasets using the provided PLINK files, creating new PLINK files.
    Afterwards converts them to EIGEN format and deletes the intermediate PLINK files.
    """
    random.seed(seed)
    bootstrap_seeds = [random.randint(0, 1_000_000) for _ in range(n_bootstraps)]
    args = [
        (_i + 1, _seed, infile_prefix, bootstrap_outdir, redo, convertf, smartpca, n_pcs, smartpca_optional_settings)
        for _i, _seed in zip(range(n_bootstraps), bootstrap_seeds)
    ]

    with Pool(n_threads) as p:
        return list(p.map(_run, args))

