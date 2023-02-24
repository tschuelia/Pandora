import random
import shutil

from multiprocessing import Pool

from pandora.custom_types import *
from pandora.logger import logger, fmt_message


def bootstrap_snp_level(infile_prefix: FilePath, outfile_prefix: FilePath, seed: int, redo: bool = False):
    random.seed(seed)
    map_in = pathlib.Path(f"{infile_prefix}.map")
    ped_in = pathlib.Path(f"{infile_prefix}.ped")
    fam_in = pathlib.Path(f"{infile_prefix}.fam")

    map_out = pathlib.Path(f"{outfile_prefix}.map")
    ped_out = pathlib.Path(f"{outfile_prefix}.ped")
    fam_out = pathlib.Path(f"{outfile_prefix}.fam")

    if map_out.exists() and ped_out.exists() and fam_out.exists() and not redo:
        logger.info(
            fmt_message(f"Skipping bootstrapping. Files {outfile_prefix}.* already exist.")
        )
        return

    # when bootstrapping on SNP level, the .fam file does not change
    shutil.copy(fam_in, fam_out)

    # sample the SNPs using the map file
    # each line in the map file corresponds to one SNP
    num_samples = sum(1 for _ in open(map_in))
    bootstrap_snp_indices = sorted(random.choices(range(num_samples), k=num_samples))

    # 1. Bootstrap the .map file
    snps = open(map_in).readlines()
    seen_snps = set()

    with open(map_out, "a") as f:
        for bootstrap_idx in bootstrap_snp_indices:
            snp_line = snps[bootstrap_idx]

            # lines look like this:
            # 1   S_Adygei-1.DG 0 0 1 1
            chrom_id, snp_id, rest = snp_line.split(maxsplit=2)
            deduplicate = snp_id

            snp_id_counter = 0

            while deduplicate in seen_snps:
                snp_id_counter += 1
                deduplicate = f"{snp_id}_r{snp_id_counter}"

            seen_snps.add(deduplicate)
            f.write(f"{chrom_id} {deduplicate} {rest}")

    # 2. Bootstrap the .ped file using the bootstrap_snp_indices above
    # the .ped file contains one line for each individual sample
    # each line has 2V + 6 fields with V being the number of samples
    # The first six fields do not change
    with open(ped_out, "a") as ped_out_handle:
        for indiv_line in open(ped_in):
            indiv_line = indiv_line.strip()
            fields = indiv_line.split()

            # the first 6 fields don't change with bootstrapping
            new_indiv_line = fields[:6]

            # the following lines correspond to the SNPs
            # each SNP accounts for two fields
            # so for each index in the bootstrap_snp_indices we have to access two fields:
            # 5 + (2 * (index + 1)) and 5 + (2 * (index + 1) + 1)
            # (5 and (index + 1) since Python is 0-indexed)
            for bootstrap_idx in bootstrap_snp_indices:
                idx_var1 = 5 + 2 * (bootstrap_idx + 1)
                new_indiv_line.append(fields[idx_var1 - 1])
                new_indiv_line.append(fields[idx_var1])

            ped_out_handle.write(" ".join(new_indiv_line))
            ped_out_handle.write("\n")


def bootstrap_indiv_level():
    raise NotImplementedError()


def _run(args):
    _i, _seed, _in, _out = args
    bootstrap_prefix = _out / f"bootstrap_{_i}"

    bootstrap_snp_level(
        infile_prefix=_in,
        outfile_prefix=bootstrap_prefix,
        seed=_seed
    )

    logger.info(fmt_message(f"Finished bootstrap dataset #{_i}"))


def create_bootstrap_datasets(infile_prefix: FilePath, bootstrap_outdir: FilePath, n_bootstraps: int, seed: int, n_threads: int):
    random.seed(seed)
    bootstrap_seeds = [random.randint(0, 1_000_000) for _ in range(n_bootstraps)]
    args = [(_i + 1, _seed, infile_prefix, bootstrap_outdir) for _i, _seed in zip(range(n_bootstraps), bootstrap_seeds)]

    with Pool(n_threads) as p:
        list(p.map(_run, args))
