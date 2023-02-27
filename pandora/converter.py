import tempfile
import textwrap
import subprocess

from pandora.custom_types import *
from pandora.logger import logger, fmt_message


def eigen_to_plink(eigen_prefix: FilePath, plink_prefix: FilePath, convertf: Executable, redo: bool = False):
    ped_out = pathlib.Path(f"{plink_prefix}.ped")
    map_out = pathlib.Path(f"{plink_prefix}.map")
    fam_out = pathlib.Path(f"{plink_prefix}.fam")

    if ped_out.exists() and map_out.exists() and fam_out.exists() and not redo:
        logger.info(
            fmt_message("Skipping file conversion from EIGEN to PLINK. Files already converted.")
        )
        return

    with tempfile.NamedTemporaryFile(mode="w") as tmpfile:
        conversion_content = f"""
               genotypename: {eigen_prefix}.geno
               snpname: {eigen_prefix}.snp
               indivname: {eigen_prefix}.ind
               outputformat: PED
               genotypeoutname: {ped_out} 
               snpoutname: {map_out} 
               indivoutname: {fam_out} 
               """
        tmpfile.write(textwrap.dedent(conversion_content))
        tmpfile.flush()

        logger.info(
            fmt_message("Converting EIGEN files to PLINK.")
        )

        cmd = [
            convertf,
            "-p",
            tmpfile.name
        ]
        subprocess.check_output(cmd)


def plink_to_eigen(plink_prefix: FilePath, eigen_prefix: FilePath, convertf: Executable, redo: bool = False):
    geno_out = pathlib.Path(f"{eigen_prefix}.geno")
    snp_out = pathlib.Path(f"{eigen_prefix}.snp")
    ind_out = pathlib.Path(f"{eigen_prefix}.ind")

    if geno_out.exists() and snp_out.exists() and ind_out.exists() and not redo:
        logger.info(
            fmt_message("Skipping file conversion from PLINK to EIGEN. Files already converted.")
        )
        return

    with tempfile.NamedTemporaryFile(mode="w") as tmpfile:
        conversion_content = f"""
        genotypename:    {plink_prefix}.ped
        snpname:         {plink_prefix}.map
        indivname:       {plink_prefix}.fam
        outputformat:    EIGENSTRAT
        genotypeoutname: {geno_out}
        snpoutname:      {snp_out}
        indivoutname:    {ind_out}
        """

        tmpfile.write(textwrap.dedent(conversion_content))
        tmpfile.flush()

        cmd = [
            convertf,
            "-p",
            tmpfile.name
        ]

        subprocess.check_output(cmd)

    # for some reason, the file conversion results in weird sample IDs -> clean them  to match the original IDs
    # the affected file is the ind_out file
    corrected_content = []
    for line in ind_out.open():
        line = line.strip()
        _, line = line.split(":", maxsplit=1)
        corrected_content.append(line)

    ind_out.open("w").write("\n".join(corrected_content))

