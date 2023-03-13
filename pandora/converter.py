import tempfile
import textwrap
import subprocess

from pandora.custom_types import *
from pandora.logger import logger, fmt_message


def run_convertf(
        convertf: Executable,
        genotype_in: FilePath,
        snp_in: FilePath,
        ind_in: FilePath,
        genotype_out: FilePath,
        snp_out: FilePath,
        ind_out: FilePath,
        outputformat: str,
        redo: bool = False
):
    if genotype_out.exists() and snp_out.exists() and ind_out.exists() and not redo:
        logger.info(
            fmt_message(f"Skipping file conversion to {outputformat}. Files already converted.")
        )

    with tempfile.NamedTemporaryFile(mode="w") as tmpfile:
        conversion_content = f"""
                       genotypename: {genotype_in}
                       snpname: {snp_in}
                       indivname: {ind_in}
                       outputformat: {outputformat}
                       genotypeoutname: {genotype_out} 
                       snpoutname: {snp_out} 
                       indivoutname: {ind_out} 
                       """
        tmpfile.write(textwrap.dedent(conversion_content))
        tmpfile.flush()

        cmd = [
            convertf,
            "-p",
            tmpfile.name
        ]
        subprocess.check_output(cmd)


def clean_converted_names(ind_out: FilePath):
    # for some reason, the file conversion from PLINK to EIGEN results in weird sample IDs
    # -> clean them  to match the original IDs the affected file is the ind_out file
    corrected_content = []
    for line in ind_out.open():
        line = line.strip()
        _, line = line.split(":", maxsplit=1)
        corrected_content.append(line)

    ind_out.open("w").write("\n".join(corrected_content))


def eigen_to_plink(eigen_prefix: FilePath, plink_prefix: FilePath, convertf: Executable, redo: bool = False):
    run_convertf(
        convertf=convertf,
        genotype_in=pathlib.Path(f"{eigen_prefix}.geno"),
        snp_in=pathlib.Path(f"{eigen_prefix}.snp"),
        ind_in=pathlib.Path(f"{eigen_prefix}.ind"),
        genotype_out=pathlib.Path(f"{plink_prefix}.ped"),
        snp_out=pathlib.Path(f"{plink_prefix}.map"),
        ind_out=pathlib.Path(f"{plink_prefix}.fam"),
        outputformat="PED",
        redo=redo,
    )


def plink_to_eigen(plink_prefix: FilePath, eigen_prefix: FilePath, convertf: Executable, redo: bool = False):
    ind_out = pathlib.Path(f"{eigen_prefix}.fam")

    run_convertf(
        convertf=convertf,
        genotype_in=pathlib.Path(f"{plink_prefix}.geno"),
        snp_in=pathlib.Path(f"{plink_prefix}.snp"),
        ind_in=pathlib.Path(f"{plink_prefix}.ind"),
        genotype_out=pathlib.Path(f"{eigen_prefix}.ped"),
        snp_out=pathlib.Path(f"{eigen_prefix}.map"),
        ind_out=ind_out,
        outputformat="EIGENSTRAT",
        redo=redo,
    )

    clean_converted_names(ind_out)


def plink_to_bplink(plink_prefix: FilePath, bplink_prefix: FilePath, convertf: Executable, redo: bool = False):
    run_convertf(
        convertf=convertf,
        genotype_in=pathlib.Path(f"{plink_prefix}.geno"),
        snp_in=pathlib.Path(f"{plink_prefix}.snp"),
        ind_in=pathlib.Path(f"{plink_prefix}.ind"),
        genotype_out=pathlib.Path(f"{bplink_prefix}.ped"),
        snp_out=pathlib.Path(f"{bplink_prefix}.map"),
        ind_out=pathlib.Path(f"{bplink_prefix}.fam"),
        outputformat="PACKEDPED",
        redo=redo,
    )


def eigen_to_bplink(eigen_prefix: FilePath, bplink_prefix: FilePath, convertf: Executable, redo: bool = False):
    run_convertf(
        convertf=convertf,
        genotype_in=pathlib.Path(f"{eigen_prefix}.geno"),
        snp_in=pathlib.Path(f"{eigen_prefix}.snp"),
        ind_in=pathlib.Path(f"{eigen_prefix}.ind"),
        genotype_out=pathlib.Path(f"{bplink_prefix}.bed"),
        snp_out=pathlib.Path(f"{bplink_prefix}.bim"),
        ind_out=pathlib.Path(f"{bplink_prefix}.fam"),
        outputformat="PACKEDPED",
        redo=redo,
    )


def bplink_to_eigen(bplink_prefix: FilePath, eigen_prefix: FilePath, convertf: Executable, redo: bool = False):
    ind_out = pathlib.Path(f"{eigen_prefix}.fam")

    run_convertf(
        convertf=convertf,
        genotype_in=pathlib.Path(f"{bplink_prefix}.geno"),
        snp_in=pathlib.Path(f"{bplink_prefix}.snp"),
        ind_in=pathlib.Path(f"{bplink_prefix}.ind"),
        genotype_out=pathlib.Path(f"{eigen_prefix}.ped"),
        snp_out=pathlib.Path(f"{eigen_prefix}.map"),
        ind_out=ind_out,
        outputformat="EIGENSTRAT",
        redo=redo,
    )

    clean_converted_names(ind_out)
