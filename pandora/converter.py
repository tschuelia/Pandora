import textwrap
import subprocess

from pandora.custom_types import *
from pandora.custom_errors import *


FILE_SUFFIXES = {
    # Format: (geno, snp, ind)
    FileFormat.ANCESTRYMAP: (".geno", ".snp", ".ind"),
    FileFormat.EIGENSTRAT: (".geno", ".snp", ".ind"),
    FileFormat.PED: (".ped", ".map", ".fam"),
    FileFormat.PACKEDPED: (".bed", ".bim", ".fam"),
    FileFormat.PACKEDANCESTRYMAP: (".geno", ".snp", ".ind"),
}


def get_filenames(prefix: pathlib.Path, format: FileFormat):
    if format not in FILE_SUFFIXES:
        raise PandoraException(f"Unrecognized file format: {format.value}")

    geno_suffix, snp_suffix, ind_suffix = FILE_SUFFIXES[format]

    geno = pathlib.Path(f"{prefix}{geno_suffix}")
    snp = pathlib.Path(f"{prefix}{snp_suffix}")
    ind = pathlib.Path(f"{prefix}{ind_suffix}")

    return geno, snp, ind


def run_convertf(
    convertf: Executable,
    in_prefix: pathlib.Path,
    in_format: FileFormat,
    out_prefix: pathlib.Path,
    out_format: FileFormat,
    redo: bool = False,
):
    geno_in, snp_in, ind_in = get_filenames(in_prefix, in_format)
    geno_out, snp_out, ind_out = get_filenames(out_prefix, out_format)

    if geno_out.exists() and snp_out.exists() and ind_out.exists() and not redo:
        return

    convertf_log = pathlib.Path(f"{out_prefix}.convertf.log")
    convertf_par = pathlib.Path(f"{out_prefix}.convertf.par")

    conversion_content = f"""
                   genotypename: {geno_in}
                   snpname: {snp_in}
                   indivname: {ind_in}
                   outputformat: {out_format.value}
                   genotypeoutname: {geno_out} 
                   snpoutname: {snp_out} 
                   indivoutname: {ind_out} 
                   """
    convertf_par.open(mode="w").write(textwrap.dedent(conversion_content))

    cmd = [convertf, "-p", str(convertf_par)]

    with convertf_log.open("w") as logfile:
        try:
            subprocess.run(cmd, stdout=logfile, stderr=logfile)
        except subprocess.CalledProcessError:
            raise RuntimeError(
                f"Error running convertf. "
                f"Check the convertf logfile {convertf_log.absolute()} for details."
            )
    subprocess.check_output(cmd)

    _clean_converted_names(ind_out)


def _clean_converted_names(ind_out: pathlib.Path):
    # for some reason, the file conversion from PLINK to EIGEN results in weird sample IDs
    # -> clean them  to match the original IDs the affected file is the ind_out file
    corrected_content = []
    for line in ind_out.open():
        line = line.strip()
        if ":" not in line:
            # looks like the file was already cleaned or we don't need to clean up
            return
        _, line = line.split(":", maxsplit=1)
        corrected_content.append(line)

    ind_out.open("w").write("\n".join(corrected_content))


def _reannotate_populations():
    # TODO: implement
    pass
