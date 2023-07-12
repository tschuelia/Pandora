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


def get_filenames(
    prefix: pathlib.Path, file_format: FileFormat
) -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """
    Constructs the triple of file names for the geno, snp, and ind files for the given data format.
    For a list of expected file endings see the Pandora wiki on GitHub.

    Args:
        prefix (pathlib.Path): Prefix of the filepath pointing to the respective dataset files.
        file_format (FileFormat): FileFormat of the respective format the datset is in.

    Returns:
        (pathlib.Path, pathlib.Path, pathlib.Path): File paths for the respective geno, snp and ind files.

    """
    if file_format not in FILE_SUFFIXES:
        raise PandoraException(f"Unrecognized file format: {file_format.value}")

    geno_suffix, snp_suffix, ind_suffix = FILE_SUFFIXES[file_format]

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
) -> None:
    """
    Uses the EIGENSOFT convertf program to convert the given dataset in in_format into the same dataset in out_format.
    If all respective out_prefix files are already present, only runs convertf if redo is True.

    Args:
        convertf (Executable): Executable of the EIGENSOFT convertf program.
        in_prefix (pathlib.Path): Prefix of the filepath pointing to the respective dataset files that should be converted.
        in_format (FileFormat): Format of the input files.
        out_prefix: Prefix of the filepath where the output should be stored.
        out_format (FileFormat): Desired output format.
        redo (bool): Whether to rerun the conversion if the output files are already present.
    """
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


def _clean_converted_names(ind_out: pathlib.Path) -> None:
    """
    For some reason, the file conversion from PLINK to EIGEN results in weird sample IDs.
    This method cleans all names to match the original IDs the affected file is the ind_out file.

    Args:
        ind_out (pathlib.Path): FilePath of the indfile to correct the sample IDs in.
    """
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
    raise NotImplementedError
