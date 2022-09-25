from pathlib import Path
import os 
import json 
from dockbo.utils.lightdock_torch_utils.structure.complex import Complex 
from dockbo.utils.lightdock_torch_utils.lightdock_constants import DEFAULT_LIST_EXTENSION
from dockbo.utils.lightdock_torch_utils.PDBIO import parse_complex_from_file


def get_pdb_files(input_file):
    """Get a list of the PDB files in the input_file"""
    file_names = []
    with open(input_file) as handle:
        for line in handle:
            file_name = line.rstrip(os.linesep)
            if Path(file_name).exists():
                file_names.append(file_name)
    return file_names


def read_input_structure(
    pdb_file_name,
    ignore_oxt=True,
    ignore_hydrogens=False,
    ignore_water=False,
    verbose_parser=False,
):
    """Reads the input structure.

    The arguments pdb_file_name can be a PDB file or a file
    containing a list of PDB files.

    ignore_oxt flag avoids saving OXT atoms.
    """
    atoms_to_ignore = []
    residues_to_ignore = []
    if ignore_oxt:
        atoms_to_ignore.append("OXT")
    if ignore_hydrogens:
        atoms_to_ignore.append("H")
    if ignore_water:
        residues_to_ignore.append("HOH")

    structures = []
    file_names = []
    file_name, file_extension = os.path.splitext(pdb_file_name)
    if file_extension == DEFAULT_LIST_EXTENSION:
        file_names.extend(get_pdb_files(pdb_file_name))
    else:
        file_names.append(pdb_file_name)
    for file_name in file_names:
        atoms, residues, chains = parse_complex_from_file(
            file_name, atoms_to_ignore, residues_to_ignore, verbose_parser
        )
        structures.append(
            {
                "atoms": atoms,
                "residues": residues,
                "chains": chains,
                "file_name": file_name,
            }
        )

    # Representatives are now the first structure, but this could change in the future
    structure = Complex.from_structures(structures)
    return structure


def get_setup_from_file(file_name):
    """Reads the simulation setup from the file_name"""
    with open(file_name) as input_file:
        return json.load(input_file)
