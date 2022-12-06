from Bio.PDB import PDBParser, PDBIO, Select
import os 

class NonHetSelect(Select):
    def accept_residue(self, residue):
        return 1 if residue.id[0] == " " else 0


def remove_hetero_atoms_and_hydrogens(path_to_pdb):
    # First, remove HET atoms
    pdb = PDBParser().get_structure(path_to_pdb.replace(".pdb", ""), path_to_pdb)
    io = PDBIO()
    io.set_structure(pdb)
    save_path = path_to_pdb.replace(".pdb", "_no_het.pdb")
    io.save(save_path, NonHetSelect())
    # Next, use pdb-tools to delete Hydrogen atoms 
    #   Need pip install pdb-tools
    #   http://www.bonvinlab.org/pdb-tools/
    save_path_noh = save_path.replace(".pdb", "_noh.pdb")
    os.system(f"pdb_delelem -H {save_path} > {save_path_noh}")

    return save_path_noh 
