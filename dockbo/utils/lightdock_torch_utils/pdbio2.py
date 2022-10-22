import Bio 
from Bio.PDB import PDBParser

# You can use a dict to convert three letter code to one letter code
# https://bioinformatics.stackexchange.com/questions/14101/extract-residue-sequence-from-pdb-file-in-biopython-but-open-to-recommendation
d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

def pdb_to_aaseq(pdb_file):
    # run parser
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('struct', pdb_file)# record)   
    all_chains = ""
    for model in structure:
        for chain in model:
            seq = []
            for residue in chain:
                seq.append(d3to1[residue.resname])
            full_seq = ''.join(seq)
            all_chains += full_seq 
    return all_chains 


if __name__ == "__main__": 
    ab_id = '5j57'
    pdb_file = f"/home/nmaus/dockbo/example/pdbs/bighat_verification2/{ab_id}_antigen.pdb"
    aaseq1 = pdb_to_aaseq(pdb_file)
    print(aaseq1)
