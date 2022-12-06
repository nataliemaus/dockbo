# read in folded pdbs, and ocmpute scores 


from .fold_all import load_seqs 



def main():
    num_seqs = 100_000 
    for ix in range(num_seqs):
        path_to_pdb = f"folded_pdbs/seq{ix}.pdb" 
