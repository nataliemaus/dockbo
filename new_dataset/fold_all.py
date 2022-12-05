
# import uuid 
# import os 
from dockbo.utils.fold_utils.fold_utils import remove_hetero_atoms_and_hydrogens
from igfold import IgFoldRunner, init_pyrosetta 
import pandas as pd 
import argparse 

def fold_protein(
    igfold_runner,
    heavy_chain_seq, 
    save_path, 
    light_chain_seq=None,
):
    sequences = {"H":heavy_chain_seq }
    if light_chain_seq is not None:
        sequences["L"] = light_chain_seq # stays constant
    igfold_runner.fold(
        save_path, # Output PDB file 
        sequences=sequences, # Antibody sequences
        do_refine=True, # Refine the antibody structure with PyRosetta
        do_renum=True, # Renumber predicted antibody structure (Chothia) :) ! 
    ) 
    pdb_path = remove_hetero_atoms_and_hydrogens(pdb_path)


def load_seqs():
    df = pd.read_csv('new_100k_seqs.csv')
    seq_ids = df['seq_ids']
    h_chains = df['h_chains']
    l_chain = df['l_chains'][0]
    return l_chain, h_chains, seq_ids 
    

def fold(args):
    init_pyrosetta()
    igfold_runner = IgFoldRunner()
    l_chain, h_chains, seq_ids  = load_seqs() 
    h_chains = h_chains[args.min_idx: args.max_idx]
    seq_ids = seq_ids[args.min_idx: args.max_idx]
    for ix, h_chain in enumerate(h_chains):
        seq_id = seq_ids[ix] 
        fold_protein( 
            igfold_runner=igfold_runner,
            heavy_chain_seq=h_chain, 
            save_path=f"folded_pdbs/seq{seq_id}.pdb", 
            light_chain_seq=l_chain,
        )




if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--work_dir', default='/home/nmaus/' ) 
    # parser.add_argument('--wandb_entity', default="nmaus" )
    # parser.add_argument('--wandb_project_name', default="adversarial-bo" )  
    parser.add_argument('--min_idx', type=int, default=0 ) 
    parser.add_argument('--max_idx', type=int, default=10 ) 
    parser.add_argument('--debug', type=bool, default=False)  
    args = parser.parse_args() 
    if args.debug:
        args.min_idx = 0
        args.max_idx = 1 

    fold(args) 
    # python3 fold_all.py --min_idx 0 --max_idx 10