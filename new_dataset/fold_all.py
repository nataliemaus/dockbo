import sys 
sys.path.append("../")
from dockbo.utils.fold_utils.fold_utils import remove_hetero_atoms_and_hydrogens
from igfold import IgFoldRunner, init_pyrosetta 
import pandas as pd 
import argparse 
import os 

def fold_protein(
    igfold_runner,
    heavy_chain_seq, 
    save_path, 
    light_chain_seq=None,
):
    # only save new pdbs if path does not already exist 
    if not os.path.exists(save_path):
        sequences = {"H":heavy_chain_seq }
        if light_chain_seq is not None: 
            sequences["L"] = light_chain_seq # stays constant
        out = igfold_runner.fold(
            save_path, # Output PDB file 
            sequences=sequences, # Antibody sequences
            do_refine=True,#  True, # Refine the antibody structure with PyRosetta
            do_renum=False, # True, # Renumber predicted antibody structure (Chothia) :) ! 
        )  ## do_renum=True causes bug :( 
        remove_hetero_atoms_and_hydrogens(save_path)
        # Debug example 2: 
        # Time fold 27.422064065933228
        # Time save hetero atoms 0.05717658996582031
        

def load_seqs():
    # RANIBIZUMAB 
    # http://opig.stats.ox.ac.uk/webapps/newsabdab/therasabdab/therasummary/?INN=Ranibizumab
    df = pd.read_csv('rani_display_with_sequences.csv')
    seq_ids = df['Unnamed: 0'].squeeze().values # (96846,)
    h_chains = df['full_sequence'].values # (96846,) 
    l_chain = "DIQLTQSPSSLSASVGDRVTITCSASQDISNYLNWYQQKPGKAPKVLIYFTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSTVPWTFGQGTKVEIK"
    return l_chain, h_chains, seq_ids 

def load_debug():
    hc1 = "TVSSTVYNPTVSSTVSSTVTVSSTVSSTVSSTVSSTVSSDKAKGYTTETVSSTVSSSSTVSVKGRVTVSSTVSSTVSTVSSTVSSTVKNQFSLTVSSTVSSTVSSYWGQGSTVRTVSSSTVTMLVDTSTVSSTVSSTVSSTVLSSVTATVSSTVAVYYCARTVSSTVSSTVSSTVEGHTVAPFDLSSTVSSTVADTATVSS"
    hc2 = "TVSSTVSSTVSSSTVSSTVSTVSSDKSTVSAKGYTTETVSSTVSSTVSSTVYNPTKGRVTVSSTVSSTVSKGRVTVSSTSTVSSTVSVVSSTVSSTVTMLVDTSTVSSTVSTVKSTVSSNQFSLSTVSSTVSSTVSSTVSSYWGQGSSTVSTVRTVSSTVSSSSVTVSSSSVTVLSSVTATVSSVSSTTVAAPFDLSSTVADCARTVSSTTAVYYVSSTVSSTVEGHTVTVSS"
    hc3 = "TVSSTVSSTVSTVSSTVSVKSSTVSSTVSSGRVTVTVGRVKTVTMLVDTSTVSSTVSSTVSSTVGQGVKSTVRTVSSVKTAPFDLSSTVSSVKTVADTAVYTVEGHTVTVSS"
    l_chain = "TVSSTVTMLVDTSTVSSTVSSTVSSTVKNQFSLVTVSS"
    h_chains = [hc1, hc2, hc3]
    seq_ids = [0,1,2] 
    return l_chain, h_chains, seq_ids 



def fold_parent():
    init_pyrosetta()
    igfold_runner = IgFoldRunner()
    h_chain = "EVQLVESGGGLVQPGGSLRLSCAASGYDFTHYGMNWVRQAPGKGLEWVGWINTYTGEPTYAADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAVYYCAKYPYYYGTSHWYFDVWGQGTLVTVSS"
    l_chain = "DIQLTQSPSSLSASVGDRVTITCSASQDISNYLNWYQQKPGKAPKVLIYFTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSTVPWTFGQGTKVEIK"
    fold_protein( 
        igfold_runner=igfold_runner,
        heavy_chain_seq=h_chain, 
        save_path=f"folded_parent.pdb", 
        light_chain_seq=l_chain,
    )

def fold(args):
    init_pyrosetta()
    igfold_runner = IgFoldRunner()
    if args.debug: 
        l_chain, h_chains, seq_ids  = load_debug() 
        prefix = "debug"
    else:
        l_chain, h_chains, seq_ids  = load_seqs() 
        prefix = "seq"

    if args.min_idx is None:
        args.min_idx = 0
    if args.max_idx is None:
        args.max_idx = len(h_chains)

    h_chains = h_chains[args.min_idx: args.max_idx]
    seq_ids = seq_ids[args.min_idx: args.max_idx]
    for ix, h_chain in enumerate(h_chains):
        try:
            fold_protein( 
                igfold_runner=igfold_runner,
                heavy_chain_seq=h_chain, 
                save_path=f"folded_pdbs/{prefix}{seq_ids[ix]}.pdb", 
                light_chain_seq=l_chain,
            )
        except Exception as e:
            try: 
                h_chain = h_chain.replace("X", "-") 
                fold_protein( 
                    igfold_runner=igfold_runner,
                    heavy_chain_seq=h_chain, 
                    save_path=f"folded_pdbs/{prefix}{seq_ids[ix]}.pdb", 
                    light_chain_seq=l_chain,
                )
            except Exception as e:
                print(f"failed to fold seq {seq_ids[ix]} to to exception {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    # parser.add_argument('--work_dir', default='/home/nmaus/' )  
    parser.add_argument('--min_idx', type=int, default=None ) 
    parser.add_argument('--max_idx', type=int, default=None ) 
    parser.add_argument('--debug', type=bool, default=False) 
    parser.add_argument('--parent_only', type=bool, default=True) 
     
    args = parser.parse_args() # (96846,) 
    if args.parent_only:
        fold_parent()
    else:
        fold(args) 
    # python3 fold_all.py --debug True --min_idx 1 --max_idx 2
    # jkgardner: conda activate og_lolbo_mols
    # gauss: conda activate igfold 
    # python3 fold_all.py --min_idx 0 --max_idx 5000 
