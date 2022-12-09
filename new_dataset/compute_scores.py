
import sys 
sys.path.append("../")
from dockbo.dockbo import DockBO
import numpy as np
import pandas as pd
from dockbo.utils.fold_utils.fold_utils import remove_hetero_atoms_and_hydrogens
import os 
import glob 

def compute_scores(args):
    antigen_path = args.work_dir + 'dockbo/new_dataset/1cz8_known_poses/known_ag_pose.pdb'
    true_ab_path = args.work_dir + f'dockbo/new_dataset/1cz8_known_poses/known_ab_pose{args.known_pose_id}.pdb'
    oracle = DockBO( 
        path_to_default_antigen_pdb=antigen_path,
        path_to_default_antigen_restraints=None,
        work_dir=args.work_dir, 
        scoring_func=args.score_f, # 'dfire', 'dfire2'
        verbose_config_opt=True,
        negate_difre2=False, 
        init_bo_w_known_pose=True, 
    )
    seq_ids = []
    energies = [] 
    for seq_id in range(args.n_seqs):
        aligned_ab_structure_path= args.work_dir + f"dockbo/new_dataset/aligned/seq{seq_id}_aligned{args.known_pose_id}.pdb"
        check_exists = aligned_ab_structure_path[0:-4] 
        check_exists = glob.glob(check_exists + "_no_het_noh_combined_structure*.pdb")
        if len(check_exists) > 0: # we have already saved this one 
            print('continuing ix', seq_id)
            seq_ids.append(seq_id) 
            energies.append(float(check_exists[0].split("y")[-1][0:-4]))
        else:
            # try: 
            if True: 
                aligned_ab_path = remove_hetero_atoms_and_hydrogens(aligned_ab_structure_path)
                if seq_id == 0:
                    true_dict = oracle(
                        config_x="default",
                        path_to_antibody_pdb=true_ab_path,
                        save_pdb_pose_path=true_ab_path.replace(".pdb", "_combined_structure")
                    )
                    energy = true_dict['energy']
                    seq_ids.append(-1) # -1 indicates the known strucutre! 
                    energies.append(energy)
                return_dict = oracle(
                    config_x="default", 
                    path_to_antibody_pdb=aligned_ab_path,
                    save_pdb_pose_path=aligned_ab_path.replace(".pdb", "_combined_structure")
                )
                energy = return_dict['energy']
                seq_ids.append(seq_id)
                energies.append(energy)
                print('computed ix', seq_id)
            # except Exception as e: 
            #     if args.debug:
            #         print(e)  
            #         import pdb 
            #         pdb.set_trace() 
            #     print('failed ix', seq_id)

    # save all computed seq ids and energies 
    seq_ids = np.array(seq_ids )
    energies = np.array(energies )
    df = pd.DataFrame() 
    df['seq_id'] = seq_ids
    df[f"{args.score_f}_score"] = energies 
    df.to_csv(args.work_dir + f"dockbo/new_dataset/{args.score_f}_scores.csv", index=None)

if __name__ == "__main__": 
    import argparse 
    parser = argparse.ArgumentParser()  
    parser.add_argument('--score_f', default="dfire" ) # "dfire, cpydock, dfire2"
    parser.add_argument('--work_dir', default='/home/nmaus/' ) 
    parser.add_argument('--known_pose_id', type=int, default=1 ) 
    parser.add_argument('--n_seqs', type=int, default=100_000 ) 
    parser.add_argument('--debug', type=bool, default=False ) 
    args = parser.parse_args() 
    if args.debug: 
        args.n_seqs = 2 
    compute_scores(args) 
    #  conda activate og_lolbo_mols 
    # python3 compute_scores.py --debug True 

    #  python3 compute_scores.py --work_dir /shared_data/



