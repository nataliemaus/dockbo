
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
        fold_software='nanonet',
        work_dir=args.work_dir, 
        max_n_bo_steps=500,
        absolute_max_n_steps=2_000,
        bsz=1,
        n_init=20,
        init_n_epochs=60,
        update_n_epochs=2,
        learning_rte=0.01,
        scoring_func=args.score_f,
        is_receptor='antibody',
        max_n_tr_restarts=1, 
        negate_difre2=False,
        init_bo_w_known_pose=True,
        cdr3_only_version=False, 
        cdr3_extra_weight=1.0,
        verbose_config_opt=True, 
    )
    if args.default_pose:
        config_x = "default"
    else:
        config_x = None #  optimize 
    seq_ids = []
    energies = [] 
    for seq_id in range(args.n_seqs):
        aligned_ab_structure_path= args.work_dir + f"dockbo/new_dataset/aligned{args.known_pose_id}/seq{seq_id}_aligned{args.known_pose_id}.pdb"
        # no score_f in saved title --> must be dfire 
        prefix = args.work_dir + f"dockbo/new_dataset/"
        if not args.default_pose:
            prefix = prefix + "optimized_pose_from_" 
        save_path = prefix + f"aligned{args.known_pose_id}_combined_structures/seq{seq_id}_aligned{args.known_pose_id}_{args.score_f}_combined_structure"
        check_exists = glob.glob(save_path + "*.pdb")
        if len(check_exists) > 0: # we have already saved this one 
            print('continuing ix', seq_id)
            seq_ids.append(seq_id) 
            energies.append(float(check_exists[0].split("y")[-1][0:-4]))
        elif not args.done_only:
            if os.path.exists(aligned_ab_structure_path): 
                # try: 
                if True:
                    aligned_ab_path = remove_hetero_atoms_and_hydrogens(aligned_ab_structure_path)
                    if seq_id == 0 and args.do_known_ab:
                        true_dict = oracle(
                            config_x=config_x,
                            path_to_antibody_pdb=true_ab_path,
                            save_pdb_pose_path=true_ab_path.replace(".pdb", "_combined_structure")
                        )
                        energy = true_dict['energy']
                        seq_ids.append(-1) # -1 indicates the known strucutre! 
                        energies.append(energy)
                    return_dict = oracle(
                        config_x=config_x, 
                        path_to_antibody_pdb=aligned_ab_path,
                        save_pdb_pose_path=save_path
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
        else:
            print("not done:", seq_id)
    save(seq_ids, energies, args.score_f, 
        save_path = args.work_dir + f"dockbo/new_dataset/{args.score_f}_scores.csv")

def save(seq_ids, energies, score_f, save_path):
    # save all computed seq ids and energies 
    seq_ids = np.array(seq_ids )
    energies = np.array(energies )
    df = pd.DataFrame() 
    df['seq_id'] = seq_ids
    df[f"{score_f}_score"] = energies 
    df.to_csv(save_path, index=None)


def read_only(score_f, default_pose=True):
    seq_ids = [] 
    energies = []
    # seq5745_aligned1_combined_structure_bestenergy680.6613
    # seq5745_aligned1_dfire2_combined_structure_bestenergy680.6613
    filenm = f"aligned1_combined_structures/*_aligned1_{score_f}_combined_structure*"
    save_path = args.work_dir + f"dockbo/new_dataset/{score_f}_scores.csv"
    if not default_pose: # (optimized_pose) 
        filenm = "optimized_pose_from_" + filenm
        save_path = args.work_dir + f"dockbo/new_dataset/optimized_{score_f}_scores.csv"
    files = glob.glob(filenm) 
    for file in files: 
        seq_ids.append(int(file.split("/")[-1].split("_")[0][3:])) 
        energies.append(float(file.split("y")[-1][0:-4]))
    save(seq_ids, energies, score_f, 
        save_path = save_path)


if __name__ == "__main__": 
    import argparse 
    parser = argparse.ArgumentParser()  
    parser.add_argument('--score_f', default="dfire2" ) # "dfire, cpydock, dfire2"
    parser.add_argument('--work_dir', default='/home/nmaus/' ) 
    parser.add_argument('--known_pose_id', type=int, default=1 ) 
    parser.add_argument('--n_seqs', type=int, default=100_000 ) 
    parser.add_argument('--debug', type=bool, default=False ) 
    parser.add_argument('--done_only', type=bool, default=False ) 
    parser.add_argument('--default_pose', type=bool, default=False ) 
    parser.add_argument('--do_known_ab', type=bool, default=False ) 
    args = parser.parse_args() 
    if args.debug: 
        args.n_seqs = 2 
    if args.done_only:
        read_only(args.score_f, default_pose=args.default_pose)
    else:
        compute_scores(args) 
    #  CUDA_VISIBLE_DEVICES=1
    # CUDA_VISIBLE_DEVICES=1 python3 compute_scores.py --default_pose True --done_only True
    
    #  conda activate og_lolbo_mols 
    # python3 compute_scores.py --debug True 

    #  python3 compute_scores.py --work_dir /shared_data/

    ## *** tmux attach -t align2 
    #   doing dfire w/ actually optimizing poses 


    ### 
    # CUDA_VISIBLE_DEVICES=1 python3 compute_scores.py --default_pose True --done_only True
    # CUDA_VISIBLE_DEVICES=1 python3 plotit.py 
    # CUDA_VISIBLE_DEVICES=1 python plotit.py --score_f2 dfire








