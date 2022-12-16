import sys 
sys.path.append("../")
from dockbo.dockbo import DockBO
import numpy as np
import pandas as pd
from dockbo.utils.fold_utils.fold_utils import remove_hetero_atoms_and_hydrogens
import glob 

def compute_scores(args):
    antigen_path = args.work_dir + args.antigen_path
    antibody_dir = args.work_dir + args.aligned_antibodies_dir
    oracle = DockBO( 
        path_to_default_antigen_pdb=antigen_path,
        work_dir=args.work_dir, 
        scoring_func=args.score_f,
        init_bo_w_known_pose=True, # initialize pose optimization w/ known pose
        verbose_config_opt=True, # print updates during pose optimization
    )
    if args.optimize_pose: # optimize pose to maximize score 
        config_x = None 
        csv_save_path = args.work_dir + f"dockbo/tutorial/computed_scores/optimized_{args.score_f}_scores.csv"
    else: # use the pose of the input pdb file and just compute the score 
        config_x = "default" 
        csv_save_path = args.work_dir + f"dockbo/tutorial/computed_scores/{args.score_f}_scores.csv"
    # grab all aligned antibodies from file (aligned using pymol)
    antibody_paths = glob.glob(antibody_dir + "*aligned1.pdb")
    # where we will save computed ab + ag structures that scores are computed from
    save_dir = args.work_dir + 'dockbo/tutorial/combined_structures/'
    energies = []
    seq_ids = [] 
    for ix, antibody_path in enumerate(antibody_paths):
        seq_id = int(antibody_path.split("/")[-1].split("_")[0][3:]) 
        # remove hydrogens and hetero atoms to prevent breaking lightdock 
        antibody_path = remove_hetero_atoms_and_hydrogens(antibody_path)
        if args.optimize_pose:
            save_path = save_dir + f"seq{seq_id}_{args.score_f}_optimized_pose"
        else:
            save_path = save_dir + f"seq{seq_id}_{args.score_f}_aligned_pose" 
        return_dict = oracle(
            config_x=config_x, 
            path_to_antibody_pdb=antibody_path,
            save_pdb_pose_path=save_path # where to save combined structure pdb file
        )
        energy = return_dict['energy']
        seq_ids.append(seq_id)
        energies.append(energy)
        if ix % args.save_frequency == 0:
            save(
                seq_ids=seq_ids, 
                energies=energies, 
                score_f=args.score_f, 
                save_path=csv_save_path,
            ) 
    save(
        seq_ids=seq_ids, 
        energies=energies, 
        score_f=args.score_f, 
        save_path=csv_save_path,
    ) 

def save(seq_ids, energies, score_f, save_path):
    # save all computed seq ids and energies 
    seq_ids = np.array(seq_ids )
    energies = np.array(energies )
    df = pd.DataFrame() 
    df['seq_id'] = seq_ids
    df[f"{score_f}_score"] = energies 
    df.to_csv(save_path, index=None)

if __name__ == "__main__": 
    import argparse 
    parser = argparse.ArgumentParser()  
    # directory where dockbo/ repo is located
    parser.add_argument('--work_dir', default='/home/nmaus/' ) 
    # file containing all folded antibody pdb files which have been aligned with pymol (aligned to the known ab structure)
    parser.add_argument('--aligned_antibodies_dir', default="dockbo/new_dataset/aligned1/" ) 
    # path to antigen pdb file 
    parser.add_argument('--antigen_path', default="dockbo/tutorial/1cz8_known_ag_pose.pdb" ) 
    # scoring function to use (dfire, difre2, or cpydock supported)
    parser.add_argument('--score_f', default="dfire" ) # "dfire, cpydock, dfire2"
    # rather than use the static aligned pose, start with the aligend pose and but allow changes to the pose to maximzie the score 
    parser.add_argument('--optimize_pose', type=bool, default=False ) 
    # how often to save collected energies to a csv (ie every ten computations)
    parser.add_argument('--save_frequency', type=int, default=100) 
    args = parser.parse_args() 
    assert args.score_f in ["dfire", "dfire2", "cpydock"]
    compute_scores(args) 

    # example command:
    # python3 tutorial.py --score_f dfire2 --work_dir /home/nmaus/






