import sys 
sys.path.append("../")
from dockbo.dockbo import DockBO
import argparse 
import glob 
import wandb 
import os 
os.environ["WANDB_SILENT"] = "True"
from dockbo.utils.set_seed import set_seed 

def test_dockbo(args_dict):
    set_seed(0) 
    work_dir = args_dict['work_dir']
    score_f = args_dict['scoring_func']
    data_dir = work_dir + "dockbo/parsed_chothia_pdbs/" 
    # all_ag_files = glob.glob(data_dir + "*_ag.pdb") 
    tracker = wandb.init(entity="nmaus", project='lightdock-sabdab', config=args_dict) 
    print('running', wandb.run.name) 

    
    all_good_ids = ['3jbe_ex1', '3k80_ex2', '4pgj_ex2', '5hm1_ex3', 
        '6h02_ex1', '6x06_ex1', '7d8b_ex1', '7lvw_ex4', '7p5v_ex1',
        '7p5w_ex2', '7p5y_ex4', '7qia_ex1', '7xji_ex1'  ] # '2xv6_ex1', '6z6v_ex2',
    
    half = len(all_good_ids) // 2
    if args.set == 1:
        good_ids = all_good_ids[0:half ] 
    elif args.set == 2:
        good_ids = all_good_ids[half: ] 
    elif args.set == 0:
        good_ids = all_good_ids 

    for ag_id in good_ids: 
        antigen_path = data_dir + ag_id + "_ag.pdb"
        try:
            oracle = DockBO( 
                path_to_default_antigen_pdb=antigen_path,
                path_to_default_antigen_restraints=None,
                work_dir=work_dir,
                max_n_bo_steps=args_dict['max_n_bo_steps'],
                bsz=args_dict['bo_bsz'],
                n_init=args_dict['n_init'],
                init_n_epochs=args_dict['init_n_epochs'],
                update_n_epochs=args_dict['update_n_epochs'],
                learning_rte=args_dict['lr'],
                scoring_func=score_f,
                is_receptor=args_dict['is_receptor'],
                max_n_tr_restarts=args_dict['max_n_tr_restarts'],
                verbose_config_opt=True,
                negate_difre2=False, 
                init_bo_w_known_pose=args_dict['init_bo_w_known_pose'],
                absolute_max_n_steps=args_dict['absolute_max_n_steps'],
                cdr3_only_version=args_dict['cdr3_only_version'],
                cdr3_extra_weight=args_dict['cdr3_extra_weight'], 
            )
        except:
            oracle = None 
        for ab_id in good_ids: 
            antibody_path = data_dir + ab_id + "_ab.pdb"
            try: 
                save_pdb_pose_path = work_dir + f"dockbo/sabdab_best_poses{args_dict['save_poses_idx']}/{score_f}_AB{ab_id}_AG{ag_id}" # WAS AB AB, but ids were still correct. 
                log_dict = {}  
                true_dict = oracle(
                    config_x="default",
                    path_to_antibody_pdb=antibody_path,
                    save_pdb_pose_path=save_pdb_pose_path + "_true"
                )
                for key in true_dict.keys():
                    log_dict[key + "_true"] = true_dict[key] 
                print(f"ab id: {ab_id}, ag id: {ag_id}, True Pose Score:{true_dict['energy']}") 
                pred_dict = oracle( 
                    config_x=None,
                    path_to_antibody_pdb=antibody_path,
                    save_pdb_pose_path=save_pdb_pose_path + "_pred"
                )
                for key in pred_dict.keys():
                    log_dict[key + "_pred"] = pred_dict[key]
                print(f"ab id: {ab_id}, ag id: {ag_id}, Pred Pose Score:{pred_dict['energy']}") 
                tracker.log(log_dict) 
            except Exception as e:
                pass 

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--work_dir', default='/home/nmaus/' ) 
    parser.add_argument('--max_n_bo_steps', type=int, default=500 ) 
    parser.add_argument('--init_n_epochs', type=int, default=60) 
    parser.add_argument('--update_n_epochs', type=int, default=2) 
    parser.add_argument('--n_init', type=int, default=100) 
    parser.add_argument('--max_n_tr_restarts', type=int, default=1)  # not currently used  
    parser.add_argument('--lr', type=float, default=0.01 ) 
    parser.add_argument('--bo_bsz', type=int, default=4) 
    parser.add_argument('--is_receptor', type=str, default='antibody' )
    parser.add_argument('--scoring_func', default='dfire')  
    parser.add_argument('--debug', type=bool, default=False)  
    parser.add_argument('--init_bo_w_known_pose', type=bool, default=False) 
    parser.add_argument('--absolute_max_n_steps', type=int, default=2_000 )  
    parser.add_argument('--save_poses_idx', type=int, default=3) 
    parser.add_argument('--cdr3_only_version', type=bool, default=False ) 
    parser.add_argument('--cdr3_extra_weight', type=float, default=1.0) 
    parser.add_argument('--set', type=int, default=1) 

    args = parser.parse_args() 
    if args.debug:
        args.save_poses_idx = 4
        args.max_n_bo_steps = 2
        args.init_n_epochs = 2
        args.absolute_max_n_steps = 2
    args_dict = vars(args) 

    # RUNNING [100:] 
    # CUDA_VISIBLE_DEVICES=0 python3 sabdab_binding_nonbinding_pairs.py --save_poses_idx 16 --set 1 --max_n_bo_steps 1000 --cdr3_only_version True --cdr3_extra_weight 1.0
    # CUDA_VISIBLE_DEVICES=1 python3 sabdab_binding_nonbinding_pairs.py --save_poses_idx 17 --set 2 --max_n_bo_steps 1000 --cdr3_only_version True --cdr3_extra_weight 1.0
    # CUDA_VISIBLE_DEVICES=0 python3 sabdab_binding_nonbinding_pairs.py --save_poses_idx 18 --set 1 --max_n_bo_steps 1000 --cdr3_only_version True --cdr3_extra_weight 2.0
    # CUDA_VISIBLE_DEVICES=1 python3 sabdab_binding_nonbinding_pairs.py --save_poses_idx 19 --set 2 --max_n_bo_steps 1000 --cdr3_only_version True --cdr3_extra_weight 2.0 

    # Possible concusion: 
    #   current DFIRE can recover true pose (ish) but not differentiate between things that bind vs. don't well
    print(args.scoring_func)

    test_dockbo(args_dict) 
