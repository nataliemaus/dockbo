import sys 
sys.path.append("../")
from dockbo.dockbo import DockBO
import argparse 
import glob 
import wandb 
import os 
os.environ["WANDB_SILENT"] = "True"
from dockbo.utils.set_seed import set_seed 
import signal 

# def log_table(tracker ):
#     cols = ["pdb_id", "true_energy"] 
#     data_list = [] 
#     for ix, score in enumerate(self.lolbo_state.top_k_scores):
#         data_list.append([ score, str(self.lolbo_state.top_k_xs[ix]) ])
#     table1 = wandb.Table(columns=cols, data=data_list)
#     tracker.log({f"table": table1}) 

    
def test_dockbo(args_dict):
    set_seed(0) 

    def handler(self, signum, frame):
        # if we Ctrl-c, make sure we log top xs, scores found
        print("Ctrl-c hass been pressed, wait while we save all collected data...")
        # log_table(tracker )
        print("Now terminating wandb tracker...")
        tracker.finish() 
        msg = "Data now saved and tracker terminated, now exiting..."
        print(msg, end="", flush=True)
        exit(1)

    signal.signal(signal.SIGINT, handler)
    fails = 0
    work_dir = args_dict['work_dir']
    score_f = args_dict['scoring_func']
    data_dir = work_dir + "dockbo/parsed_chothia_pdbs/" 
    all_ag_files = glob.glob(data_dir + "*_ag.pdb")

    tracker = wandb.init(entity="nmaus", project='lightdock-sabdab', config=args_dict) 
    print('running', wandb.run.name) 

    # for ag_id in bighatpdbs:
    pdb_id_strings = []
    for antigen_path in all_ag_files[100:]:
        # if True:
        try:
            ag_file_string = antigen_path.split("/")[-1].split(".")[0]
            pdb_id = ag_file_string[0:4]
            ex_num = int(ag_file_string.split("_")[-2][2:])
            antibody_path = data_dir + f"{pdb_id}_ex{ex_num}_ab.pdb" 
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
            save_pdb_pose_path = work_dir + f"dockbo/sabdab_best_poses{args_dict['save_poses_idx']}/{score_f}_{pdb_id}_ex{ex_num}"
            log_dict = {}  
            true_dict = oracle(
                config_x="default",
                path_to_antibody_pdb=antibody_path,
                save_pdb_pose_path=save_pdb_pose_path + "_true"
            )
            for key in true_dict.keys():
                log_dict[key + "_true"] = true_dict[key]
            print(f"pdb_id:{pdb_id}, ex_num:{ex_num}, True Pose Score:{true_dict['energy']}")
            pred_dict = oracle( 
                config_x=None,
                path_to_antibody_pdb=antibody_path,
                save_pdb_pose_path=save_pdb_pose_path + "_pred"
            )
            for key in pred_dict.keys():
                log_dict[key + "_pred"] = pred_dict[key]
            print(f"pdb_id:{pdb_id}, ex_num:{ex_num}, Pred Pose Score:{pred_dict['energy']}")
            log_dict['n_fails'] = fails 
            pdb_id_strings.append(f"{pdb_id}_ex{ex_num}") 
            log_dict['pdb_id_strings'] = pdb_id_strings 
            tracker.log(log_dict)

        except Exception as e:
            print(f"failed on pdb_id:{pdb_id}, ex_num:{ex_num}, due to", e )
            fails += 1 


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
    parser.add_argument('--absolute_max_n_steps', type=int, default=10_000 )  
    parser.add_argument('--save_poses_idx', type=int, default=3) 
    parser.add_argument('--cdr3_only_version', type=bool, default=False ) 
    parser.add_argument('--cdr3_extra_weight', type=float, default=1.0) 
    args = parser.parse_args() 
    if args.debug:
        args.save_poses_idx = 4
        args.max_n_bo_steps = 2
        args.init_n_epochs = 2
        args.absolute_max_n_steps = 2
    args_dict = vars(args) 

    # 
    # CUDA_VISIBLE_DEVICES=0 python3 test_dockbo_sabdab.py --scoring_func dfire --save_poses_idx 0 --init_bo_w_known_pose True  ?? meh problems? 


    # DONE 
    # CUDA_VISIBLE_DEVICES=0 python3 test_dockbo_sabdab.py --scoring_func dfire --save_poses_idx 3 

    # RUNNING 
    # CUDA_VISIBLE_DEVICES=1 python3 test_dockbo_sabdab.py --scoring_func dfire2 --save_poses_idx 5 
    # CUDA_VISIBLE_DEVICES=0 python3 test_dockbo_sabdab.py --scoring_func dfire --save_poses_idx 6 --cdr3_only_version True 
    # confirm that when we init w/ known pose we get ~ 0 translation, 0 rotation 
    # CUDA_VISIBLE_DEVICES=1 python3 test_dockbo_sabdab.py --scoring_func dfire --save_poses_idx 7 --init_bo_w_known_pose True  
    # XXX flawed CUDA_VISIBLE_DEVICES=0 python3 test_dockbo_sabdab.py --scoring_func dfire2 --save_poses_idx 8 --cdr3_only_version True 
    # CUDA_VISIBLE_DEVICES=0 python3 test_dockbo_sabdab.py --scoring_func dfire --save_poses_idx 9 --max_n_bo_steps 1000  

    # XXX CUDA_VISIBLE_DEVICES=1 python3 test_dockbo_sabdab.py --scoring_func dfire --save_poses_idx 10 --max_n_bo_steps 1000 --cdr3_only_version True 
    # CUDA_VISIBLE_DEVICES=1 python3 test_dockbo_sabdab.py --scoring_func dfire --save_poses_idx 11 --max_n_bo_steps 1000   
    # XXX CUDA_VISIBLE_DEVICES=0 python3 test_dockbo_sabdab.py --scoring_func dfire2 --save_poses_idx 12 --max_n_bo_steps 1000 --cdr3_only_version True  ## DOES NOT WORK!, KILLED!  
    # CUDA_VISIBLE_DEVICES=0 python3 test_dockbo_sabdab.py --scoring_func dfire2 --save_poses_idx 13 --max_n_bo_steps 1000   
    #   UPDATE: CDR3 extra weight! ... 

    # RUNNING [100:] 
    # CUDA_VISIBLE_DEVICES=0 python3 test_dockbo_sabdab.py --scoring_func dfire --save_poses_idx 14 --max_n_bo_steps 1000 --cdr3_only_version True --cdr3_extra_weight 1.0
    # CUDA_VISIBLE_DEVICES=1 python3 test_dockbo_sabdab.py --scoring_func dfire --save_poses_idx 15 --max_n_bo_steps 1000 --cdr3_only_version True --cdr3_extra_weight 2.0
    # CUDA_VISIBLE_DEVICES=0 python3 test_dockbo_sabdab.py --scoring_func dfire --save_poses_idx 14 --max_n_bo_steps 1000 --cdr3_only_version True --cdr3_extra_weight 0.5
    # CUDA_VISIBLE_DEVICES=1 python3 test_dockbo_sabdab.py --scoring_func dfire --save_poses_idx 15 --max_n_bo_steps 10000 --cdr3_only_version True --cdr3_extra_weight 1.0

    # 0 dfire v0 (no recodring cdrs stuff, no addition of default config to turbo startin gpoint )
    # 1 dfire v1 
    # 2 dfrie2 v1 
    # 4 debug !! 

    # 3 dfire v2 (no more init with best, makes too easy to converge there, also record rotation + traslation found)
    # 5 dfire2 v2 (no more init with best, makes too easy to converge there, also record rotation + traslation found)

    print(args.scoring_func)

    test_dockbo(args_dict) 
