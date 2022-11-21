import sys 
sys.path.append("../")
from dockbo.dockbo import DockBO
import argparse 


def test_dockbo(args_dict):
    work_dir = args_dict['work_dir']
    score_f = args_dict['scoring_func']

    if args_dict['bighat_version'] == 2:
        bighatpdbs = ['7p16', '7s7r', '6lfo', '6knm', '5j57']
        BIGHAT_PDB_DIR = f"dockbo/example/pdbs/bighat_verification2/" 
    elif args_dict['bighat_version'] == 1:
        bighatpdbs = ['7qcq', '5ivn', '7a48', '2x6m', '6i2g', '7a4t'] 
        BIGHAT_PDB_DIR = f"dockbo/example/pdbs/bighat_verification/" 
    else:
        raise RuntimeError("bad version")
    
    pdbs_dir = work_dir + BIGHAT_PDB_DIR 
    for ag_id in bighatpdbs:
        antigen_path = pdbs_dir + f"{ag_id}_antigen.pdb"  
        ag_rest_path = None
        if args_dict['restrain_ag']:
            ag_rest_path = work_dir + f"dockbo/example/temp_restraints/antibody{ag_id}antigen{ag_id}_restraints.list" 
        oracle = DockBO(
            path_to_default_antigen_pdb=antigen_path,
            path_to_default_antigen_restraints=ag_rest_path,
            work_dir=work_dir,
            max_n_bo_steps=args_dict['max_n_bo_steps'],
            bsz=args_dict['bo_bsz'],
            n_init=10,
            n_epochs=args_dict['n_epochs'],
            learning_rte=args_dict['lr'],
            scoring_func=score_f,
            is_receptor=args_dict['is_receptor'],
            max_n_tr_restarts=args_dict['max_n_tr_restarts'],
            verbose_config_opt=True,
            negate_difre2=args_dict['negate_difre2'], 
        )
        if args_dict['same_ab_only']:
            ab_ids = [ag_id]
        else:
            ab_ids = bighatpdbs
        for ab_id in ab_ids: 
            antibody_path = pdbs_dir + f"{ab_id}_antibody.pdb"  
            if args_dict['negate_difre2'] and score_f == 'dfire2':
                score_f_str = 'NEG' + score_f 
            else:
                score_f_str = score_f 
            ab_rest_path = None 
            if args_dict['restrain_ab']:
                ab_rest_path = work_dir + f"dockbo/example/temp_restraints/antibody{ab_id}antigen{ab_id}_restraints.list" 
            config_x = None
            if args_dict['default_config']:
                config_x = 'default'
            score = oracle(
                config_x=config_x,
                path_to_antibody_pdb=antibody_path,
                antibody_restraints_path=ab_rest_path,
                save_pdb_pose_path=work_dir + f"dockbo/dockbo_best_poses{args_dict['save_poses_idx']}/{score_f_str}_ag{ag_id}_ab{ab_id}"
            )
            print(f"Ag:{ag_id}, Ab:{ab_id}, Score:{score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    # parser.add_argument('--avg_over', type=int, default=1 ) 
    parser.add_argument('--work_dir', default='/home/nmaus/' ) 
    parser.add_argument('--max_n_bo_steps', type=int, default=500 ) # 1_000
    parser.add_argument('--n_epochs', type=int, default=40) # 60 
    parser.add_argument('--max_n_tr_restarts', type=int, default=1)   
    parser.add_argument('--lr', type=float, default=0.01 ) 

    parser.add_argument('--is_receptor', type=str, default='antibody' )
    parser.add_argument('--negate_difre2', type=bool, default=False )
    parser.add_argument('--restrain_ag', type=bool, default=False)
    parser.add_argument('--restrain_ab', type=bool, default=False )

    parser.add_argument('--bighat_version', type=int, default=2 )
    
    parser.add_argument('--scoring_func', default='dfire2') 
    parser.add_argument('--save_poses_idx', type=int, default=10) 
    parser.add_argument('--bo_bsz', type=int, default=10) 

    parser.add_argument('--default_config', type=bool, default=False)
    parser.add_argument('--same_ab_only', type=bool, default=True)


    # poses4 = dfire2 
    # poses3 = dfire

    # W/ UPDATES in notes: 
    # poses5 = dfire 
    # poses6 = dfire2 !!TURNS OUT DFIRE2 RESTRAINTS ARE FIXED!! :) 

    # poses 8 = dfire w/ constraints 0.05, 0.2 ! 
    # poses 9 = dfire w/ constraints 0.05, 0.05
    # poses 10 = dfire w/ constraints 0.05 0.05 bo_bsz 10 
    # poses 11 = dfire w/ constraints 0.05 0.05 bo_bsz 10 
    # poses 12 = dfire w/ constraints 0.05 0.05 bo_bsz 100 

    # poses 13, no more bo, try get scores of default poses for DFIRE 
    # poses 14, no more bo, try get scores of default poses for DFIRE2 

    # So far I have confirmed that w/ dfire, just maximizing restraints recovers exact pose! 
    #   TODO: confirm that for dfire2 

    # CUDA_VISIBLE_DEVICES=1 python3 test_dockbo.py --scoring_func dfire --bo_bsz 100 --save_poses_idx 12
    #  --negate_difre2 True 

    # 
    # CUDA_VISIBLE_DEVICES=0 python3 test_dockbo.py --scoring_func dfire2 --save_poses_idx 10 --bighat_version 2 
    # CUDA_VISIBLE_DEVICES=0 python3 test_dockbo.py --scoring_func dfire --save_poses_idx 16 --bighat_version 1 

    args = parser.parse_args() 
    args_dict = vars(args) 

    print(args.scoring_func)

    DEBUG_MODE = False 
    if DEBUG_MODE:
        args_dict['max_n_bo_steps'] = 2
        args_dict['n_epochs'] = 2
        args_dict['max_n_tr_restarts'] = 1

    test_dockbo(args_dict) 


# DFIRE TRUE DIAGONAL ENERGIES:
# Ag:7p16, Ab:7p16, Score:26.563594818115234
# Ag:7s7r, Ab:7s7r, Score:28.649986267089844
# Ag:6lfo, Ab:6lfo, Score:28.45932388305664
# Ag:6knm, Ab:6knm, Score:47.854042053222656
# Ag:5j57, Ab:5j57, Score:19.279605865478516