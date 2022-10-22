import sys 
sys.path.append("../")
from dockbo.dockbo import DockBO
import argparse 
bighatpdbs = ['7p16', '6lfo', '7s7r', '5j57', '6knm']
BIGHAT_PDB_DIR = f"dockbo/example/pdbs/bighat_verification2/"

def test_dockbo(args_dict):
    work_dir = args_dict['work_dir']
    pdbs_dir = work_dir + BIGHAT_PDB_DIR 
    score_f = args_dict['scoring_func']

    for ag_id in bighatpdbs:
        antigen_path = pdbs_dir + f"{ag_id}_antigen.pdb"  
        oracle = DockBO(
            path_to_default_antigen_pdb=antigen_path,
            work_dir=work_dir,
            max_n_bo_steps=args_dict['max_n_bo_steps'],
            bsz=1,
            n_init=10,
            n_epochs=args_dict['n_epochs'],
            learning_rte=args_dict['lr'],
            scoring_func=score_f,
            is_receptor=args_dict['is_receptor'],
            max_n_tr_restarts=args_dict['max_n_tr_restarts'],
            verbose_config_opt=True,
            negate_difre2=args_dict['negate_difre2'], 
        )
        for ab_id in bighatpdbs:
            antibody_path  = pdbs_dir + f"{ab_id}_antibody.pdb"  
            if args_dict['negate_difre2'] and score_f == 'dfire2':
                score_f_str = 'NEG' + score_f 
            else:
                score_f_str = score_f 
            score = oracle(
                path_to_antibody_pdb=antibody_path,
                save_pdb_pose_path=work_dir + f"dockbo/dockbo_best_poses/{score_f_str}_ag{ag_id}_ab{ab_id}"
            )
            print(f"Ag:{ag_id}, Ab:{ab_id}, Score:{score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    # parser.add_argument('--avg_over', type=int, default=1 ) 
    parser.add_argument('--work_dir', default='/home/nmaus/' ) 
    parser.add_argument('--max_n_bo_steps', type=int, default=100 )
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--max_n_tr_restarts', type=int, default=2) 
    parser.add_argument('--lr', type=float, default=0.01 )
    parser.add_argument('--is_receptor', type=str, default='antibody' )
    parser.add_argument('--scoring_func', default='dfire2')

    parser.add_argument('--negate_difre2', type=bool, default=False )

    args = parser.parse_args() 
    args_dict = vars(args) 
    test_dockbo(args_dict) 


