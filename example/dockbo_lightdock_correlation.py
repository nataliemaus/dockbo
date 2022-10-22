import sys 
sys.path.append("../")
from dockbo.dockbo import DockBO
import torch 
import random 
import numpy as np 
import os 
sys.path.append("../../protein-BO/utils/")
from lightdock_oracle import LightDockOracle
import glob 
import pandas as pd 
import json 
import time 
os.environ["WANDB_SILENT"] = "True"
import wandb 


def create_restraint_file(args_dict, pdbs_dir, antibody_id=None, antigen_id=None):
    path_to_new_restrains = args_dict['work_dir'] + 'dockbo/example/temp_restraints'
    if args_dict['restraintsv3']:
        path_to_new_restrains = args_dict['work_dir'] + 'dockbo/example/temp_restraintsv3'
    if not os.path.exists(path_to_new_restrains) :
        os.mkdir(path_to_new_restrains)
    path_to_new_restrains = path_to_new_restrains + '/'

    if args_dict['receptor_is'] == 'antibody':
        antibody_char = "R"
        antigen_char = "L"
    elif args_dict['receptor_is'] == 'antigen':
        antibody_char = "L"
        antigen_char = "R"
    else:
        raise RuntimeError("invalid is receptor")
    restraints = []
    if args_dict['restrain_antibody']: 
        path_to_new_restrains = path_to_new_restrains + 'antibody' + antibody_id 
        antibody_restraint_file = pdbs_dir + f"{antibody_id}" + "_antibody_restraints.pdb"
        if args_dict['restraintsv3'] and antigen_id == '5j57':
            antibody_restraint_file = pdbs_dir + f"{antibody_id}" + "_antibody_restraints_v3.pdb"

        f = open(antibody_restraint_file, "r")
        data = f.read() 
        data = data.split('\n')
        prev_idx = 0
        for dat in data:
            if dat.startswith('ATOM'):
                dat = dat.split()
                residue = dat[3]
                chain = dat[4]
                idx = dat[5]
                try:
                    idx = int(idx)
                except:
                    pass 
                if (type(idx) == int) and (chain in ['A', 'B', 'C', 'S']) and (type(residue) == str):
                    if idx != prev_idx:
                        restraints.append(antibody_char + " " + chain + "." + residue + "." + str(idx))
                        prev_idx = idx 
        f.close()
    if args_dict['restrain_antigen']:
        path_to_new_restrains = path_to_new_restrains + 'antigen' + antigen_id 
        antigen_restraint_file = pdbs_dir + f"{antigen_id}" + "_antigen_restraints.pdb"
        if args_dict['restraintsv3'] and antigen_id == '5j57':
            antigen_restraint_file = pdbs_dir + f"{antigen_id}" + "_antigen_restraints_v3.pdb"
        f = open(antigen_restraint_file, "r")
        data = f.read() 
        data = data.split('\n')
        prev_idx = 0
        # 30, 36, 42
        for dat in data:
            if dat.startswith('ATOM'):
                dat = dat.split()
                residue = dat[3]
                chain = dat[4]
                idx = dat[5]
                if idx != prev_idx:
                    restraints.append(antigen_char + " " + chain + "." + residue + "." + idx)
                    prev_idx = idx 
        f.close()

    
    path_to_new_restrains = path_to_new_restrains + '_restraints.list' 
    if not os.path.exists(path_to_new_restrains) :
        f2 = open(path_to_new_restrains, "a") 
        for item in restraints:
            f2.write(item + '\n') 
        f2.close()  
    return path_to_new_restrains


def set_seed(seed):
    assert type(seed) == int
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = False
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed) 
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)

def start_wandb(args_dict):
    import wandb 
    tracker = wandb.init(entity="nmaus", project='lightdock-dockbo-correlation', config=args_dict) 
    print('running', wandb.run.name) 
    return tracker 


def load_antibody_data(args_dict):
    df = pd.read_csv('../ldock_opt_table.csv')
    antibodies = df['best_input_seen'].values.tolist() # do these first! (more luck they bind)
    # scores = df['best_found'].values.tolist() 

    if args_dict['include_oas_data']:
        data_dir = args_dict['work_dir'] + 'protein-BO/utils/oas_heavy_ighg_data/'
        raw_data_files = glob.glob(data_dir + f"*.csv.gz")
        tot_n_seqs = 0 
        # antibodies = []
        for data_file in raw_data_files:
            metadata = ','.join(pd.read_csv(data_file, nrows=0).columns)
            metadata = json.loads(metadata) 
            df = pd.read_csv(data_file, header=1) 
            seqs = df['sequence_alignment_aa'].values
            tot_n_seqs += len(seqs) 
            if len(seqs.shape) > 1:
                seqs = seqs.squeeze() 
            antibodies = antibodies + seqs.tolist()
            if tot_n_seqs >= args_dict['N']:
                break 

    return antibodies 


# og version 
def run(args_dict): # test dfire 
    set_seed(args_dict['seed'])
    antigen_path = f"dockbo/example/pdbs/{args_dict['antigen_id']}.pdb"
    # flip antibody and antigen below for direct comparison to way I'm running default lightdock, works :) !
    dockbo = DockBO(
        work_dir=args_dict['work_dir'],
        verbose_config_opt=False, # print progress during BO 
        max_n_bo_steps=args_dict['max_n_bo_steps'], # num bo iterations, increase to get better scores 
        bsz=1, # bo bsz (using thompson sampling)
        path_to_default_antigen_pdb=args_dict['work_dir'] + antigen_path, 
        verbose_timing=False, # print out times it takes to do stuff 
        n_init=1,
        scoring_func=args_dict['dockbo_scoring_func'],
    )
    lightdock = LightDockOracle(
        path_to_antigen_pdb=antigen_path,
        n_runs=args_dict['n_runs'],
        n_runs2=args_dict['n_runs2'],
        scoring=args_dict['lightdock_scoring_func'], # use default (fast-dfire) validated by bighat! 
        fold_software=None,
    )
    antibodies = load_antibody_data(args_dict)

    tracker = start_wandb(args_dict) 
    n_evals = 0 
    n_fails = 0 
    for antibody_seq in antibodies: 
        try:
            a = time.time() 
            pdb_path, temp_out_dir = dockbo.fold_protein(antibody_seq)
            b = time.time()  
            dockbo_score = dockbo(
                path_to_antigen_pdb=None, # No antigen path specified --> use default (aviods recomputing default structure)
                antibody_aa_seq=None, 
                path_to_antibody_pdb=pdb_path, # same_nanonet_pdb_antibody, # None, # option to pass in antibody pdb file direction, otherwise we fold seq
                config_x=None,
            )
            c = time.time() 
            lightdock_score = lightdock(
                protein_string=None,
                antibody_files=(pdb_path, temp_out_dir)
            )
            d = time.time() 
            n_evals += 1
            tracker.log({
                'dockbo_score':dockbo_score,
                'lightdock_score':lightdock_score,
                'n_evals':n_evals,
                'time_fold':b - a,
                'time_dockbo':c-b,
                'time_lightdock':d-c,
            })
        except: 
            n_fails += 1 
            tracker.log({'n_fails':n_fails,})

    tracker.finish() 
    # CUDA_VISIBLE_DEVICES=1 


# bighatpdbs = [ '7qcq', '5ivn', '7a48', '2x6m', '6i2g', '7a4t'] 
# from prody import *
# # pdbfile = '5ivn_no_waters.pdb'
# # pdb = '5ivn'
# for pdb in bighatpdbs: 
#     chA = parsePDB(pdb, chain='A')
#     writePDB(f'{pdb}_A', chA)
#     chB =  parsePDB(pdb, chain='B')
#     writePDB(f'{pdb}_B', chB)

# bighatpdbs_v2 = ['5j57', '6knm', '6lfo', '7p16', '7s7r']
# for pdb in bighatpdbs_v2:
#     chA = parsePDB(pdb, chain='A')
#     writePDB(f'{pdb}_A', chA)
#     chB =  parsePDB(pdb, chain='B')
#     writePDB(f'{pdb}_B', chB)


# v2, bughat corelllation
def bighat(args_dict): # test dfire 
    tracker = start_wandb(args_dict) 
    if args_dict['seed'] is not None:
        set_seed(args_dict['seed']) 
    # /home/nmaus/dockbo/example/pdbs/bighat_verification2/6lfo_antibody.pdb
    if args_dict['bighat_version'] == 'v1':
        bighatpdbs = ['2x6m', '6i2g', '7a4t', '7qcq', '5ivn', '7a48'] 
        pdbs_dir = f"dockbo/example/pdbs/bighat_verification/"
    elif args_dict['bighat_version'] == 'v2':
        bighatpdbs = ['7p16', '6lfo', '7s7r', '5j57', '6knm']
        pdbs_dir = f"dockbo/example/pdbs/bighat_verification2/"

    # TEST MADE FILES 
    debug = False 
    if debug:
        pdbs_dir = f"dockbo/example/pdbs/bighat_verification2/"
        pdbs_dir = args_dict['work_dir'] + pdbs_dir 
        bighatpdbs = ['7p16', '6lfo', '7s7r', '5j57', '6knm']
        for ab in bighatpdbs:
            for ag in bighatpdbs:
                create_restraint_file(args_dict, pdbs_dir, antibody_id=ab, antigen_id=ag) 
        bighatpdbs = ['2x6m', '6i2g', '7a4t', '7qcq', '5ivn', '7a48'] 
        pdbs_dir = f"dockbo/example/pdbs/bighat_verification/"
        pdbs_dir = args_dict['work_dir'] + pdbs_dir 
        for ab in bighatpdbs:
            for ag in bighatpdbs:
                create_restraint_file(args_dict, pdbs_dir, antibody_id=ab, antigen_id=ag) 
        import sys 
        sys.exit() 

    n_evals = 0  
    data_list = [] 
    for antigen_pdb_id in [bighatpdbs[args_dict['antigen_idx']] ]:
        antigen_path = pdbs_dir + f"{antigen_pdb_id}_antigen.pdb"  
        if args_dict['oracle'] == 'dockbo' or args_dict['oracle'] == 'both':
            dockbo = DockBO(
                work_dir=args_dict['work_dir'],
                verbose_config_opt=False, # print progress during BO 
                max_n_bo_steps=args_dict['max_n_bo_steps'], # num bo iterations, increase to get better scores 
                bsz=1, # bo bsz (using thompson sampling)
                path_to_default_antigen_pdb=args_dict['work_dir'] + antigen_path, 
                verbose_timing=False, # print out times it takes to do stuff 
                n_init=1,
                scoring_func=args_dict['dockbo_scoring_func'],
            )
        if args_dict['oracle'] == 'lightdock' or args_dict['oracle'] == 'both':
            lightdock = LightDockOracle(
                path_to_antigen_pdb=antigen_path,
                n_runs=args_dict['n_runs'],
                n_runs2=args_dict['n_runs2'],
                scoring=args_dict['lightdock_scoring_func'], # use default (fast-dfire) validated by bighat! 
                fold_software=None,
                remove_temp_dir=args_dict['remove_temp_dir'], 
                pdb_dir=args_dict['lightdock_temp_dir'] + f'_antigen{antigen_pdb_id}',
                receptor_is=args_dict['receptor_is'],
            )   
        
        if args_dict['antibody_idx'] is None: 
            antibody_pdbs = bighatpdbs 
        else:
            antibody_pdbs = [bighatpdbs[args_dict['antibody_idx']] ]
        for antibody_pdb_id in antibody_pdbs:
            antibody_path = args_dict['work_dir'] + pdbs_dir + f"{antibody_pdb_id}_antibody.pdb" # antibody_w_waters.pdb" 
            log_dict = {} 
            if args_dict['oracle'] == 'dockbo' or args_dict['oracle'] == 'both':
                dockbo_scores = []
                times = []
                for run in range(args_dict['avg_over']):
                    start = time.time() 
                    try:
                        dbo_score1 = dockbo(
                            path_to_antigen_pdb=None, # No antigen path specified --> use default (aviods recomputing default structure)
                            antibody_aa_seq=None, 
                            path_to_antibody_pdb=antibody_path, # same_nanonet_pdb_antibody, # None, # option to pass in antibody pdb file direction, otherwise we fold seq
                            config_x=None,
                        )
                    except:
                        dbo_score1 = -1e10000 # if faillure due to bad restrains file 
                    # get best config vector which maximized score 
                    # best_config = dockbo.best_config 
                    dockbo_scores.append(dbo_score1)
                    times.append(time.time() - start )
                dockbo_scores = np.array(dockbo_scores)
                dockbo_avg_score = dockbo_scores.mean() 
                dockbo_std_score = dockbo_scores.std()  
                log_dict['dockbo_score'] = dockbo_avg_score # mean score 
                log_dict['std_dockbo_score'] = dockbo_std_score
                log_dict['dockbo_time'] = np.array(times).mean() # avg time 
            else:
                dockbo_avg_score = None 
                dockbo_std_score = None 
            if args_dict['oracle'] == 'lightdock' or args_dict['oracle'] == 'both':
                if args_dict['restrain_antibody'] or args_dict['restrain_antigen']: 
                    restraint_file = create_restraint_file(args_dict, args_dict['work_dir'] + pdbs_dir, antibody_id=antibody_pdb_id, antigen_id=antigen_pdb_id)
                else: 
                    restraint_file = "" 
                lightdock.restraint_file = restraint_file
                start = time.time() 
                try:
                    lightdock_score = lightdock(
                        protein_string=None,
                        antibody_file=antibody_path,
                    )
                except:
                    lightdock_score = -1e10000 
                log_dict['lightdock_score'] = lightdock_score
                log_dict['lightdock_time'] = time.time() - start 
            else:
                lightdock_score = None

            n_evals += 1
            log_dict['n_evals'] = n_evals
            tracker.log(log_dict) 

            cols = ['antigen_pdb_id', 'antibody_pdb_id', 'avg_dockbo_score', 'std_dockbo_score', 'lightdock_score']
            l1 = [antigen_pdb_id, antibody_pdb_id, dockbo_avg_score, dockbo_std_score, lightdock_score] 
            data_list.append(l1)
            table1 = wandb.Table(columns=cols, data=data_list)
            tracker.log({f"results_table": table1})

    tracker.finish() 


if __name__ == "__main__": 
    import argparse 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--verbose', type=bool, default=False)  
    parser.add_argument('--seed', type=int, default=None )
    parser.add_argument('--work_dir', default='/home/nmaus/' ) 
    parser.add_argument('--antigen_id', default='6obd' ) 
    # nums steps for lightdock (more steps --> more accurate but takes way longer)
    parser.add_argument('--N', type=int, default=1_000_000 ) 
    parser.add_argument('--include_oas_data', type=bool, default=True) 
    parser.add_argument('--bighat', type=bool, default=True)  
    parser.add_argument('--max_n_bo_steps', type=int, default=10 )
    parser.add_argument('--dockbo_scoring_func', default='dfire2' ) 
    parser.add_argument('--avg_over', type=int, default=1 ) 

    parser.add_argument('--oracle', default='lightdock')  # lightdock, dockbo, or both 
    parser.add_argument('--lightdock_scoring_func', default='')  # "" --> fast dfire!, or do dfire, or dfire2
    parser.add_argument('--remove_temp_dir', type=bool, default=False) # remove swarm files, etc. made by lightdock
    parser.add_argument('--receptor_is', default='antibody')  # receptor = antibody, ligand = antigen!  
    parser.add_argument('--n_runs', type=int, default=100 )
    parser.add_argument('--n_runs2', type=int, default=200 )

    # only things to change: 
    parser.add_argument('--bighat_version', default='v2')  # v1 or v2 (which set of antibody/antigen pairs to use)
    parser.add_argument('--restrain_antibody', type=bool, default=True)
    parser.add_argument('--restrain_antigen', type=bool, default=True) 
    # parser.add_argument('--lightdock_temp_dir', default='dockbo/example/lightdock_abrestrained') 
    parser.add_argument('--antigen_idx', type=int, default=0 ) # for this antigen, do all antibodies 
    parser.add_argument('--antibody_idx', type=int, default=None ) # for this antigen, do all antibodies 
    parser.add_argument('--restraintsv3', type=bool, default=False ) # improved 5j57 restraints (v3) 

    args = parser.parse_args() 
    args_dict = vars(args) 

    if args_dict['restrain_antibody'] and args_dict['restrain_antigen']:
        args_dict['lightdock_temp_dir'] = 'dockbo/example/lightdock_bothrestrained'
    elif args_dict['restrain_antibody']:
        args_dict['lightdock_temp_dir'] = 'dockbo/example/lightdock_abrestrained'
    elif args_dict['restrain_antigen']:
        args_dict['lightdock_temp_dir'] = 'dockbo/example/lightdock_agrestrained'
    else:
        args_dict['lightdock_temp_dir'] = 'dockbo/example/lightdock_norestraints'
    
    if args_dict['restraintsv3']:
        args_dict['lightdock_temp_dir'] = args_dict['lightdock_temp_dir'] + 'v3'

    print(args_dict['lightdock_temp_dir'])
    # tmux attach -t bighat , bighat2 
    if args_dict['bighat']:
        bighat(args_dict)
    else:
        run(args_dict)

# if args_dict['bighat_version'] == 'v1':
#         bighatpdbs = ['2x6m', '6i2g', '7a4t', '7qcq', '5ivn', '7a48'] 
# elif args_dict['bighat_version'] == 'v2':
#   bighatpdbs = ['7p16', '6lfo', '7s7r', '5j57', '6knm']
# conda activate og_lolbo_mols
# python3 dockbo_lightdock_correlation.py --lightdock_scoring_func dfire2 --bighat_version v2 --antigen_idx 4 --antibody_idx 4

# done ag 0, 1 w/ all abs 
# TODO other ags with all abs ... (meh)