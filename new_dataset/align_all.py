import pymolPy3
import sys 
sys.path.append("../")
import os 

def align(known_pose_id, ab_path, ab_name):
    # ** MUST RELOAD EACH TIME OR BREAKS! 
    pm = pymolPy3.pymolPy3(0)
    assert known_pose_id in [1,2]
    known_pose_path=f'1cz8_known_poses/known_ab_pose{known_pose_id}'
    known_pose = f"known_ab_pose{known_pose_id}"
    pm('delete all') 
    pm(f'load {known_pose_path}.pdb')
    pm(f'load {ab_path}')
    pm(f'super {ab_name}, {known_pose}')
    pm(f'select old_ab, model {known_pose}') # you'll have to figure out which chain is which
    pm(f'remove old_ab') 
    pm(f"save aligned{known_pose_id}/{ab_name}_aligned{known_pose_id}.pdb") 
    pm('delete all') 


def align_all(args):
    assert os.path.exists(f'1cz8_known_poses/known_ab_pose{args.known_pose_id}.pdb')
    for seq_id in range(args.n_seqs):
        ab_structure_path = f"folded_pdbs/seq{seq_id}.pdb"
        ab_name = ab_structure_path.split("/")[-1][0:-4] # f"seq{seq_id}"
        if (not os.path.exists(f"aligned{args.known_pose_id}/{ab_name}_aligned{args.known_pose_id}.pdb")) and os.path.exists(ab_structure_path): 
            try: 
                align(known_pose_id=args.known_pose_id,ab_path=ab_structure_path,ab_name=ab_name)
            except:
                pass 


if __name__ == "__main__": 
    import argparse 
    parser = argparse.ArgumentParser()  
    parser.add_argument('--n_seqs', type=int, default=100_000 ) 
    parser.add_argument('--debug', type=bool, default=False ) 
    parser.add_argument('--known_pose_id', type=int, default=1 ) 
    args = parser.parse_args() 
    assert args.known_pose_id in [1,2] 
    if args.debug: 
        args.n_seqs = 2 
    # conda activate pymol 
    # python3 align_all.py --debug True 
    align_all(args)




