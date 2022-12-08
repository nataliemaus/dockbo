import pymolPy3
import sys 
sys.path.append("../")
import os 
# from dockbo.utils.fold_utils.fold_utils import remove_hetero_atoms_and_hydrogens

def align(pm, known_pose_id, new_ab_structure_path, new_ab_structure):
    assert known_pose_id in [1,2]
    known_pose_path=f'1cz8_known_poses/known_ab_pose{known_pose_id}'
    known_pose = f"known_ab_pose{known_pose_id}"
    pm('delete all') 
    pm(f'load {known_pose_path}.pdb')
    pm(f'load {new_ab_structure_path}.pdb')
    pm(f'super {new_ab_structure}, {known_pose}')
    pm(f'select old_ab, model {known_pose}') # you'll have to figure out which chain is which 
    pm(f'remove old_ab') 
    pm(f"save {new_ab_structure_path}_aligned{known_pose_id}.pdb") 
    pm(f'delete all') 


def align_all(n_seqs=100_000):
    pm = pymolPy3.pymolPy3(0)
    for seq_id in range(n_seqs):
        new_path = f"folded_pdbs/seq{seq_id}"
        ab_name = f"seq{seq_id}"
        try: 
            # if not os.path.exists(f"{new_path}_aligned{1}.pdb"):
            if True: 
                align(
                    pm=pm, 
                    known_pose_id=1,
                    new_ab_structure_path=new_path, # save_path.replace(".pdb", "")
                    new_ab_structure=ab_name, # save_path.replace(".pdb", "")
                )
                # remove_hetero_atoms_and_hydrogens(f"{new_path}_aligned1.pdb") 
            # if not os.path.exists(f"{new_path}_aligned{2}.pdb"):
            if True: 
                align(
                    pm=pm, 
                    known_pose_id=2,
                    new_ab_structure_path=new_path, # save_path.replace(".pdb", "")
                    new_ab_structure=ab_name, # save_path.replace(".pdb", "")
                )
        except:
            pass 


if __name__ == "__main__": 
    import argparse 
    parser = argparse.ArgumentParser()  
    parser.add_argument('--n_seqs', type=int, default=100_000 ) 
    parser.add_argument('--debug', type=bool, default=False ) 
    args = parser.parse_args() 
    if args.debug: 
        args.n_seqs = 2 
    # conda activate pymol 
    # python3 align_all.py
    align_all(n_seqs = args.n_seqs)




