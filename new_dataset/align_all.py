import pymolPy3


def align(pm, known_pose, new_ab_structure):
    pm = pymolPy3.pymolPy3(0)
    pm('delete all')
    # Load everything 
    pm(f'load {known_pose}.pdb')
    pm(f'load {new_ab_structure}.pdb')
    pm(f'super {new_ab_structure}, {known_pose}')
    pm(f'select old_ab, model {known_pose} and chain XXX') # you'll have to figure out which chain is which 
    pm(f'remove old_ab') 
    pm('save new_structure.pdb')
    pm('delete all') 

align(
    pm=pm, 
    known_pose='1cz8_known_poses/known_ab_pose1', 
    new_ab_structure=save_path.replace(".pdb", "")
)