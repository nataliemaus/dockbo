import torch 
import numpy as np 
from scipy.spatial import Delaunay
from scipy.spatial import KDTree
from dockbo.utils.lightdock_torch_utils.lightdock_constants import (
    SWARM_DISTANCE_TO_SURFACE_CUTOFF
)

def is_valid_config(config_x, check_validity_utils):
    ''' config_x = 27 D vector specifying ligand location + orientation
        method returns a boolean indicating whether config_x is valid
        config_x is valid iff it translation components (config_x[0:3]) 
        move ligand center to a location that lies outside of the receptor 
        (ie the ligand can't be totally inside of the receptor)
    '''
    if torch.is_tensor(config_x):
        config_x = config_x.detach().cpu().numpy() 
    elif type(config_x) == list:
        config_x = np.array(config_x)
    lig_center = config_x[0:3] 
    # receptor_kd_tree = KDTree(receptor_coords )
    receptor_kd_tree = check_validity_utils['receptor_kd_tree'] 
    # count number of atoms in receptor kd tree that are within SWARM_DISTANCE_TO_SURFACE_CUTOFF of ligand center, 
    num_receptor_atoms_close_to_lig_center = len(receptor_kd_tree.query_ball_point(lig_center, SWARM_DISTANCE_TO_SURFACE_CUTOFF))
    # a vaid ligand center has NO receptor atoms too closeby 
    is_valid =  num_receptor_atoms_close_to_lig_center <= 0 

    return is_valid


# Old method for taking final ligand and receptor poses and checking
#   if they overlap at all (stronger constraint than above)
def check_overlap(
    ligand_pose,
    receptor_pose,
    check_hull=True,
    check_within3=True,
    verbose=False,
):
    is_overlap = False 
    if check_hull:
        # ! HERE, CHECK FOR OVERLAP IN COORDS! 
        ligand_pose_np = ligand_pose.squeeze().detach().cpu().numpy()
        receptor_pose_np = receptor_pose.squeeze().detach().cpu().numpy()

        receptor_convex_hull = Delaunay(receptor_pose_np)
        bad, good = 0, 0
        for point in ligand_pose_np: 
            if receptor_convex_hull.find_simplex(point) < 0:
                good += 1
            else:
                bad += 1
        prcnt_bad = bad/(good+bad)
        check_str = "within convex hull of receptor"
        if verbose:
            print(f"{bad}/{good+bad}={prcnt_bad:.4f} of ligand atoms {check_str}.")
        if bad > 0:
            is_overlap = True 

    if check_within3:
         # ! HERE, CHECK FOR OVERLAP IN COORDS! 
        ligand_pose_np = ligand_pose.squeeze().detach().cpu().numpy()
        receptor_pose_np = receptor_pose.squeeze().detach().cpu().numpy()

        receptor_kd_tree = KDTree(receptor_pose_np )
        bad, good = 0, 0
        for point in ligand_pose_np: 
            if len(receptor_kd_tree.query_ball_point(point, SWARM_DISTANCE_TO_SURFACE_CUTOFF)) <= 0:
                good += 1
            else:
                bad += 1
        prcnt_bad = bad/(good+bad)
        check_str = "within 3.0 of any receptor atom"
        if verbose:
            print(f"{bad}/{good+bad}={prcnt_bad:.4f} of ligand atoms {check_str}.")
        if bad > 0:
            is_overlap = True 
    
    return is_overlap
