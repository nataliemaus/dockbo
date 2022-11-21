
### REMOVING points within receptor as potential ligand centers 
#   first 3 points in 27 d vector = coords = center of ligand 

# Note: SWARM_DISTANCE_TO_SURFACE_CUTOFF = 3.0 

# lightdock.prep.starting_points line 197 
# lightdock.prep.poses 


# op_vector = [tx, ty, tz, q.w, q.x, q.y, q.z, ANMreceptor, ANMligand] 
        # x, y, z = 3D vector location (v - ligand_origin)
        # q's = rotations of ligand 
        # x,y,z must be within bounding box -30,30 AND outside of receptor! 
        #   ie secondary constraint that x,y,z cannot be point within receptor!! 

center2 = [0,0,0] # bad example 
center1 = [19.2995851, -17.5295151, 10.1282317]
import scipy 
Hull = scipy.spatial.ConvexHull(mol_coords, incremental=False, qhull_options=None)
# len(Hull.vertices) = 45 == way less than 594
vertex_pts = mol_coords[Hull.vertices] # (45, 3)

# returns -1 when center outside hull, otherwise number of triantles it intersects
#   could use as constraint? We shouldn't need to learn constraint here though
#   it is defined literally! Can just toss these points out in ts!! 
from scipy.spatial import Delaunay
receptor_convex_hull = Delaunay(mol_coords)
is_valid = receptor_convex_hull.find_simplex(center)<0 



def is_valid_config_x(self, config_x, receptor_kd_tree):
        if type(config_x) != list:
                config_x = config_x.tolist() 
        # Return true iff config_x[0:3] = (x,y,z) lies outside of receptor 
        return len(receptor_kd_tree.query_ball_point(config_x[0:3], SWARM_DISTANCE_TO_SURFACE_CUTOFF)) <= 0 



 lig_verticies = ligand_pose_np[lig_hull.vertices] 
rec_verticies = receptor_pose_np[rec_hull.vertices] 
from Geometry3D import *
lig_poly = ConvexPolygon(lig_verticies)
rec_poly = ConvexPolygon(rec_verticies)
inter = intersection(lig_poly,rec_poly)

# from shapely.geometry import Polygon

from shapely.geometry import Polygon
p1 = Polygon([(0,0), (1,1), (1,0)])
p2 = Polygon([(0,1), (1,0), (1,1)])
print(p1.intersects(p2))

import pdb 
pdb.set_trace() 
overlap1 = clouds_overlap(receptor_pose.squeeze().detach().cpu().numpy(), ligand_pose_coords)







        # check_validity_utils['receptor_translation'] = np.array(self.rec_translation)
        # rt = self.rec_translation.numpy() 
        # rc1 = self.receptor_coords
        # rc2 = receptor_coords 
        # rc1 - rt == rc2 !! 
        from scipy.spatial import distance
        # Calculate receptor and ligand max diameters
        distances_matrix_rec = distance.pdist(self.receptor.representative(False))
        receptor_max_diameter = np.max(distances_matrix_rec)
        distances_matrix_lig = distance.pdist(self.ligand.representative())
        ligand_max_diameter = np.max(distances_matrix_lig)

        check_validity_utils['receptor_max_diameter'] = receptor_max_diameter
        check_validity_utils['ligand_max_diameter'] = ligand_max_diameter 