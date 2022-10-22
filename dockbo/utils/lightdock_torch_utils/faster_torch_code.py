import torch 
from pytorch3d import transforms
from pathlib import Path
from dockbo.utils.lightdock_torch_utils.scoring.functions import ScoringFunction
from dockbo.utils.lightdock_torch_utils.lightdock_constants import (
    TH_DEVICE,
    TH_DTYPE,
    DEFAULT_CONTACT_RESTRAINTS_CUTOFF,
    DEFAULT_LIGHTDOCK_PREFIX,
    DEFAULT_ANM_RMSD,
    DEFAULT_MASK_FILE,
    STARTING_NM_SEED,
)
from dockbo.utils.lightdock_torch_utils.setup_sim import read_input_structure
from dockbo.utils.lightdock_torch_utils.structure.nm import calculate_nmodes
from dockbo.utils.lightdock_torch_utils.boundaries import get_default_box
from dockbo.utils.lightdock_torch_utils.PDBIO import write_pdb_to_file, write_mask_to_file
import copy 


def get_bbox_tensor(setup):
    bounding_box = get_default_box(setup['use_anm'], setup['anm_rec'], setup['anm_lig'])
    bounding_box = bounding_box.boundaries 
    bounding_box = [[bbox.lower_limit, bbox.upper_limit,] for bbox in bounding_box]
    bounding_box = torch.tensor(bounding_box)
    return bounding_box


def get_coords(adapter):
    receptor_coords = torch.cat([torch.from_numpy(c.coordinates).to(device=TH_DEVICE, dtype=TH_DTYPE) for c in adapter.receptor_model.coordinates])
    ligand_coords = torch.cat([torch.from_numpy(c.coordinates).to(device=TH_DEVICE, dtype=TH_DTYPE) for c in adapter.ligand_model.coordinates])
    return receptor_coords, ligand_coords 


def get_receptor_indicies(adapter):
    res_index = []
    atom_index = []
    for o in adapter.receptor_model.objects:
        res_index.append(o.residue_index)
        atom_index.append(o.atom_index)
    last = res_index[-1]
    return res_index, atom_index, last 


def get_ligand_indicies(adapter, last):
    res_index = []
    atom_index = []
    for o in adapter.ligand_model.objects:
        res_index.append(o.residue_index + last)
        atom_index.append(o.atom_index)
    return res_index, atom_index


def get_indicies(adapter, receptor_indicies):
    if receptor_indicies is None:
        receptor_indicies = get_receptor_indicies(adapter) 
    res_index, atom_index, last = receptor_indicies
    ligand_res_indicies, ligand_atom_indicies = get_ligand_indicies(adapter, last)
    res_index = res_index + ligand_res_indicies
    atom_index = atom_index + ligand_atom_indicies
    res_index = torch.tensor(res_index, dtype=torch.long, device=TH_DEVICE)
    atom_index = torch.tensor(atom_index, dtype=torch.long, device=TH_DEVICE)
    
    return res_index, atom_index 


def update_adapter(adapter, receptor, ligand, new_receptor=False, new_ligand=False ):
    # adapter = DefinedModelAdapter(receptor, ligand, None, None) 
    new_adapter = copy.deepcopy(adapter)
    # set new antibody ligand 
    if new_ligand:
        new_adapter.set_ligand_model(ligand, None)
    if new_receptor:
        new_adapter.set_receptor_model(receptor, None)
    for i in range(len(receptor.atom_coordinates)):
        new_adapter.receptor_model.coordinates[i].coordinates = receptor.atom_coordinates[i].coordinates
        new_adapter.ligand_model.coordinates[i].coordinates = ligand.atom_coordinates[i].coordinates

    return new_adapter 


def calculate_anm(structure, num_nmodes, rmsd, seed): # , file_name):
    """Calculates ANM for representative structure"""
    original_file_name = structure.structure_file_names[structure.representative_id]
    # We have to use the parsed structure by LightDock
    parsed_lightdock_structure = Path(original_file_name).parent / Path(
        DEFAULT_LIGHTDOCK_PREFIX % Path(original_file_name).name
    )
    modes = calculate_nmodes(parsed_lightdock_structure, num_nmodes, rmsd, seed, structure)
    return torch.tensor(modes, dtype=TH_DTYPE, device=TH_DEVICE)


def save_lightdock_structure(structure):
    """Saves the structure parsed by LightDock"""
    for structure_index, file_name in enumerate(structure.structure_file_names):
        moved_file_name = Path(file_name).parent / Path(
            DEFAULT_LIGHTDOCK_PREFIX % Path(file_name).name
        )
        if not moved_file_name.exists():
            # only write lightdock file if it doesn't alreay exist 
            write_pdb_to_file(structure, moved_file_name, structure[structure_index])
        mask_file_name = Path(file_name).parent / Path(
            DEFAULT_MASK_FILE % Path(file_name).stem
        )
        if not mask_file_name.exists():
             # only write corresponding mask file if it doesn't alreay exist 
            write_mask_to_file(structure.nm_mask, mask_file_name)


def prep_receptor(setup):
    receptor = read_input_structure(setup['receptor_pdb'], setup['noxt'], setup['noh'], setup['now'], setup['verbose_parser'])
    rec_translation = receptor.move_to_origin()
    save_lightdock_structure(receptor)
    receptor.n_modes = calculate_anm(receptor, setup['anm_rec'], DEFAULT_ANM_RMSD, STARTING_NM_SEED) # , DEFAULT_REC_NM_FILE)
    rec_translation = torch.tensor(rec_translation).float() # (3,) gives translation of receptor 
    return receptor, rec_translation


def prep_ligand(setup):
    ligand = read_input_structure(setup['ligand_pdb'], setup['noxt'], setup['noh'], setup['now'], setup['verbose_parser'])
    lig_translation = ligand.move_to_origin()
    save_lightdock_structure(ligand)
    ligand.n_modes = calculate_anm(ligand, setup['anm_lig'], DEFAULT_ANM_RMSD, STARTING_NM_SEED) # , DEFAULT_LIG_NM_FILE)
    lig_translation = torch.tensor(lig_translation).float()
    return ligand, lig_translation 


def calculate_dfire2_torch(
    res_index, # int[:]
    atom_index, # int[:]
    coordinates, # np.ndarray[np.float64_t, ndim=2] 
    potentials, # np.ndarray[np.float64_t, ndim=3] 
    n, # np.uint32_t 
    interface_cutoff, # np.float64_t
):
    th_m1 = torch.tensor(-1.0, dtype=TH_DTYPE, device=TH_DEVICE) 
    dist_matrix = torch.cdist(coordinates, coordinates) * 2
    # Make sure we only condiser j > i (upper triangular)
    is_, js_ = torch.where(torch.ones((n,n), device=TH_DEVICE, dtype=TH_DTYPE))
    dist_matrix = torch.where(js_.reshape(n,n) > is_.reshape(n,n), dist_matrix, th_m1)
    # Make sure res_index[i] != res_index[j]
    res_index_is = res_index[is_].reshape(n,n)
    res_index_js = res_index[js_].reshape(n,n)
    dist_matrix = torch.where(res_index_is != res_index_js, dist_matrix, th_m1)
    # make sure d <= interface_cutoff:
    dist_matrix_interface = torch.where(dist_matrix <= interface_cutoff, dist_matrix, th_m1)
    # Grab interface receptor, ligand 
    interface_receptor, interface_ligand = torch.where(dist_matrix_interface >= 0)
    interface_receptor = interface_receptor.tolist()
    interface_ligand = interface_ligand.tolist()
    # Convert to int and make sure d < 30
    dist_matrix = dist_matrix.long()
    dist_matrix = torch.where(dist_matrix < 30, dist_matrix, th_m1.long())
    # Index into potential with atom indexes 
    #   (energy +=  potentials.item(atom_index[i], atom_index[j], d))
    good_is, good_js = torch.where(dist_matrix >= 0)
    good_dists = dist_matrix[good_is, good_js]
    atom_index_is = atom_index[good_is]
    atom_index_js = atom_index[good_js]
    energy = potentials[atom_index_is,atom_index_js,good_dists].sum()/100.0

    return energy, set(interface_receptor), set(interface_ligand)

def compute_ligand_receptor_poses(
    ld_sol,
    setup,
    receptor,
    ligand,
    receptor_coords,
    ligand_coords,
):
    ## Unpack solution into matrices
    # First 3 elements are a translation vector for the ligand
    translation = torch.tensor(ld_sol[:3], device=TH_DEVICE, dtype=TH_DTYPE)
    # Next 4 elements define a quaternion for rotating the ligand
    rotation = torch.tensor(ld_sol[3:3 + 4], device=TH_DEVICE, dtype=TH_DTYPE)
    # Next anm_rec elements have to do with moving the receptor atoms slightly to account for deflection
    rec_extent = torch.tensor(ld_sol[3 + 4:3 + 4 + setup['anm_rec']], device=TH_DEVICE, dtype=TH_DTYPE)
    # Next anm_lig elements have to do with moving the ligand atoms slightly to account for deflection
    lig_extent = torch.tensor(ld_sol[-setup['anm_lig']:], device=TH_DEVICE, dtype=TH_DTYPE)
    # Update receptor and ligand coordinates based on ANM model
    receptor_pose = receptor_coords[receptor.nm_mask, :] + (rec_extent.unsqueeze(-1).unsqueeze(-1) * receptor.n_modes).sum(-3, keepdim=True)
    ligand_pose = ligand_coords[ligand.nm_mask, :] + (lig_extent.unsqueeze(-1).unsqueeze(-1) * ligand.n_modes).sum(-3, keepdim=True)
    # Rotate and translate ligand
    ligand_pose = transforms.quaternion_apply(rotation, ligand_pose) + translation

    return receptor_pose, ligand_pose

def dfire2_torch(
    ld_sol,
    receptor_coords,
    ligand_coords,
    res_index,
    atom_index,
    potentials,
    receptor_restraints,
    ligand_restraints,
    setup,
    receptor,
    ligand,
):  
    receptor_pose, ligand_pose = compute_ligand_receptor_poses(
        ld_sol,
        setup,
        receptor,
        ligand,
        receptor_coords,
        ligand_coords,
    ) # torch.Size([1, 6625, 3]), torch.Size([1, 594, 3]) 
    potentials_energy = torch.tensor(potentials.energy, device=TH_DEVICE, dtype=TH_DTYPE).reshape(167, 167, 30)
    all_coords = torch.cat((receptor_pose, ligand_pose), dim=-2)
    molecule_length = len(res_index)
    energy, _, interface_ligand = calculate_dfire2_torch(res_index, atom_index, all_coords.squeeze(-3), potentials_energy, molecule_length, DEFAULT_CONTACT_RESTRAINTS_CUTOFF)
    # restraints factors alwats return 0 when there are no addional constraints 
    perc_lig_restraints = ScoringFunction.restraints_satisfied(ligand_restraints, set(interface_ligand))
    perc_rec_restraints = ScoringFunction.restraints_satisfied(receptor_restraints, set(receptor_coords))
    final_score = energy + perc_lig_restraints * energy + perc_rec_restraints * energy
    # negate energy to obtain maximization problem! 
    return -1*final_score

# adapter.receptor_model,
# adapter.ligand_model,
def get_dfire_score(
    ld_sol,
    setup,
    receptor,
    ligand,
    receptor_coords,
    ligand_coords,
    dfire_oracle,
    lig_numbs,
    rec_numas,
    adapter,
):
    receptor_pose, ligand_pose = compute_ligand_receptor_poses(
        ld_sol,
        setup,
        receptor,
        ligand,
        receptor_coords,
        ligand_coords,
    )
    score = dfire_oracle(
        lig_numbs=lig_numbs,
        rec_numas=rec_numas,
        receptor_coordinates=receptor_pose,
        ligand_coordinates=ligand_pose,
    )

    # old_score = get_dfire_score_old(
    #     ld_sol,
    #     setup,
    #     adapter.receptor_model,
    #     adapter.ligand_model,
    #     receptor_coords,
    #     ligand_coords,
    #     dfire_oracle,
    #     receptor_pose,
    #     ligand_pose,
    # )
    # print(f"og score: {old_score}, torch score: {score}")
    # ALWAYS THE SAME! 

    return score 


def get_numbs(ligand):
    rnumbs, anumbs = [], []
    for atom in ligand.objects:
        rnumbs.append(atom.dfire_residue_index)
        anumbs.append(atom.atom_index)
    rnumbs, anumbs = torch.tensor(rnumbs).int(), torch.tensor(anumbs).int()
    return rnumbs, anumbs 


def get_anumas(receptor):
    rnumas, anumas = [], []
    for atom in receptor.objects:
        rnumas.append(atom.dfire_residue_index)
        anumas.append(atom.atom_index)
    rnumas, anumas = torch.tensor(rnumas).int(), torch.tensor(anumas).int()
    return rnumas, anumas


def calculate_dfire_torch(
    rec_numas,
    receptor_coordinates,
    lig_numbs,
    ligand_coordinates, 
    dfire_dist_to_bins,
    dfire_energy,
):
    dist_matrix = torch.cdist(receptor_coordinates.squeeze(), ligand_coordinates.squeeze())
    atom_indexes = torch.where(dist_matrix <= 15.0)
    rec_idxs = atom_indexes[0]
    lig_idxs = atom_indexes[1]
    if len(rec_idxs) == 0:
        return 0.0 
    dist_matrix *= 2.0
    dist_matrix -= 1.0
    rnumas, anumas = rec_numas 
    rnumbs, anumbs = lig_numbs
    dist_matrix = dist_matrix.int() 
    rnumas = rnumas[rec_idxs]
    anumas = anumas[rec_idxs]
    rnumbs = rnumbs[lig_idxs]
    anumbs = anumbs[lig_idxs]
    idxs = torch.cat((rec_idxs.unsqueeze(-1), lig_idxs.unsqueeze(-1)), dim=-1) # .detach().cpu() # torch.Size([7514, 2])
    dists = dist_matrix[idxs[:,0], idxs[:,1]] 
    dfire_dist_to_bins = torch.tensor(dfire_dist_to_bins).int() 
    dfire_bins = dfire_dist_to_bins[dists.long()] - 1
    dfire_bins = torch.clamp(dfire_bins, dfire_bins.min(), 19.0)
    energies = dfire_energy[rnumas.long(), anumas.long(), rnumbs.long(), anumbs.long(), dfire_bins.long()] # torch.Size([8752])
    score = (energies.sum() * 0.0157 - 4.7) * -1.0 
    return score  



# def get_dfire_score_old(
#     ld_sol,
#     setup,
#     receptor,
#     ligand,
#     receptor_coords,
#     ligand_coords,
#     dfire_oracle,
#     receptor_pose,
#     ligand_pose,
# ):
#     if torch.is_tensor(ld_sol):
#         ld_sol = ld_sol.detach().cpu()
#         ld_sol = ld_sol.numpy() 
#     if torch.is_tensor(receptor_pose):
#         receptor_pose = receptor_pose.detach().cpu()
#         receptor_pose= receptor_pose.numpy() 
#     if torch.is_tensor(ligand_pose,):
#         ligand_pose = ligand_pose.detach().cpu()
#         ligand_pose = ligand_pose.numpy() 
    

#     energy = calculate_dfire_og(
#         receptor=receptor,
#         receptor_coordinates=receptor_pose.squeeze() ,
#         ligand=ligand,
#         ligand_coordinates=ligand_pose.squeeze() , 
#         dfire_dist_to_bins=dfire_oracle.potential.dfire_dist_to_bins,
#         dfire_energy=dfire_oracle.potential.og_dfire_energy,
#         interface_cutoff=3.9
#     )
#     return energy 


# import scipy 
# import numpy as np 
# def calculate_dfire_og(
#     receptor,
#     receptor_coordinates,
#     ligand,
#     ligand_coordinates, 
#     dfire_dist_to_bins,
#     dfire_energy,
#     interface_cutoff=3.9
# ):
#     dist_matrix = scipy.spatial.distance.cdist(receptor_coordinates, ligand_coordinates)
#     atom_indexes = np.where(dist_matrix <= 15.)
#     dist_matrix *= 2.0
#     dist_matrix -= 1.0
#     energy = 0.0
#     interface_receptor = []
#     interface_ligand = []
    
#     for i,j in zip(atom_indexes[0], atom_indexes[1]):
#         rec_atom = receptor.objects[i]
#         lig_atom = ligand.objects[j]
#         rnuma = rec_atom.dfire_residue_index
#         anuma = rec_atom.atom_index
#         rnumb = lig_atom.dfire_residue_index
#         anumb = lig_atom.atom_index
#         # convert numpy.float64 to int
#         d = dist_matrix[i][j]
#         if d <= interface_cutoff:
#             interface_receptor.append(i)
#             interface_ligand.append(j)
#         dfire_bin = dfire_dist_to_bins[int(d)]-1
#         energy += dfire_energy[rnuma][anuma][rnumb][anumb][dfire_bin]
    
#     # Convert and change energy sign
#     return (energy * 0.0157 - 4.7) * -1. # , set(interface_receptor), set(interface_ligand)
