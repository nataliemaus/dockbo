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
    STARTING_NM_SEED
)
from dockbo.utils.lightdock_torch_utils.setup_sim import read_input_structure
from dockbo.utils.lightdock_torch_utils.structure.nm import calculate_nmodes
from dockbo.utils.lightdock_torch_utils.scoring.dfire2_driver import DefinedModelAdapter 
from dockbo.utils.lightdock_torch_utils.boundaries import get_default_box
from dockbo.utils.lightdock_torch_utils.lightdock_errors import LightDockError
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


def get_ligand_indicies(adapter):
    res_index = []
    atom_index = []
    for o in adapter.ligand_model.objects:
        res_index.append(o.residue_index)  # + last)
        atom_index.append(o.atom_index)
    return res_index, atom_index


def get_indicies(adapter, ligand_indicies):
    res_index, atom_index, last  = get_receptor_indicies(adapter) 
    if ligand_indicies is None:
        ligand_indicies = get_ligand_indicies(adapter)
    ligand_res_indicies, ligand_atom_indicies = ligand_indicies
    ligand_res_indicies = [res_idx + last for res_idx in ligand_res_indicies] 
    res_index = res_index + ligand_res_indicies
    atom_index = atom_index + ligand_atom_indicies
    res_index = torch.tensor(res_index, dtype=torch.long, device=TH_DEVICE)
    atom_index = torch.tensor(atom_index, dtype=torch.long, device=TH_DEVICE)
    
    return res_index, atom_index 


def init_adapter(ligand ):
    adapter = DefinedModelAdapter(
        receptor=None, 
        ligand=ligand,
    ) 
    return adapter 


def update_adapter(adapter, receptor, ligand, new_ligand=False ):
    # adapter = DefinedModelAdapter(receptor, ligand, None, None) 
    new_adapter = copy.deepcopy(adapter)
    # set new antibody receptor 
    new_adapter.set_receptor_model(receptor, None)
    if new_ligand:
        new_adapter.set_ligand_model(ligand, None)
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
    receptor.move_to_origin()
    save_lightdock_structure(receptor)
    receptor.n_modes = calculate_anm(receptor, setup['anm_rec'], DEFAULT_ANM_RMSD, STARTING_NM_SEED) # , DEFAULT_REC_NM_FILE)
    return receptor 


def prep_ligand(setup):
    ligand = read_input_structure(setup['ligand_pdb'], setup['noxt'], setup['noh'], setup['now'], setup['verbose_parser'])
    ligand.move_to_origin()
    save_lightdock_structure(ligand)
    ligand.n_modes = calculate_anm(ligand, setup['anm_lig'], DEFAULT_ANM_RMSD, STARTING_NM_SEED) # , DEFAULT_LIG_NM_FILE)
    return ligand 


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
    potentials_energy = torch.tensor(potentials.energy, device=TH_DEVICE, dtype=TH_DTYPE).reshape(167, 167, 30)
    # Rotate and translate ligand
    ligand_pose = transforms.quaternion_apply(rotation, ligand_pose) + translation
    all_coords = torch.cat((receptor_pose, ligand_pose), dim=-2)
    molecule_length = len(res_index)
    energy, _, interface_ligand = calculate_dfire2_torch(res_index, atom_index, all_coords.squeeze(-3), potentials_energy, molecule_length, DEFAULT_CONTACT_RESTRAINTS_CUTOFF)
    perc_lig_restraints = ScoringFunction.restraints_satisfied(ligand_restraints, set(interface_ligand))
    perc_rec_restraints = ScoringFunction.restraints_satisfied(receptor_restraints, set(receptor_coords))

    return energy + perc_lig_restraints * energy + perc_rec_restraints * energy
