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
from dockbo.utils.lightdock_torch_utils.scoring.dfire2_driver import DFIRE2_ATOM_TYPES 
# from dockbo.utils.lightdock_torch_utils.scoring.dfire_driver import DFIRE_ATOM_TYPES 

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

def update_adapter(
    adapter,
    receptor,
    receptor_restraints,
    ligand,
    ligand_restraints,
    new_receptor=False,
    new_ligand=False 
):
    # adapter = DefinedModelAdapter(receptor, ligand, None, None) 
    new_adapter = copy.deepcopy(adapter)
    # set new antibody ligand 
    if new_ligand:
        new_adapter.set_ligand_model(ligand, ligand_restraints)
    if new_receptor:
        new_adapter.set_receptor_model(receptor, receptor_restraints)
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

def prep_receptor(setup, move_to_origin=False):
    receptor = read_input_structure(setup['receptor_pdb'], setup['noxt'], setup['noh'], setup['now'], setup['verbose_parser'])
    if move_to_origin:
        rec_translation = receptor.move_to_origin() 
        rec_translation = torch.tensor(rec_translation).float() # (3,) gives translation of receptor 
    else:
        rec_translation = None
    save_lightdock_structure(receptor)
    receptor.n_modes = calculate_anm(receptor, setup['anm_rec'], DEFAULT_ANM_RMSD, STARTING_NM_SEED) # , DEFAULT_REC_NM_FILE)
    
    return receptor, rec_translation

def prep_ligand(setup, move_to_origin=False):
    ligand = read_input_structure(setup['ligand_pdb'], setup['noxt'], setup['noh'], setup['now'], setup['verbose_parser'])
    if move_to_origin:
        lig_translation = ligand.move_to_origin()
        lig_translation = torch.tensor(lig_translation).float()
    else:
        lig_translation = None 
    save_lightdock_structure(ligand)
    ligand.n_modes = calculate_anm(ligand, setup['anm_lig'], DEFAULT_ANM_RMSD, STARTING_NM_SEED) # , DEFAULT_LIG_NM_FILE)
    return ligand, lig_translation 

EPS = 1e-10 
def calculate_dfire2_torch(
    res_index, # int[:]
    atom_index, # int[:]
    coordinates, # np.ndarray[np.float64_t, ndim=2] 
    potentials, # np.ndarray[np.float64_t, ndim=3] 
    n, # np.uint32_t 
    interface_cutoff, # np.float64_t
    ab_atom_to_residue_dict=None,
    cdr3_only_version=False,
    is_receptor="antibody",
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

    if cdr3_only_version:
        cdr = 3
        temp_atom_index_is = [] # rec 
        temp_atom_index_js = [] # lig 
        temp_good_is = []
        temp_good_js = []
        if is_receptor == "antibody":
            ab_idxs = atom_index_is
        elif is_receptor == "antigen":
            ab_idxs = atom_index_js
        else:
            assert 0 
        for i, atom_idx in enumerate(ab_idxs):
            if ab_atom_to_residue_dict[atom_idx.item()] in range(CHOTHIA_CDR_DEFINITIONS["HCHAIN"][cdr][0], CHOTHIA_CDR_DEFINITIONS["HCHAIN"][cdr][1]+1):
                temp_atom_index_is.append(atom_index_is[i])
                temp_atom_index_js.append(atom_index_js[i]) 
                temp_good_is.append(good_is[i])
                temp_good_js.append(good_js[i])
        atom_index_is = torch.tensor(temp_atom_index_is)
        atom_index_js = torch.tensor(temp_atom_index_js)  
        good_is = torch.tensor(temp_good_is).long() 
        good_js = torch.tensor(temp_good_js).long()  
        good_dists = dist_matrix[good_is, good_js] 
    
    if len(good_dists) > 0:
        energy = potentials[atom_index_is,atom_index_js,good_dists].sum()/100.0
    else:
        energy = torch.tensor(0.0).float().cuda() 

    return energy, set(interface_receptor), set(interface_ligand)

def compute_ligand_receptor_poses(
    ld_sol,
    setup,
    receptor,
    ligand,
    receptor_coords,
    ligand_coords,
):
    if ld_sol == "default": # default pose 
        return receptor_coords, ligand_coords 
    try:
        ld_sol = ld_sol.squeeze() 
    except:
        pass # ld_sol is a list or other type, cna't squeeze 
    ## Unpack solution into matrices 
    # First 3 elements are a translation vector for the ligand
    translation = torch.tensor(ld_sol[:3], device=TH_DEVICE, dtype=TH_DTYPE)
    # Next 4 elements define a quaternion for rotating the ligand
    rotation = torch.tensor(ld_sol[3:3 + 4], device=TH_DEVICE, dtype=TH_DTYPE)
    # normalize rotation 
    norm = torch.linalg.vector_norm(rotation) 
    rotation = rotation / max(norm, EPS) 
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
    is_receptor,
    get_cdr_stats,
    cdr3_only_version=False,
):  
    if ld_sol == 'default' or (ld_sol is None):
        receptor_pose, ligand_pose = receptor_coords.unsqueeze(0), ligand_coords.unsqueeze(0)
    else:
        if type(ld_sol) != list:
            ld_sol = ld_sol.squeeze().tolist() 
        receptor_pose, ligand_pose = compute_ligand_receptor_poses(
            ld_sol,
            setup,
            receptor,
            ligand,
            receptor_coords,
            ligand_coords,
        )
    
    ab_atom_to_residue_dict = None 
    if cdr3_only_version:
        if is_receptor == 'antibody': antibody_molecule = receptor 
        elif is_receptor == 'antigen': antibody_molecule = ligand 
        ab_atom_to_residue_dict = get_atom_to_residue_dict(antibody_molecule, scoring_func="dfire2")

    potentials_energy = torch.tensor(potentials.energy, device=TH_DEVICE, dtype=TH_DTYPE).reshape(167, 167, 30)
    all_coords = torch.cat((receptor_pose, ligand_pose), dim=-2)
    molecule_length = len(res_index)
    energy, interface_receptorX, interface_ligandX = calculate_dfire2_torch(
        res_index, 
        atom_index, 
        all_coords.squeeze(-3), 
        potentials_energy, 
        molecule_length, 
        DEFAULT_CONTACT_RESTRAINTS_CUTOFF,
        ab_atom_to_residue_dict=ab_atom_to_residue_dict,
        cdr3_only_version=cdr3_only_version,
        is_receptor=is_receptor,
    )

    # Fix (prev version of restraints did not work! )
    dist_matrix2 = torch.cdist(receptor_pose.squeeze(), ligand_pose.squeeze())
    interface_receptor2, interface_ligand2 = get_interface_atoms(dist_matrix2, interface_cutoff=15)  # DEFAULT_CONTACT_RESTRAINTS_CUTOFF)

    return_dict = {} 
    if get_cdr_stats:
        if is_receptor == 'antibody':
            return_dict = cdrs_in_interface_stats(antibody_interface_atoms=list(interface_receptor2), antibody_molecule=receptor, scoring_f="dfire2" )
        elif is_receptor == 'antigen':
            return_dict = cdrs_in_interface_stats(antibody_interface_atoms=list(interface_ligand2), antibody_molecule=ligand, scoring_f="dfire2" )
    return_dict['energy'] = energy 

    return return_dict

    if False:
        # restraints factors alwats return 0 when there are no addional constraints 
        perc_lig_restraints = ScoringFunction.restraints_satisfied(ligand_restraints, set(interface_ligand2))
        # check = ScoringFunction.restraints_satisfied(ligand_restraints, set([i for i in range(3_000)]))
        perc_rec_restraints = ScoringFunction.restraints_satisfied(receptor_restraints, set(interface_receptor2))

    # return energy, perc_rec_restraints, perc_lig_restraints
    return energy, 0.0, 0.0 

CHOTHIA_CDR_DEFINITIONS = {
    "LCHAIN": {
        1: (25, 33),
        2: (49, 53),
        3: (90, 97),
    },
    "HCHAIN": {
        1: (25, 33),
        2: (50, 58),
        3: (92, 104),
    },
}


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
    is_receptor,
    get_cdr_stats,
    cdr3_only_version=False,
    cdr3_extra_weight=1.0,
):
    if ld_sol == 'default':
        receptor_pose, ligand_pose = receptor_coords, ligand_coords
    else:
        receptor_pose, ligand_pose = compute_ligand_receptor_poses(
            ld_sol,
            setup,
            receptor,
            ligand,
            receptor_coords,
            ligand_coords,
        )
    # score = dfire_oracle(
    #     lig_numbs=lig_numbs,
    #     rec_numas=rec_numas,
    #     receptor_coordinates=receptor_pose,
    #     ligand_coordinates=ligand_pose,
    #     receptor_restraints=adapter.receptor_model.restraints,
    #     ligand_restraints=adapter.ligand_model.restraints,
    # )
    ab_atom_to_residue_dict = None 
    if cdr3_only_version:
        if is_receptor == 'antibody': antibody_molecule = receptor 
        elif is_receptor == 'antigen': antibody_molecule = ligand 
        ab_atom_to_residue_dict = get_atom_to_residue_dict(antibody_molecule, scoring_func="dfire")
    
        cdr3_energy, _, _ = calculate_dfire_torch(
            rec_numas=rec_numas,
            receptor_coordinates=receptor_pose,
            lig_numbs=lig_numbs,
            ligand_coordinates=ligand_pose, 
            dfire_dist_to_bins=dfire_oracle.potential.dfire_dist_to_bins,
            dfire_energy=dfire_oracle.potential.dfire_energy,
            ab_atom_to_residue_dict=ab_atom_to_residue_dict,
            cdr3_only=True,
            is_receptor=is_receptor,
        )
    else:
        cdr3_energy = 0.0 

    energy, interface_receptor, interface_ligand = calculate_dfire_torch(
        rec_numas=rec_numas,
        receptor_coordinates=receptor_pose,
        lig_numbs=lig_numbs,
        ligand_coordinates=ligand_pose, 
        dfire_dist_to_bins=dfire_oracle.potential.dfire_dist_to_bins,
        dfire_energy=dfire_oracle.potential.dfire_energy,
        ab_atom_to_residue_dict=ab_atom_to_residue_dict,
        cdr3_only=False,
        is_receptor=is_receptor,
    )
    
    # total energy + energy w/ only cdr3 pairs 
    energy = energy + cdr3_extra_weight * cdr3_energy # v1 == cdr3_energy only 
    return_dict = {} 
    if get_cdr_stats:
        if is_receptor == 'antibody':
            antibody_interface_atoms = interface_receptor
            antibody_molecule=receptor
        elif is_receptor == 'antigen':
            antibody_interface_atoms=list(interface_ligand)
            antibody_molecule=ligand
        try:
            antibody_interface_atoms = list(antibody_interface_atoms)
        except: # special casse when we only have a single interface atom 
            antibody_interface_atoms = [interface_receptor]
        return_dict = cdrs_in_interface_stats(antibody_interface_atoms=antibody_interface_atoms, antibody_molecule=antibody_molecule, scoring_f="dfire" )

    return_dict['energy'] = energy 
    return return_dict 

    # cdrs_in_interface_stats(antibody_interface_atoms, antibody_molecule, scoring_f )
    # XXX perc_receptor_restraints, perc_ligand_restraints = 0.0, 0.0
    if False:
        perc_receptor_restraints = ScoringFunction.restraints_satisfied(
            adapter.receptor_model.restraints, interface_receptor
        )
        perc_ligand_restraints = ScoringFunction.restraints_satisfied(
            adapter.ligand_model.restraints, interface_ligand
        )
    # return energy, perc_receptor_restraints, perc_ligand_restraints
    # return energy 
    # energy + perc_receptor_restraints * energy + perc_ligand_restraints * energy

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
    # ALWAYS THE SAME! :) 

    # return score 


def cdrs_in_interface_stats(antibody_interface_atoms, antibody_molecule, scoring_f ): #interface_receptor, receptor, "dfire"
    # convert interface atoms to interface residues
    atom_to_residue_idx = get_atom_to_residue_dict(antibody_molecule, scoring_f)
    interface_residues = set([atom_to_residue_idx[atom] for atom in antibody_interface_atoms])
    # count num interface residues in each cdr 
    return_dict = {} 
    cdr_counts = {}
    for cdr in range(1,4): 
        cdr_counts[cdr] = 0  
        return_dict[f'N ab interface residues in cdr {cdr}'] = 0 
    for idx in interface_residues:
        for cdr in range(1,4): 
            if idx in range(CHOTHIA_CDR_DEFINITIONS["HCHAIN"][cdr][0], CHOTHIA_CDR_DEFINITIONS["HCHAIN"][cdr][1]+1):
                cdr_counts[cdr] += 1 
                return_dict[f'N ab interface residues in cdr {cdr}']  += 1 
    total_counts = len(interface_residues)
    return_dict[f'Total N ab interface residues'] = total_counts 
    for cdr in range(1,4): 
        return_dict[f'Perc ab interface residues in cdr {cdr}'] = cdr_counts[cdr]/max(total_counts, 1e-10) 
    count_any_cdr = cdr_counts[1] + cdr_counts[2] + cdr_counts[3] 
    return_dict[f'Total N ab interface residues in any cdr'] = count_any_cdr
    return_dict[f'Perc ab interface residues in any cdr'] =  count_any_cdr/max(total_counts, 1e-10) 
    return_dict[f'N ab interface residues outside cdrs'] =  total_counts-count_any_cdr
    return_dict[f'Perc ab interface residues outside cdrs'] = (total_counts-count_any_cdr)/max(total_counts, 1e-10) 
    
    return return_dict 


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
    ab_atom_to_residue_dict=None,
    cdr3_only=False, 
    is_receptor="antibody"
):
    dist_matrix = torch.cdist(receptor_coordinates.squeeze(), ligand_coordinates.squeeze())
    atom_indexes = torch.where(dist_matrix <= 15.0) 
    rec_idxs = atom_indexes[0]
    lig_idxs = atom_indexes[1]  

    if cdr3_only:
        cdr = 3
        temp_rec_idxs = [] 
        temp_lig_idxs = [] 
        if is_receptor == "antibody":
            ab_idxs = rec_idxs 
        elif is_receptor == "antigen":
            ab_idxs = lig_idxs 
        else:
            assert 0 
        for i, atom_idx in enumerate(ab_idxs):
            ab_residue_idx = ab_atom_to_residue_dict[atom_idx.item()] 
            if ab_residue_idx in range(CHOTHIA_CDR_DEFINITIONS["HCHAIN"][cdr][0], CHOTHIA_CDR_DEFINITIONS["HCHAIN"][cdr][1]+1):
                temp_rec_idxs.append(rec_idxs[i])
                temp_lig_idxs.append(lig_idxs[i]) 
        rec_idxs = torch.tensor(temp_rec_idxs)
        lig_idxs = torch.tensor(temp_lig_idxs)  


    if len(rec_idxs) == 0:
        return 0.0, 0.0, 0.0  
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
    interface_receptor, interface_ligand = get_interface_atoms(dist_matrix,interface_cutoff=15) #  # 3.9

    return score, interface_receptor, interface_ligand


def get_interface_atoms(dist_matrix, interface_cutoff):
    interface_atom_indexes = torch.where(dist_matrix <= interface_cutoff)
    interface_receptor = interface_atom_indexes[0].tolist()
    interface_ligand = interface_atom_indexes[1].tolist()

    return set(interface_receptor), set(interface_ligand)


def get_atom_to_residue_dict(molecule, scoring_func):
    assert scoring_func in ["dfire", "dfire2"]
    if scoring_func == "dfire":
        atom_to_residue_idx = get_atom_to_residue_dict_dfire(molecule)
    elif scoring_func == "dfire2":
        atom_to_residue_idx = get_atom_to_residue_dict_dfire2(molecule)
    return atom_to_residue_idx


def get_atom_to_residue_dict_dfire2(molecule):
    atom_to_residue_idx = {}
    atom_index = 0
    for residue in molecule.residues:
        for rec_atom in residue.atoms:
            rec_atom_type = rec_atom.residue_name + " " + rec_atom.name
            if rec_atom_type in DFIRE2_ATOM_TYPES:
                atom_to_residue_idx[atom_index] = residue.number 
                atom_index += 1
    return atom_to_residue_idx


def get_atom_to_residue_dict_dfire(molecule):
    atom_to_residue_idx = {}
    atom_index = 0
    for chain in molecule.chains:
        for residue in chain.residues:
            for rec_atom in residue.atoms:
                # rec_atom_type = rec_atom.residue_name + rec_atom.name
                atom_to_residue_idx[atom_index] = residue.number 
                atom_index += 1
    return atom_to_residue_idx


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
