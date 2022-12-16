"""Implementation of the pyDock scoring function.

C-implementation of the pyDock scoring function and using the freesasa library:
https://github.com/mittinatten/freesasa
"""
import numpy as np
import freesasa
from freesasa import Structure
import sys 
sys.path.append("/home/nmaus/dockbo/")
from dockbo.utils.lightdock_torch_utils.structure.model import DockingModel
from dockbo.utils.lightdock_torch_utils.scoring.functions import ModelAdapter, ScoringFunction
from dockbo.utils.lightdock_torch_utils.scoring.cpydock.amber import (
    translate,
    amber_types,
    amber_charges,
    amber_masses,
) 
from dockbo.utils.lightdock_torch_utils.scoring.cpydock.vdw import (
    vdw_vdw_energy,
    vdw_vdw_radii,
)
from dockbo.utils.lightdock_torch_utils.scoring.cpydock.solvation import get_solvation
from dockbo.utils.lightdock_torch_utils.lightdock_errors import NotSupportedInScoringError
freesasa.setVerbosity(freesasa.silent)
from dockbo.utils.lightdock_torch_utils.lightdock_constants import DEFAULT_CONTACT_RESTRAINTS_CUTOFF

"""PyDock scoring function parameters"""
default_hydrogen_extension = ".H"
# default_amber_extension = ".amber" 
# Energetic terms
scoring_vdw_weight = 0.1
vdw_input_file = "vdw.in"
default_max_electrostatics_cutoff = 1
default_min_electrostatics_cutoff = -1
default_vdw_cutoff = 1.0
# AMBER
# amber_elec_constant = 18.2223
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PyInt_AsUnsignedLongMask PyLong_AsUnsignedLongMask
#include <Python.h>
#include "structmember.h"
#include "numpy/arrayobject.h"
import torch 
EPSILON = 4.0
FACTOR = 332.0
MAX_ES_CUTOFF = 1.0
MIN_ES_CUTOFF = -1.0
VDW_CUTOFF = 1.0
HUGE_DISTANCE = 10000.0
ELEC_DIST_CUTOFF = 30.0
ELEC_DIST_CUTOFF2 = ELEC_DIST_CUTOFF*ELEC_DIST_CUTOFF
VDW_DIST_CUTOFF = 10.0
VDW_DIST_CUTOFF2 = VDW_DIST_CUTOFF*VDW_DIST_CUTOFF
SOLVATION_DISTANCE = 6.4 
SOLVATION_DISTANCE2 = 6.4*6.4
SCORING_VDW_WEIGHT = 0.1

class CPyDockModel(DockingModel):
    """Prepares the structure necessary for the C-implementation of the pyDock scoring function"""

    def __init__(
        self,
        objects,
        coordinates,
        restraints,
        charges,
        vdw_energy,
        vdw_radii,
        des_energy,
        des_radii,
        sasa,
        hydrogens,
        reference_points=None,
        n_modes=None,
    ):
        super(CPyDockModel, self).__init__(
            objects, coordinates, restraints, reference_points
        )
        self.charges = charges
        self.vdw_energy = vdw_energy
        self.vdw_radii = vdw_radii
        self.des_energy = des_energy
        self.des_radii = des_radii
        self.sasa = sasa 
        self.hydrogens = hydrogens
        self.n_modes = n_modes

    def clone(self):
        """Creates a copy of the current model"""
        return CPyDockModel(
            self.objects,
            self.coordinates.copy(),
            self.restraints,
            self.charges,
            self.vdw_energy,
            self.vdw_radii,
            self.des_energy,
            self.des_radii,
            self.sasa,
            self.hydrogens,
            reference_points=self.reference_points.copy(),
        )


class CPyDockAdapter(ModelAdapter):
    """Adapts a given Complex to a DockingModel object suitable for this
    PyDock scoring function.
    """

    def _get_docking_model(self, molecule, restraints):
        atoms = molecule.atoms
        parsed_restraints = {}
        # Assign properties to atoms
        for atom_index, atom in enumerate(atoms):
            res_id = f"{atom.chain_id}.{atom.residue_name}.{atom.residue_number}{atom.residue_insertion}"
            if restraints and res_id in restraints:
                try:
                    parsed_restraints[res_id].append(atom_index)
                except:
                    parsed_restraints[res_id] = [atom_index]
            try:
                res_name = atom.residue_name
                atom_name = atom.name
                if res_name == "HIS":
                    res_name = "HID"
                if atom_name in translate:
                    atom_name = translate[atom.name]
                atom_id = "%s-%s" % (res_name, atom_name)
                atom.amber_type = amber_types[atom_id]
                atom.charge = amber_charges[atom_id] 
                atom.mass = amber_masses[atom.amber_type]
                atom.vdw_energy = vdw_vdw_energy[atom.amber_type]
                atom.vdw_radius = vdw_vdw_radii[atom.amber_type]
            except KeyError:
                raise NotSupportedInScoringError(
                    "Residue {} or atom {} not supported. ".format(res_id, atom_name)
                    + "PyDock only supports AMBER94 types."
                )

        # Prepare common model information
        elec_charges = np.array([atom.charge for atom in atoms])
        vdw_energies = np.array([atom.vdw_energy for atom in atoms])
        vdw_radii = np.array([atom.vdw_radius for atom in atoms])
        coordinates = molecule.copy_coordinates()
        des_energy, des_radii = get_solvation(molecule)

        # Calculate desolvation reference energy
        structure = Structure()
        des_radii_no_H = []
        for i, atom in enumerate(atoms):
            if not atom.is_hydrogen():
                structure.addAtom(
                    atom.name,
                    atom.residue_name,
                    atom.residue_number,
                    atom.chain_id,
                    atom.x,
                    atom.y,
                    atom.z,
                )
                des_radii_no_H.append(des_radii[i])
        structure.setRadii(list(des_radii_no_H))
        sasa_result = freesasa.calc(structure)
        sasa = []
        j = 0
        for i, atom in enumerate(atoms):
            if not atom.is_hydrogen():
                sasa.append(sasa_result.atomArea(j))
                j += 1
            else:
                sasa.append(-1.0)
        sasa = np.array(sasa)
        hydrogens = np.array([0 if atom.is_hydrogen() else 1 for atom in atoms])

        reference_points = ModelAdapter.load_reference_points(molecule)
        try:
            return CPyDockModel(
                atoms,
                coordinates,
                parsed_restraints,
                elec_charges,
                vdw_energies,
                vdw_radii,
                des_energy,
                des_radii,
                sasa,
                hydrogens,
                reference_points=reference_points,
                n_modes=molecule.n_modes.copy(),
            )
        except AttributeError:
            return CPyDockModel(
                atoms,
                coordinates,
                parsed_restraints,
                elec_charges,
                vdw_energies,
                vdw_radii,
                des_energy,
                des_radii,
                sasa,
                hydrogens,
                reference_points=reference_points,
            )


class CPyDock(ScoringFunction):
    def __init__(self, weight=1.0):
        super(CPyDock, self).__init__(weight)
        try:
            with open(vdw_input_file) as vdw_file:
                self.scoring_vdw_weight = float(vdw_file.readline())
        except (IOError, ValueError) as e:
            print("Error (%s), using default VDW cutoff" % str(e)) 
            self.scoring_vdw_weight = scoring_vdw_weight

    def __call__(self, receptor, receptor_coordinates, ligand, ligand_coordinates):
        """Computes the pyDock scoring energy using receptor and ligand which are
        instances of DockingModel.
        """
        energy = cpydock_calculate_energy(
            receptor_coordinates,
            ligand_coordinates,
            all_rec_charges=receptor.charges,
            all_lig_charges=ligand.charges,
            all_rec_vdw_e=receptor.vdw_energy,
            all_lig_vdw_e=ligand.vdw_energy,
            all_rec_vdw_radii=receptor.vdw_radii,
            all_lig_vdw_radii=ligand.vdw_radii,
            # all_rec_h=receptor.hydrogens,
            # all_lig_h=ligand.hydrogens,
            all_rec_sasa=receptor.sasa,
            all_lig_sasa=ligand.sasa,
            all_rec_des_e=receptor.des_energy,
            all_lig_des_e=ligand.des_energy,
        ) 
        return energy 

# Needed to dynamically load the scoring functions from command line
# DefinedScoringFunction = CPyDock
# DefinedModelAdapter = CPyDockAdapter

def cpydock_calculate_energy(
    receptor_coordinates,
    ligand_coordinates,
    all_rec_charges,
    all_lig_charges,
    all_rec_vdw_e,
    all_lig_vdw_e,
    all_rec_vdw_radii,
    all_lig_vdw_radii,
    all_rec_sasa,
    all_lig_sasa,
    all_rec_des_e,
    all_lig_des_e,
):
    all_rec_charges = torch.from_numpy(all_rec_charges).float().cuda() 
    all_lig_charges = torch.from_numpy(all_lig_charges).float().cuda() 
    all_rec_des_e = torch.from_numpy(all_rec_des_e).float().cuda()  
    all_lig_des_e = torch.from_numpy(all_lig_des_e).float().cuda() 
    all_rec_vdw_radii = torch.from_numpy(all_rec_vdw_radii).float().cuda()  
    all_lig_vdw_radii = torch.from_numpy(all_lig_vdw_radii).float().cuda() 
    all_rec_sasa = torch.from_numpy(all_rec_sasa).float().cuda() 
    all_lig_sasa = torch.from_numpy(all_lig_sasa).float().cuda() 
    all_rec_vdw_e = torch.from_numpy(all_rec_vdw_e).float().cuda() 
    all_lig_vdw_e = torch.from_numpy(all_lig_vdw_e).float().cuda() 
    dist_matrix = torch.cdist(receptor_coordinates.squeeze(), ligand_coordinates.squeeze())
    atom_indexes = torch.where(dist_matrix <= 15.0) 
    rec_idxs = atom_indexes[0]
    lig_idxs = atom_indexes[1]  

    # Electrostatics energy
    atom_indexes = torch.where(dist_matrix <= ELEC_DIST_CUTOFF) 
    rec_idxs = atom_indexes[0]
    rec_charges = all_rec_charges[rec_idxs]
    lig_idxs = atom_indexes[1] 
    lig_charges = all_lig_charges[lig_idxs] 
    dists_sq = dist_matrix[atom_indexes]**2
    total_elec = rec_charges * lig_charges / dists_sq
    total_elec = torch.clamp(total_elec, min=MIN_ES_CUTOFF*EPSILON/FACTOR ,max=MAX_ES_CUTOFF*EPSILON/FACTOR)
    total_elec = total_elec.sum() 
    # // Convert total electrostatics to Kcal/mol:
    #     //      - coordinates are in Ang
    #     //      - charges are in e (elementary charge units)
    total_elec = total_elec * FACTOR / EPSILON 

    # Van der Waals energy
    atom_indexes = torch.where(dist_matrix <= VDW_DIST_CUTOFF) 
    rec_idxs = atom_indexes[0]
    rec_vdw = all_rec_vdw_e[rec_idxs]
    rec_vdw_radii = all_rec_vdw_radii[rec_idxs]
    lig_idxs = atom_indexes[1] 
    lig_vdw = all_lig_vdw_e[lig_idxs]
    lig_vdw_radii = all_lig_vdw_radii[lig_idxs]
    dists = dist_matrix[atom_indexes]
    vdw_energy = torch.sqrt(rec_vdw * lig_vdw)
    vdw_radius = rec_vdw_radii + lig_vdw_radii 
    p6 = (vdw_radius / dists)**6 
    total_vdw_e = vdw_energy * (p6*p6 - 2.0 * p6) 
    total_vdw_e = torch.clamp(total_vdw_e, max=VDW_CUTOFF)
    total_vdw_e = total_vdw_e.sum() 

    # Calculate contact solvation for receptor
    mask = (dist_matrix <= SOLVATION_DISTANCE) & (dist_matrix > 0.0) & (all_rec_sasa.unsqueeze(-1) > 0)
    atom_indexes = torch.where(mask) 
    dists = dist_matrix[atom_indexes]
    rec_idxs = atom_indexes[0]
    rec_des_energy = all_rec_des_e[rec_idxs]
    rec_asa = all_rec_sasa[rec_idxs] 
    solv_rec = -10.0 * torch.sqrt(dists) + 65.0 
    solv_rec = torch.clamp(solv_rec, max=rec_asa) 
    total_solvation_rec = solv_rec * rec_des_energy 
    total_solvation_rec = total_solvation_rec.sum() 

    # Calculate contact solvation for ligand
    mask = (dist_matrix <= SOLVATION_DISTANCE) & (dist_matrix > 0.0) & (all_lig_sasa.unsqueeze(0) > 0)
    atom_indexes = torch.where(mask)
    dists = dist_matrix[atom_indexes]
    lig_idxs = atom_indexes[1]
    lig_des_energy = all_lig_des_e[lig_idxs]
    lig_asa = all_lig_sasa[lig_idxs] 
    solv_lig = -10.0 * torch.sqrt(dists) + 65.0 
    solv_lig = torch.clamp(solv_lig, max=lig_asa) 
    total_solvation_lig = solv_lig * lig_des_energy 
    total_solvation_lig = total_solvation_lig.sum() 
    solv = -1 * (total_solvation_rec + total_solvation_lig)
    energy = (total_elec + SCORING_VDW_WEIGHT * total_vdw_e + solv) * -1.0

    return energy 
