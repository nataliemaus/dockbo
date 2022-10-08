import sys
sys.path.append('../')
from dockbo.utils.fold_utils.nanonet.nanonet import NanoNet 
from dockbo.utils.fold_utils.fold_utils import remove_hetero_atoms_and_hydrogens
import os
import shutil
import uuid
import torch
from scipy.spatial import KDTree
from dockbo.utils.lightdock_torch_utils.lightdock_constants import (
    DEFAULT_NMODES_REC,
    DEFAULT_NMODES_LIG,
)
# from dockbo.utils.lightdock_torch_utils.setup_sim import get_setup_from_file 
from dockbo.utils.lightdock_torch_utils.scoring.dfire2_driver import DFIRE2Potential
from dockbo.utils.lightdock_torch_utils.faster_torch_code import (
    dfire2_torch,
    prep_receptor,
    prep_ligand,
    update_adapter,
    get_indicies,
    get_receptor_indicies,
    get_coords,
    get_bbox_tensor,
    get_dfire_score,
    get_numbs,
    get_anumas,
)
from dockbo.utils.lightdock_torch_utils.feasibility_utils import is_valid_config
import time 
from dockbo.utils.lightdock_torch_utils.scoring.dfire_driver import DefinedScoringFunction as dfireOracle
from dockbo.utils.lightdock_torch_utils.scoring.dfire2_driver import DefinedModelAdapter as Dfire2Adapter
from dockbo.utils.lightdock_torch_utils.scoring.dfire_driver import DefinedModelAdapter as DfireAdapter
from dockbo.utils.bo_utils.run_turbo import run_turbo

class DockBO():

    def __init__(
        self,
        path_to_default_antigen_pdb=None,
        fold_software='nanonet',
        work_dir='/home/nmaus/',
        pdb_dir='dockbo/dockbo/utils/fold_utils/temp_pdbs',
        restraint_file="",
        include_lightchain_in_folded_structure=False,
        wt_seq_light_chain=None,
        max_n_bo_steps=100,
        bsz=1,
        n_init=10,
        n_epochs=20,
        learning_rte=0.01,
        verbose_config_opt=False,
        use_anm=True, 
        anm_rec=DEFAULT_NMODES_REC,
        anm_lig=DEFAULT_NMODES_LIG,
        verbose_timing=False,
        scoring_func='dfire',
        is_receptor='antibody',
    ):
        self.is_receptor = is_receptor
        self.scoring_func = scoring_func 
        assert scoring_func in ['dfire', 'dfire2']
        self.verbose_timing = verbose_timing 
        start_init = time.time() 
        # temporary directory for lightdock files 
        self.pdb_dir = work_dir + pdb_dir
        # make sure pdb dir is new empty dir
        if not os.path.exists(self.pdb_dir):
            os.mkdir(self.pdb_dir)

        # LightDock Args:
        self.setup = {
            'use_anm':use_anm, 'anm_rec':anm_rec,'anm_lig':anm_lig,
            'noxt':True, 'noh':True, 'now':True,'verbose_parser':False,
        }
        self.bounding_box = get_bbox_tensor(self.setup) 
        if path_to_default_antigen_pdb is not None:
            # copy antibody to simulation directoy
            os.system(f"cp {path_to_default_antigen_pdb} {self.pdb_dir}")
            self.default_antigen_path = self.pdb_dir + '/' + path_to_default_antigen_pdb.split('/')[-1]
            # prep antigen as receptor! 
            if self.is_receptor == 'antigen':
                self.setup['receptor_pdb'] = self.default_antigen_path 
                self.default_receptor, _ = prep_receptor(self.setup)
            elif self.is_receptor == 'antibody':
                self.setup['ligand_pdb'] = self.default_antigen_path 
                self.default_ligand, _ = prep_ligand(self.setup)
        else:
            self.default_antigen_path = None 
            self.default_receptor = None
            self.default_ligand = None

        if self.scoring_func == 'dfire':
            AdapterClass = DfireAdapter
            self.dfire_oracle = dfireOracle(work_dir=work_dir) 
        elif self.scoring_func == 'dfire2':
            AdapterClass = Dfire2Adapter
            self.potentials = DFIRE2Potential(work_dir=work_dir)

        self.default_adapter = AdapterClass(
            receptor=self.default_receptor, 
            ligand=self.default_ligand,
        )
        if (self.scoring_func == 'dfire'):
            if self.default_receptor is not None:
                self.default_rec_numas = get_anumas(self.default_adapter.receptor_model)
            else:
                self.default_rec_numas = None
            if self.default_ligand is not None:
                self.default_lig_numas = get_numbs(self.default_adapter.ligand_model)
            else:
                self.default_lig_numas = None

        self.default_receptor_indicies = None 
        if self.default_receptor is not None:
            self.default_receptor_indicies = get_receptor_indicies(self.default_adapter )

        self.default_ligand_indicies = None 

        # TuRBO ARGS: 
        self.max_n_bo_steps = max_n_bo_steps 
        self.n_init = n_init
        self.bsz = bsz
        self.n_epochs = n_epochs
        self.learning_rte = learning_rte
        self.best_config = None 
        self.verbose_config_opt = verbose_config_opt

        # FOLD ARGS:
        self.include_lightchain_in_folded_structure = include_lightchain_in_folded_structure
        self.wt_seq_light_chain = wt_seq_light_chain # constant light chain passed into igfold 
        # include default wt lightchain with predicted heavy chain to 
        #   create folded antibody structure (can only be included w/ igfold)
        if self.include_lightchain_in_folded_structure:
            assert fold_software == 'igfold'
            assert self.wt_seq_light_chain is not None 
        self.fold_software = fold_software
        assert self.fold_software in ['igfold', 'nanonet']
        self.restraint_file = restraint_file

        if self.fold_software == 'igfold':
            from igfold import IgFoldRunner, init_pyrosetta 
            init_pyrosetta()
            self.igfold = IgFoldRunner()
        else:
            assert self.fold_software == 'nanonet'
            # create nanonet object to create structures from strings
            self.nanonet = NanoNet(
                out_dir=self.pdb_dir,
                nanonet_path=work_dir + 'NanoNet/NanoNet.py',
            )
        if verbose_timing:
            print(f"Time to complete oracle init: {time.time() - start_init}")

    def __call__(
        self,
        path_to_antigen_pdb=None,
        antibody_aa_seq=None, 
        path_to_antibody_pdb=None, 
        config_x=None,
    ):
        ''' Inputs: 
                path_to_antigen_pdb: path to anitgen pdb file, if None will use default antigen from init
                antibody_aa_seq: string sequence of anino acids whcih specify antibody, will be converted to pdb using self.fold_software
                path_to_antibody_pdb: path to pdb file for antibody, used instead of amino acid seq if given
                config_x: iterable of 27 numbers giving configuration of ligand-receptor pair. If None, config is optimzed w/ TuRBO
            Output: 
                lightdock score 
        '''
        start_call = time.time() 

        # clamp config_x to be within bounding box 
        if config_x is not None: 
            if not torch.is_tensor(config_x):
                config_x = torch.tensor(config_x).float() 
            if len(config_x.shape) > 1:
                config_x = config_x.squeeze()
            assert len(config_x) == 27 
            config_x = torch.clamp(config_x, self.bounding_box[:,0], self.bounding_box[:,1]) 
        
        # grab original work dir
        self.og_dir = os.getcwd()

        # If no antibody pdb provided, fold aa seq 
        if path_to_antibody_pdb is None: 
            start_fold = time.time() 
            assert antibody_aa_seq is not None # must provide aa seq to fold 
            # fold protein and create pdb files with igfold or nanonet 
            path_to_antibody_pdb, temp_pdb_dir = self.fold_protein(antibody_aa_seq)
            if self.verbose_timing:
                print(f"time to fold w/ {self.fold_software}: {time.time() - start_fold}")
        # otherwise, copy provided anitbody pdb to sim directory
        else: 
            temp_pdb_dir = self.pdb_dir + '/' + str(uuid.uuid1())
            os.mkdir(temp_pdb_dir) 
            # copy antibody to working directoy
            os.system(f"cp {path_to_antibody_pdb} {temp_pdb_dir}")
            path_to_antibody_pdb = temp_pdb_dir + '/' + path_to_antibody_pdb.split('/')[-1]

        # create new dir for additional auxilary simulation files
        new_ab_directory = f"{temp_pdb_dir}/{uuid.uuid1()}"
        os.makedirs(new_ab_directory)

        # run lightdock simulation to get binding score
        top_score = self.run_docking(
            new_ab_directory,
            path_to_antibody_pdb,
            path_to_antigen_pdb,
            config_x=config_x,
        )
        # remove temp dir
        shutil.rmtree(temp_pdb_dir)
        # reset to initial dir 
        os.chdir( self.og_dir )

        if self.verbose_timing:
            print(f"time for full oracle call: {time.time() - start_call}")

        return top_score 

    def fold_protein(self, protein_seq):
        if self.fold_software == 'nanonet':
            # create pdb files with nanonet
            paths_to_pdbs, temp_pdb_dir = self.nanonet.fold([protein_seq], store=True)
            return paths_to_pdbs[0], temp_pdb_dir
        elif self.fold_software == 'igfold':
            temp_out_dir = self.pdb_dir + '/' + str(uuid.uuid1())
            os.mkdir(temp_out_dir) 
            sequences = {"H":protein_seq }
            if self.include_lightchain_in_folded_structure:
                sequences["L"] = self.wt_seq_light_chain # stays constant
            pdb_path = temp_out_dir + '/' + str(uuid.uuid4()) + ".pdb"
            self.igfold.fold(
                pdb_path, # Output PDB file 
                sequences=sequences, # Antibody sequences
                do_refine=True, # Refine the antibody structure with PyRosetta
                do_renum=True, # Renumber predicted antibody structure (Chothia)
            )
            pdb_path = remove_hetero_atoms_and_hydrogens(pdb_path)

        return pdb_path, temp_out_dir

    def prep_lightdock(
        self,
        directory,
        antibody_path,
        path_to_antigen_pdb=None,
    ):
        # move to sim directory 
        os.chdir(directory)

        # if we have a path to a new antigen pdb, read it in and prep antigen
        #   otherwise, we assume defaul receptor antigan  
        new_receptor, new_ligand = False, False 
        if path_to_antigen_pdb is not None:
            # copy antibody to simulation directoy
            os.system(f"cp {path_to_antigen_pdb} {directory}")
            antigen_path = directory + '/' + path_to_antigen_pdb.split('/')[-1]
            # prep antigen ligand
            if self.is_receptor == 'antigen':
                self.setup['receptor_pdb'] = antigen_path
                self.receptor, _ = prep_receptor(self.setup)
                new_receptor = True
            elif self.is_receptor == 'antibody':
                self.setup['ligand_pdb'] = antigen_path
                self.ligand, _ = prep_ligand(self.setup)
                new_ligand = True
        else:
            if self.is_receptor == 'antigen':
                self.setup['receptor_pdb'] = self.default_antigen_path 
                self.receptor = self.default_receptor  
            elif self.is_receptor == 'antibody':
                self.setup['ligand_pdb'] = self.default_antigen_path 
                self.ligand = self.default_ligand 
        
        # prep antibody ligand 
        if self.is_receptor == 'antigen':
            self.setup['ligand_pdb'] = antibody_path
            self.ligand, self.lig_translation = prep_ligand(self.setup) 
        elif self.is_receptor == 'antibody':
            self.setup['receptor_pdb'] = antibody_path
            self.receptor, self.rec_translation = prep_receptor(self.setup) 

        # update adapter 
        self.adapter = update_adapter(
            self.default_adapter,
            self.receptor,
            self.ligand,
            new_receptor=new_receptor,
            new_ligand=new_ligand,
        )
        # get combined indicies for receptor and ligand 
        self.res_index, self.atom_index = get_indicies(self.adapter, self.default_receptor_indicies)
        # get coords for receptor and ligand 
        self.receptor_coords, self.ligand_coords = get_coords(self.adapter)

        if (self.scoring_func == 'dfire'):
            if new_receptor:
                self.rec_numas = get_anumas(self.adapter.receptor_model)
            else:
                self.rec_numas = self.default_rec_numas
            if new_ligand:
                self.lig_numbs = get_numbs(self.adapter.ligand_model)
            else:
                self.lig_numbs = self.default_lig_numas

    def run_docking(
        self,
        directory,
        antibody_path,
        path_to_antigen_pdb=None,
        config_x=None,
    ):
        # prep lightdock for new receptor, ligand pair 
        start = time.time() 
        self.prep_lightdock(
            directory,
            antibody_path,
            path_to_antigen_pdb=path_to_antigen_pdb,
        )
        if self.verbose_timing:
            print(f'prep ligthdock time: {time.time() - start}')
        # if not config specified, use TuRBO to find optimal config (config that maximizes score)
        start = time.time() 
        check_validity_utils = self.get_check_validity_utils()
        if config_x is None: 
            best_config, best_score = self.optimize_configuration(check_validity_utils) 
            self.best_config = best_config
        # otherwise, use given config_x (27,) to get score
        else:
            # make sure given config_x is valid (x,y,z loc falls outside of receptor)
            assert is_valid_config(config_x, check_validity_utils)
            best_score = self.get_lightdock_score(config_x)
            self.best_config = config_x.unsqueeze(0)
        if self.verbose_timing:
            print(f'compute score time: {time.time() - start}')

        return best_score 


    def get_lightdock_score(self, x):
        if self.scoring_func == 'dfire2':
            if type(x) != list:
                x = x.squeeze().tolist()
            score = dfire2_torch(
                x,
                self.receptor_coords,
                self.ligand_coords, 
                self.res_index, 
                self.atom_index, 
                self.potentials, 
                self.adapter.receptor_model.restraints, 
                self.adapter.ligand_model.restraints,
                setup=self.setup,
                receptor=self.receptor,
                ligand=self.ligand,
            )
        elif self.scoring_func == 'dfire':
            score = get_dfire_score(
                x,
                self.setup,
                self.receptor,
                self.ligand,
                self.receptor_coords,
                self.ligand_coords,
                self.dfire_oracle,
                self.lig_numbs,
                self.rec_numas,
                self.adapter,
            )
        return score 


    def get_check_validity_utils(self):
        '''Define check_validity_utils dict of all items needed 
            to check validity of new config_x (check if it is inside receptor)
        '''
        check_validity_utils = {} 
        check_validity_utils['receptor_kd_tree']  = KDTree(self.receptor_coords.detach().cpu().numpy()  )
        return check_validity_utils


    def optimize_configuration(self, check_validity_utils):
        best_score, best_config = run_turbo(
            self.bounding_box,
            self.n_init,
            check_validity_utils,
            self.max_n_bo_steps,
            self.bsz,
            self.n_epochs,
            self.learning_rte,
            self.verbose_config_opt,
            self.get_lightdock_score,
        )
        return best_config, best_score  
