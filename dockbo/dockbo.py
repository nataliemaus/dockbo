import sys
sys.path.append('../')
from dockbo.utils.fold_utils.nanonet.nanonet import NanoNet 
from dockbo.utils.fold_utils.fold_utils import remove_hetero_atoms_and_hydrogens
import os
import shutil
import uuid
import torch
from dockbo.utils.lightdock_torch_utils.lightdock_constants import (
    TH_DEVICE,
    TH_DTYPE,
    DEFAULT_NMODES_REC,
    DEFAULT_NMODES_LIG,
)
# from dockbo.utils.lightdock_torch_utils.setup_sim import get_setup_from_file 
from dockbo.utils.lightdock_torch_utils.scoring.dfire2_driver import DFIRE2Potential
from dockbo.utils.lightdock_torch_utils.faster_torch_code import (
    dfire2_torch,
    prep_receptor,
    prep_ligand,
    init_adapter,
    update_adapter,
    get_indicies,
    get_ligand_indicies,
    get_coords,
    get_bbox_tensor,
)
from dockbo.utils.bo_utils.bo_utils import TurboState, update_state, generate_batch, get_surr_model
import time 

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
    ):
        self.verbose_timing = verbose_timing 
        start_init = time.time() 
        # temporary directory for lightdock files 
        self.pdb_dir = work_dir + pdb_dir
        # make sure pdb dir is new empty dir
        if not os.path.exists(self.pdb_dir):
            os.mkdir(self.pdb_dir)

        # LightDock Args:
        self.potentials = DFIRE2Potential(work_dir=work_dir) 
        self.setup = {
            'use_anm':use_anm,
            'anm_rec':anm_rec,
            'anm_lig':anm_lig,
            'noxt':True, 'noh':True, 'now':True,
            'verbose_parser':False,
        }
        self.bounding_box = get_bbox_tensor(self.setup)
        if path_to_default_antigen_pdb is not None:
            # copy antibody to simulation directoy
            os.system(f"cp {path_to_default_antigen_pdb} {self.pdb_dir}")
            self.default_antigen_path = self.pdb_dir + '/' + path_to_default_antigen_pdb.split('/')[-1]
            # prep antigen ligand
            self.setup['ligand_pdb'] = self.default_antigen_path 
            self.default_ligand = prep_ligand(self.setup)
        else:
            self.default_antigen_path = None 
            self.default_ligand = None
        self.default_adapter = init_adapter(self.default_ligand )
        self.default_ligand_indicies = None 
        if self.default_ligand is not None:
            self.default_ligand_indicies = get_ligand_indicies(self.default_adapter )

        # TuRBO ARGS: 
        self.max_n_bo_steps = max_n_bo_steps 
        self.n_init = n_init
        self.bsz = bsz
        self.n_epochs = n_epochs
        self.learning_rte = learning_rte
        self.previous_best_config = None 
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
            

        # if no new antigen path is given, use default ligand 
        if path_to_antigen_pdb is None:
            assert self.default_ligand is not None
        
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
        #   otherwise, we assume defaul ligand   
        if path_to_antigen_pdb is not None:
            # copy antibody to simulation directoy
            os.system(f"cp {path_to_antigen_pdb} {directory}")
            antigen_path = directory + '/' + path_to_antigen_pdb.split('/')[-1]
            # prep antigen ligand
            self.setup['ligand_pdb'] = antigen_path
            self.ligand = prep_ligand(self.setup)
            new_ligand = True
        else:
            self.setup['ligand_pdb'] = self.default_antigen_path 
            self.ligand = self.default_ligand 
            new_ligand = False
        
        # prep antibody receptor 
        self.setup['receptor_pdb'] = antibody_path
        self.receptor = prep_receptor(self.setup) 
        # update adapter 
        self.adapter = update_adapter(self.default_adapter, self.receptor, self.ligand, new_ligand=new_ligand )
        # get combined indicies for receptor and ligand 
        self.res_index, self.atom_index = get_indicies(self.adapter, self.default_ligand_indicies)
        # get coords for receptor and ligand 
        self.receptor_coords, self.ligand_coords = get_coords(self.adapter)


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
        if config_x is None: 
            best_config, best_score = self.optimize_configuration() 
            self.previous_best_config = best_config
        # otherwise, use given config_x (27,) to get score
        else:
            best_score = self.get_lightdock_score(config_x)
            self.previous_best_config = config_x.unsqueeze(0)
        if self.verbose_timing:
            print(f'compute score time: {time.time() - start}')

        return best_score 


    def get_lightdock_score(self, x):
        if type(x) != list:
            x = x.tolist()
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

        return score 


    def optimize_configuration(self):
        # GOAL: Find optimal ld_sol (27 numbers) within bounds given by bounding_box
        #   which are the optimal configuraiton of ligand and protein to get max score 
        lower_bounds = self.bounding_box[:,0].cuda() 
        upper_bounds = self.bounding_box[:,1].cuda()
        bound_range = (upper_bounds - lower_bounds).cuda() 

        # xs normalized 0 to 1, unnormalize to get lightdock scores 
        def unnormalize(x):
            unnormalized = x.cuda()*bound_range + lower_bounds
            return unnormalized 

        # Initialization data
        train_x = torch.rand(self.n_init, 27)  # random between 0 and 1 
        if self.previous_best_config is not None: # initialize w/ best config from previous opt
            train_x = torch.cat((train_x, self.previous_best_config)) 
        train_y = [self.get_lightdock_score(unnormalize(x)) for x in train_x]
        train_y = torch.tensor(train_y).float().unsqueeze(-1)

        # Run TuRBO 
        turbo_state = TurboState() 
        for _ in range(self.max_n_bo_steps): 
            # get surr model updated on data 
            surr_model = get_surr_model( 
                train_x=train_x,
                train_y=train_y,
                n_epochs=self.n_epochs,
                learning_rte=self.learning_rte,
            )
            # generate batch of candidates in trust region w/ thompson sampling
            x_next = generate_batch(
                state=turbo_state,
                model=surr_model,  # GP model
                X=train_x,  # Evaluated points on the domain [0, 1]^d
                Y=train_y,  # Function values
                batch_size=self.bsz,
                dtype=TH_DTYPE,
                device=TH_DEVICE,
            )
            # compute scores for batch of candidates 
            y_next = [self.get_lightdock_score(unnormalize(x)) for x in x_next]
            y_next = torch.tensor(y_next).float().unsqueeze(-1)
            # update data 
            train_x = torch.cat((train_x, x_next.detach().cpu()))
            train_y = torch.cat((train_y, y_next)) 
            # update turbo state 
            turbo_state = update_state(turbo_state, y_next)
            if self.verbose_config_opt:
                print(f'N configs evaluated:{len(train_y)}, best config score:{train_y.max().item()}')

        best_score = train_y.max().item() 
        best_config = train_x[train_y.argmax()].squeeze().unsqueeze(0)

        # best_config
        # tensor([[0.6578, 0.3866, 0.9290, 0.3974, 0.5406, 0.4401, 0.4125, 0.3368, 0.3825,
        #  0.4919, 0.9924, 0.9122, 0.1150, 0.5077, 0.2370, 0.0897, 0.3209, 0.8278,
        #  0.9107, 0.1867, 0.6293, 0.0325, 0.4872, 0.4158, 0.8804, 0.4152, 0.3239]])
        # best score: 684,672

        return best_config, best_score  

# python3 example.py /home/nmaus/

