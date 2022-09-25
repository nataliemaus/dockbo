import sys
sys.path.append('../')
from dockbo.utils.fold_utils.nanonet.nanonet import NanoNet 
from dockbo.utils.fold_utils.fold_utils import remove_hetero_atoms_and_hydrogens
import os
import shutil
import uuid
import torch
from dockbo.utils.lightdock_torch_utils.lightdock_constants import TH_DEVICE, TH_DTYPE
from dockbo.utils.lightdock_torch_utils.setup_sim import get_setup_from_file 
from dockbo.utils.lightdock_torch_utils.scoring.dfire2_driver import DFIRE2Potential
from dockbo.utils.lightdock_torch_utils.faster_torch_code import (
    dfire2_torch,
    prep_receptor_and_ligand,
    get_adapter,
    get_indicies,
    get_coords,
    get_bbox_tensor,
)
from dockbo.utils.bo_utils.bo_utils import TurboState, update_state, generate_batch, get_surr_model


class DockBO():

    def __init__(
        self,
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
    ):
        # Argss for TuRBO configuration optimization:
        self.max_n_bo_steps = max_n_bo_steps 
        self.n_init = n_init
        self.bsz = bsz
        self.n_epochs = n_epochs
        self.learning_rte = learning_rte
        self.potentials = DFIRE2Potential(work_dir=work_dir) 
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
        self.pdb_dir = work_dir + pdb_dir
        # make sure pdb dir is new empty dir
        if not os.path.exists(self.pdb_dir):
            os.mkdir(self.pdb_dir)

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


    def __call__(
        self,
        path_to_antigen_pdb,
        antibody_aa_seq=None, 
        path_to_antibody_pdb=None, # option to pass in antibody pdb file direction, otherwise we fold seq
    ):
        ''' Inputs: 
                protein_string: string of amino acids for antibody
            Output: 
                lightdock score 
        '''
        # grab original work dir
        self.og_dir = os.getcwd()
        if path_to_antibody_pdb is None: # if no antibody pdb provided 
            assert antibody_aa_seq is not None # must provide aa seq to fold 
            # fold protein and create pdb files with igfold or nanonet 
            path_to_antibody_pdb, temp_pdb_dir = self.fold_protein(antibody_aa_seq)
        else: # otherwise use anitbody pdb file directly 
            temp_pdb_dir = self.pdb_dir + '/' + str(uuid.uuid1())
            os.mkdir(temp_pdb_dir) 
            # copy antibody to working directoy 
            os.system(f"cp {path_to_antibody_pdb} {temp_pdb_dir}")
            path_to_antibody_pdb = temp_pdb_dir + '/' + path_to_antibody_pdb.split('/')[-1]

        # create new dir for simulation files
        new_ab_directory = f"{temp_pdb_dir}/{uuid.uuid1()}"
        os.makedirs(new_ab_directory)
        # run lightdock simulation to get binding score
        top_score = self.run_docking(
            new_ab_directory,
            path_to_antibody_pdb,
            path_to_antigen_pdb,
        )
        # remove temp dir
        shutil.rmtree(temp_pdb_dir)
        # reset to initial dir 
        os.chdir( self.og_dir )

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


    def run_docking(
        self,
        directory,
        antibody_path,
        path_to_antigen_pdb,
    ):
        # copy antibody to working directoy 
        os.system(f"cp {path_to_antigen_pdb} {directory}")
        antigen_path = directory + '/' + path_to_antigen_pdb.split('/')[-1]
        # run light dock setup 
        os.chdir(directory)
        if self.restraint_file:
            os.system(f"lightdock3_setup.py {antibody_path} {antigen_path} \
                --noxt --noh --now -anm -r {self.restraint_file}")
        else:
            os.system(f"lightdock3_setup.py {antibody_path} {antigen_path} \
                --noxt --noh --now -anm")
        self.setup = get_setup_from_file('setup.json')
        self.receptor, self.ligand = prep_receptor_and_ligand(self.setup)
        self.adapter = get_adapter(self.receptor, self.ligand) # takes longest time 
        self.res_index, self.atom_index = get_indicies(self.adapter)
        self.receptor_coords, self.ligand_coords = get_coords(self.adapter)
        self.bounding_box = get_bbox_tensor(self.setup)

        best_config, best_score = self.optimize_configuration() 
        self.previous_best_config = best_config

        return best_score 


    def get_lightdock_score(self, x):
        score = dfire2_torch(
            x.tolist(),
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
        #   = optimal configuraiton of ligand and protein to get max score 
        lower_bounds = self.bounding_box[:,0].cuda()  # torch.Size([1, 27])
        upper_bounds = self.bounding_box[:,1].cuda()   # torch.Size([1, 27])
        bound_range = (upper_bounds - lower_bounds).cuda() 
        # train_x normalized 0 to 1, unnormalize to get lightdock score 
        # train_y remains unnormalized! (for now, tbd)
        def unnormalize(x):
            unnormalized = x.cuda()*bound_range + lower_bounds
            return unnormalized 
        # Initialization data
        train_x = torch.rand(self.n_init, 27)  # random between 0 and 1 
        if self.previous_best_config is not None: # initialize w/ best config from previous opt
            train_x = torch.cat((train_x, self.previous_best_config)) 
        train_y = [self.get_lightdock_score(unnormalize(x)) for x in train_x]
        train_y = torch.tensor(train_y).float().unsqueeze(-1)
        turbo_state = TurboState() 
        for _ in range(self.max_n_bo_steps): 
            surr_model = get_surr_model( # get surr model updated on data 
                train_x=train_x,
                train_y=train_y,
                n_epochs=self.n_epochs,
                learning_rte=self.learning_rte,
            )
            x_next = generate_batch(
                state=turbo_state,
                model=surr_model,  # GP model
                X=train_x,  # Evaluated points on the domain [0, 1]^d
                Y=train_y,  # Function values
                batch_size=self.bsz,
                dtype=TH_DTYPE,
                device=TH_DEVICE,
            )
            y_next = [self.get_lightdock_score(unnormalize(x)) for x in x_next]
            y_next = torch.tensor(y_next).float().unsqueeze(-1)
            train_x = torch.cat((train_x, x_next.detach().cpu()))
            train_y = torch.cat((train_y, y_next)) # , dim=-1)
            turbo_state = update_state(turbo_state, y_next)
            if self.verbose_config_opt:
                print(f'N configs evaluated:{len(train_y)}, best config score:{train_y.max().item()}')

        best_score = train_y.max().item() 
        best_config = train_x[train_y.argmax()].squeeze().unsqueeze(0)

        return best_config, best_score  


