import torch 
import sys 
sys.path.append("../")
import os 
# from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull # , convex_hull_plot_2d
from pathlib import Path
# from pytorch3d import transforms 
from utils.quaternion_utils import quaternion_invert, quaternion_apply
EPS = 1e-10 
import glob 
os.environ["WANDB_SILENT"] = "true" 
from dockbo.utils.lightdock_torch_utils.lightdock_constants import (
    DEFAULT_LIST_EXTENSION,
    TH_DEVICE,
    TH_DTYPE,
)
import wandb 
from dockbo.utils.lightdock_torch_utils.structure.complex import Complex 
# from dockbo.utils.lightdock_torch_utils.PDBIO import write_pdb_to_file
import numpy as np 
# from dockbo.utils.lightdock_torch_utils.setup_sim import read_input_structure
import copy 
from dockbo.utils.lightdock_torch_utils.scoring.dfire_driver import DefinedModelAdapter # as DfireAdapter
from dockbo.utils.lightdock_torch_utils.scoring.dfire2_driver import DFIRE2_ATOM_TYPES # DefinedModelAdapter 
from dockbo.utils.lightdock_torch_utils.PDBIO import parse_complex_from_file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch.nn as nn 
import math 
import gc 

def get_pdb_files(input_file):
    """Get a list of the PDB files in the input_file"""
    file_names = []
    with open(input_file) as handle:
        for line in handle:
            file_name = line.rstrip(os.linesep)
            if Path(file_name).exists():
                file_names.append(file_name)
    return file_names


def read_input_structure(
    pdb_file_name,
    ignore_oxt=True,
    ignore_hydrogens=True,
    ignore_water=True,
    verbose_parser=False,
):
    """Reads the input structure.

    The arguments pdb_file_name can be a PDB file or a file
    containing a list of PDB files.

    ignore_oxt flag avoids saving OXT atoms.
    """
    atoms_to_ignore = []
    residues_to_ignore = []
    if ignore_oxt:
        atoms_to_ignore.append("OXT")
    if ignore_hydrogens:
        atoms_to_ignore.append("H")
    if ignore_water:
        residues_to_ignore.append("HOH")

    structures = []
    file_names = []
    file_name, file_extension = os.path.splitext(pdb_file_name)
    if file_extension == DEFAULT_LIST_EXTENSION:
        file_names.extend(get_pdb_files(pdb_file_name))
    else:
        file_names.append(pdb_file_name)
    for file_name in file_names:
        atoms, residues, chains = parse_complex_from_file(
            file_name, atoms_to_ignore, residues_to_ignore, verbose_parser
        )
        structures.append(
            {
                "atoms": atoms,
                "residues": residues,
                "chains": chains,
                "file_name": file_name,
            }
        )

    # Representatives are now the first structure, but this could change in the future
    structure = Complex.from_structures(structures)
    return structure

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

def prep_structure(pdb_filename, move_to_origin=False):
    structure = read_input_structure(pdb_filename)
    if move_to_origin:
        origin_translation = structure.move_to_origin() 
        origin_translation = torch.tensor(origin_translation).float() # (3,) gives translation of receptor 
    else:
        origin_translation = None
    return structure, origin_translation


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

def get_indicies(adapter):
    receptor_indicies = get_receptor_indicies(adapter) 
    res_index, atom_index, last = receptor_indicies 
    ligand_res_indicies, ligand_atom_indicies = get_ligand_indicies(adapter, last)
    res_index = res_index + ligand_res_indicies
    atom_index = atom_index + ligand_atom_indicies
    res_index = torch.tensor(res_index, dtype=torch.long, device=TH_DEVICE)
    atom_index = torch.tensor(atom_index, dtype=torch.long, device=TH_DEVICE)
    return res_index, atom_index 

def get_adapter(ligand_structure, receptor_structure):
        default_adapter = DefinedModelAdapter(
            receptor=receptor_structure, 
            ligand=ligand_structure,
            receptor_restraints=None,
            ligand_restraints=None,
        )

        adapter = update_adapter(
                adapter=default_adapter,
                receptor=receptor_structure,
                receptor_restraints=None,
                ligand=ligand_structure,
                ligand_restraints=None,
                new_receptor=False,
                new_ligand=False,
            )
        return adapter 


def get_data(
    data_dict
):
    C = 10
    # Randomly rotate combined structure/pose
    #   rotation the entire combined configuration does not change the pose 
    init_R = torch.randn(4,)
    init_R = init_R / torch.linalg.vector_norm(init_R) 
    ligand_coords = data_dict['ligand_coords'] - data_dict['ab_origin_translation'].cuda() 
    receptor_coords = data_dict['receptor_coords'] - data_dict['ag_origin_translation'].cuda() 
    ligand_coords = quaternion_apply(init_R, ligand_coords) 
    receptor_coords = quaternion_apply(init_R, receptor_coords) 
    ligand_coords = ligand_coords + data_dict['ab_origin_translation'].cuda() 
    receptor_coords = receptor_coords + data_dict['ag_origin_translation'].cuda() 

    # Randomly translate both ligand and receptor 
    #   translating combined structure does not change configuraiotn 
    init_T = torch.randn(3,).cuda()*C
    receptor_coords = receptor_coords - data_dict['ag_origin_translation'].cuda() + init_T
    # also add back the origin translation to recover init receptor coords 

    # The ground truth ab T we want to recover is the reverse if the origin T plus the random init T
    ab_translation_label = - data_dict['ab_origin_translation'].cuda() + init_T 

    # Now randomly rotate the ligand (ab) away from starting/currect position 
    ab_rotation_label = torch.randn(4,) 
    ab_rotation_label = ab_rotation_label / torch.linalg.vector_norm(ab_rotation_label) 
    inverse_rotation = quaternion_invert(ab_rotation_label) 
    ligand_coords = quaternion_apply(inverse_rotation, ligand_coords) 

    data_dict['label'] = torch.cat((ab_translation_label.cuda(), ab_rotation_label.cuda())).unsqueeze(0)

    ab_coords = ligand_coords[data_dict['ab_convex_hull']]
    ag_coords = receptor_coords[data_dict['ag_convex_hull']]
    ab_atom_idxs = data_dict['ab_atom_idxs'][data_dict['ab_convex_hull']]
    ag_atom_idxs = data_dict['ag_atom_idxs'][data_dict['ag_convex_hull']]

    ab_data = torch.cat((ab_coords, ab_atom_idxs.cuda() ), -1) # (N_atoms, 4)
    ag_data = torch.cat(( ag_coords, ag_atom_idxs.cuda() ), -1) # (N_atoms, 4) 

    # points in convex hull only 
    data_dict['ab_data'] = ab_data # points in convex hull only 
    data_dict['ag_data'] = ag_data

    return data_dict


def get_new_pose(
    translation,
    rotation,
    current_coords,
):
    norm = torch.linalg.vector_norm(rotation) 
    rotation = rotation / max(norm, EPS) 
    if rotation is None:
        new_pose = current_coords + translation
    else:
        new_pose = quaternion_apply(rotation.cuda(), current_coords.cuda()) + translation.cuda() 
    return new_pose 


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5_000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x) 


class Net(nn.Module):
    """
    Transformer to predict rotation and translation 
    https://n8henrie.com/2021/08/writing-a-transformer-classifier-in-pytorch/
    """
    def __init__(
        self,
        nhead=8,
        dim_feedforward=2048,
        num_layers=6,
        activation="relu",
        embedding_dim = 64,
        pos_dropout = 0.1,
        max_len = 5_000,
        enc_dropout = 0.1,
        extra_dropout=0.1,
    ):
        super().__init__()

        self.vocab_size = len(DFIRE2_ATOM_TYPES)
        self.atom_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embedding_dim - 3)
        self.ab_position_encoding = PositionalEncoding(embedding_dim, dropout=pos_dropout, max_len=max_len)
        self.ag_position_encoding = PositionalEncoding(embedding_dim, dropout=pos_dropout, max_len=max_len)
        self.ab_ag_position_encoding = PositionalEncoding(embedding_dim, dropout=pos_dropout, max_len=max_len)
        
        ab_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=enc_dropout,
            activation=activation,
            batch_first=True,
        )
        self.ab_transformer_encoder = nn.TransformerEncoder(
            ab_encoder_layer,
            num_layers=num_layers,
        )

        ag_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=enc_dropout,
            activation=activation,
            batch_first=True,
        )
        self.ag_transformer_encoder = nn.TransformerEncoder(
            ag_encoder_layer,
            num_layers=num_layers,
        )

        ab_ab_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=enc_dropout,
            activation=activation,
            batch_first=True,
        )
        self.ab_ag_transformer_encoder = nn.TransformerEncoder(
            ab_ab_encoder_layer,
            num_layers=num_layers,
        )

        self.dropout = nn.Dropout(p=extra_dropout)
        self.fc1 = nn.Linear(embedding_dim*3,embedding_dim*3//2 )
        self.fc2 = nn.Linear(embedding_dim*3//2,embedding_dim*3//4 )
        self.fc3 = nn.Linear(embedding_dim*3//4, 7 )
        self.mse_loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean') 
        self.cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)


    def forward(self, x):
        ab_data, ag_data = x # torch.Size([62, 4]), torch.Size([69, 4])) 
        ab_coords, ag_coords = ab_data[:,:,0:3], ag_data[:,:,0:3] #  (torch.Size([62, 3]), torch.Size([69, 3]))
        ab_atoms, ag_atoms = ab_data[:,:,3], ag_data[:,:,3] # torch.Size([62]), torch.Size([69]))
        ab_atoms = self.atom_embedding(ab_atoms.int().cuda()) # torch.Size([1, 62, 256])
        ag_atoms = self.atom_embedding(ag_atoms.int().cuda()) # torch.Size([1, 69, 256])

        ab = torch.cat((ab_coords, ab_atoms), -1) # torch.Size([1, 62, 256])
        ag = torch.cat((ag_coords, ag_atoms), -1) # torch.Size([1, 69, 256])
        separation = torch.zeros(ab.shape[0], 1, ab.shape[-1]).cuda()  # torch.Size([1, 1, 64])
        ab_ag = torch.cat((ab, separation, ag), -2) # torch.Size([1, 132, 64])

        ab = self.ab_position_encoding(ab) # torch.Size([1, 62, 64])
        ag = self.ag_position_encoding(ag) # torch.Size([1, 69, 64]) 
        ab_ag = self.ab_ag_position_encoding(ab_ag) # torch.Size([1, 132, 64]) 

        ab = self.ab_transformer_encoder(ab) # torch.Size([1, 62, 256])
        ag = self.ag_transformer_encoder(ag) # torch.Size([1, 69, 256])
        ab_ag = self.ab_ag_transformer_encoder(ab_ag) # torch.Size([1, 132, 64])
        ab = ab.mean(dim=1) # torch.Size([1, 256]) = bsz x 256  
        ag = ag.mean(dim=1) # torch.Size([1, 256]) = bsz x 256 
        ab_ag = ab_ag.mean(dim=1) #  # torch.Size([1, 256]) = bsz x 256 
        output = torch.cat((ab, ag, ab_ag), -1) # torch.Size([1, 192])  = bsz x embed_dim*3 
        output = self.fc1(output)
        output = torch.nn.functional.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = torch.nn.functional.relu(output) 
        output = self.dropout(output)
        output = self.fc3(output)  # torch.Size([1, 7]) = bsz x 7 
        # output[:,3:] = torch.nn.functional.normalize(output[:,3:].clone(), p=2.0, dim=1, eps=1e-12, out=None)

        # normalize rotation 
        # rots = output[:,3:]
        # norm = rots.norm(p=2, dim=1, keepdim=True)
        # rots = rots.div(norm.expand_as(rots))
        # output[:,3:] = rots

        return output 


    def loss(self, model_output, label):
        true_T = label[:, 0:3] 
        pred_T = model_output[:, 0:3]

        true_R = label[:, 3:] 
        pred_R = model_output[:, 3:] 
        pred_R = torch.nn.functional.normalize(pred_R.clone(), p=2.0, dim=1, eps=1e-12, out=None)
        rot_normed_mse = self.mse_loss(pred_R, true_R) 

        mse = self.mse_loss(model_output, label)
        cos_sim = self.cos_similarity(true_T, pred_T)

        return mse, cos_sim.sum(), rot_normed_mse 




def get_convex_hull(
    ligand_coords, 
    receptor_coords,
):
    # include only ab and ag atoms in convex hull 
    ab_convex_hull = ConvexHull(ligand_coords.detach().cpu().numpy()).vertices  
    ag_convex_hull = ConvexHull(receptor_coords.detach().cpu().numpy()).vertices  

    return ab_convex_hull, ag_convex_hull 


def prep_data_dict(antibody_pdb_file, antigen_pdb_file):
    ab_structure, ab_origin_translation = prep_structure(antibody_pdb_file, move_to_origin=True)
    ag_structure, ag_origin_translation = prep_structure(antigen_pdb_file, move_to_origin=True) 
    adapter = get_adapter(
        ligand_structure=ab_structure,
        receptor_structure=ag_structure,
    )
    receptor_coords, ligand_coords = get_coords(adapter) # both start at origin 
    
    ab_atom_ids = [] 
    for mol in ab_structure.residues: 
        for atom in mol.atoms: 
            ab_atom_ids.append(atom.residue_name + " " + atom.name)
    ag_atom_ids = [] 
    for mol in ag_structure.residues: 
        for atom in mol.atoms: 
            ag_atom_ids.append(atom.residue_name + " " + atom.name)
    ab_atom_idxs = torch.tensor([DFIRE2_ATOM_TYPES[atom_id] for atom_id in ab_atom_ids]).float().unsqueeze(-1)
    ag_atom_idxs = torch.tensor([DFIRE2_ATOM_TYPES[atom_id] for atom_id in ag_atom_ids]).float().unsqueeze(-1)
    ab_convex_hull, ag_convex_hull = get_convex_hull(ligand_coords, receptor_coords) 
    data_dict = {} 

    data_dict['ab_convex_hull'] = ab_convex_hull
    data_dict['ag_convex_hull'] = ag_convex_hull
    data_dict['receptor_coords'] = receptor_coords
    data_dict['ligand_coords'] = ligand_coords
    data_dict['ab_atom_idxs'] = ab_atom_idxs
    data_dict['ag_atom_idxs'] = ag_atom_idxs
    data_dict['ab_origin_translation'] = ab_origin_translation
    data_dict['ag_origin_translation'] = ag_origin_translation

    return data_dict 


def prep_data(args):
    pdb_id_to_data = {}
    all_ab_files = glob.glob(f"../parsed_chothia_pdbs/*_ab.pdb") # 1739 total examples 
    all_ab_files.sort() 
    if args.debug:
        all_ab_files = all_ab_files[0:20] 
    for antibody_pdb_file in all_ab_files: 
        pdb_id = antibody_pdb_file.split("/")[-1][0:-7]
        antigen_pdb_file = f"../parsed_chothia_pdbs/{pdb_id}_ag.pdb"
        data_dict = prep_data_dict(antibody_pdb_file, antigen_pdb_file)
        pdb_id_to_data[pdb_id] = data_dict 
    return pdb_id_to_data


def start_wandb(args_dict):
    import wandb 
    tracker = wandb.init(
        entity=args_dict['wandb_entity'], 
        project=args_dict['wandb_project_name'],
        config=args_dict, 
    ) 
    print('running', wandb.run.name) 
    return tracker 


def train(args):
    lowest_loss = torch.inf 
    args_dict = vars(args)
    tracker = start_wandb(args_dict)
    print("starting data prep")
    pdb_id_to_data_dict = prep_data(args) 
    print("finished data prep ")
    model_save_path = 'saved_models/' + wandb.run.name + '_model_state.pkl'  
    all_ab_files = glob.glob(f"../parsed_chothia_pdbs/*_ab.pdb") # 1739 total examples 
    all_ab_files.sort() 
    if args.debug:
        all_ab_files = all_ab_files[0:20] 
    perc90 = int(len(all_ab_files)*0.9)
    train_ab_files = all_ab_files[0:perc90]
    val_ab_files = all_ab_files[perc90:]
    model = Net(
        nhead=args.nhead, # 8
        dim_feedforward=args.dim_feedforward, 
        num_layers=args.num_layers, # 6,
        activation="relu",
        embedding_dim=args.embedding_dim,  #64,256? 
        pos_dropout=args.pos_dropout, # 0.1,
        max_len=args.max_len,  #  1_000,
        enc_dropout=args.enc_dropout, # 0.1,
        extra_dropout=args.extra_dropout,
    ) 
    model = model.cuda() 
    model = model.train() 
    optimizer = torch.optim.Adam([
        {'params': model.parameters()}, ], 
        lr=args.lr
    ) 
    for epoch in range(args.max_epochs): 
        losses = []
        mses = []
        cos_sims = []
        rot_norm_mses = []
        for antibody_pdb_file in train_ab_files: 
            pdb_id = antibody_pdb_file.split("/")[-1][0:-7]
            data_dict = pdb_id_to_data_dict[pdb_id]

            # form batch from multiple translation + rotation of same pose/ same pdb 
            ab_datas = [] 
            ag_datas = []
            labels = [] 
            for _ in range(args.bsz): 
                new_dict = get_data(data_dict) 
                labels.append(data_dict['label'] )
                ab_datas.append(new_dict['ab_data'].unsqueeze(0)) 
                ag_datas.append(new_dict['ag_data'].unsqueeze(0))

            ab_data = torch.cat(ab_datas, 0 ) 
            ag_data = torch.cat(ag_datas, 0 ) 
            label = torch.cat(labels, 0 ) 

            x = ab_data.cuda(), ag_data.cuda()  
            optimizer.zero_grad() 
            output = model(x) 

            mse, cos_sim, rot_normed_mse = model.loss(output, label ) 
            loss = mse - cos_sim + rot_normed_mse # lowest possible loss = -1 
            loss.backward()
            optimizer.step() 

            losses.append(loss.item())
            mses.append(mse.item())
            cos_sims.append(cos_sim.item())
            rot_norm_mses.append(rot_normed_mse.item())

            tracker.log({
                'mse':mse.item(),
                'cos_sim':cos_sim.item(),
                'rot_normed_mse':rot_normed_mse.item(),
                'loss':loss.item(),
            }) 

        avg_train_loss = np.array(losses).mean()
        avg_train_mse = np.array(mses).mean()
        avg_train_cos_sim = np.array(cos_sims).mean() 
        avg_train_rot_normed_mse = np.array(rot_norm_mses).mean()
        tracker.log({
            'avg_train_loss':avg_train_loss, 
            'epoch':epoch,
            'avg_train_mse':avg_train_mse,
            'avg_train_cos_sim':avg_train_cos_sim,
            'avg_train_rot_normed_mse':avg_train_rot_normed_mse,
        })
    
        if epoch % args.compute_val_freq == 0: 
            gc.collect() 
            model = model.eval()  
            losses = []
            mses = []
            cos_sims = []
            rot_norm_mses = []
            for antibody_pdb_file in val_ab_files: 
                pdb_id = antibody_pdb_file.split("/")[-1][0:-7]
                data_dict = pdb_id_to_data_dict[pdb_id]
                new_dict = get_data(data_dict) 
                ab_data = new_dict['ab_data']
                ag_data = new_dict['ag_data']
                label = data_dict['label'] 
                x = ab_data.unsqueeze(0).cuda(), ag_data.unsqueeze(0).cuda()  
                output = model(x) 
                mse, cos_sim, rot_normed_mse = model.loss(output, label ) 
                loss = mse - cos_sim + rot_normed_mse # lowest possible loss = -1 
                losses.append(loss.item())
                mses.append(mse.item())
                cos_sims.append(cos_sim.item())
                rot_norm_mses.append(rot_normed_mse.item())

            avg_val_loss = np.array(losses).mean()
            avg_val_mse = np.array(mses).mean()
            avg_val_cos_sim = np.array(cos_sims).mean() 
            avg_val_rot_normed_mse = np.array(rot_norm_mses).mean()
            tracker.log({
                'avg_val_loss':avg_val_loss, 
                'epoch':epoch,
                'avg_val_mse':avg_val_mse,
                'avg_val_cos_sim':avg_val_cos_sim,
                'avg_val_rot_normed_mse':avg_val_rot_normed_mse,
            })
            
            if avg_val_loss < lowest_loss: 
                lowest_loss = avg_val_loss 
                tracker.log({'lowest_avg_val_loss': lowest_loss, 'saved_model_at_epoch': epoch+1 }) 
                torch.save(model.state_dict(), model_save_path) 
            model = model.train() # back in train mode 

    tracker.finish() 


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--lr', type=float, default=0.00001)  
    parser.add_argument('--compute_val_freq', type=int, default=5 ) 
    parser.add_argument('--load_ckpt_wandb_name', default="" ) 
    parser.add_argument('--max_epochs', type=int, default=1_000_000_000 ) 
    parser.add_argument('--dim_feedforward', type=int, default=1024 )   
    parser.add_argument('--nhead', type=int, default=8 )   
    parser.add_argument('--num_layers', type=int, default=6 )   
    parser.add_argument('--embedding_dim', type=int, default=128 )  
    parser.add_argument('--pos_dropout', type=float, default=0.1 )  
    parser.add_argument('--enc_dropout', type=float, default=0.1 ) 
    parser.add_argument('--extra_dropout', type=float, default=0.1 ) 
    parser.add_argument('--max_len', type=int, default=1_000 ) 
    parser.add_argument('--bsz', type=int, default=128 ) 
    parser.add_argument('--debug', type=bool, default=False ) 
    parser.add_argument('--wandb_entity', default="nmaus" )
    parser.add_argument('--wandb_project_name', default="train-ab-binding-model" )  
    args = parser.parse_args() 

    # CUDA_VISIBLE_DEVICES=1 python3 pose_prediction_model.py --lr 0.00005 --dim_feedforward 4096 --bsz 256 --num_layers 32 --nhead 8

    train(args) 

    # CUDA_VISIBLE_DEVICES=1

# TODO: add padding so that we can use bsz > 1 