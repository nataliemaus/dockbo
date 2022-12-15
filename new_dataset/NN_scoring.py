import pandas as pd 
import numpy as np 
import torch 
import torch.nn as nn 
import argparse 
import wandb 
from torch.utils.data import TensorDataset, DataLoader

def load_data():
    df = pd.read_csv('rani_display_with_sequences.csv')
    cdr3s = df['cdr3'].values # (96846,)
    log_r3_r2_rep1 = df['log(R3/R2) replicate 1'].values
    bool_arr = np.logical_not(np.isnan(log_r3_r2_rep1))
    cdr3s = cdr3s[bool_arr] # (67769,)
    log_r3_r2_rep1 = log_r3_r2_rep1[bool_arr] # (67,769,)
    lengths = [len(cdr3) for cdr3 in cdr3s]
    lengths = np.array(lengths) 
    min_length = lengths.min() # 8 
    max_length = lengths.max() # 18 
    # get vocab 
    all_chars = []
    for cdr3 in cdr3s: 
        for char in cdr3:
            all_chars.append(char)
    all_chars = np.array(all_chars)
    vocab = np.unique(all_chars) # (21,) 
    vocab = {char: ix+1 for ix, char in enumerate(vocab)}
    vocab['-'] = 0 # pad token 

    return vocab, max_length, cdr3s, log_r3_r2_rep1

class Net(nn.Module):
    """
    Transformer to predict values from cdrs 
    https://n8henrie.com/2021/08/writing-a-transformer-classifier-in-pytorch/
    """
    def __init__(
        self,
        vocab,
        max_seq_length=18,
        embedding_dim=16,
        nhead=8,
        dim_feedforward=256,
        num_layers=6,
        enc_dropout=0.1,
        extra_dropout=0.1,
        attention=False,
    ):
        super().__init__()
        self.vocab = vocab 
        self.attention = attention
        self.max_seq_length = max_seq_length
        self.vocab_size = len(vocab)
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, 
            embedding_dim=embedding_dim
        )
        if self.attention:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=enc_dropout,
                activation="relu",
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        self.dropout = nn.Dropout(p=extra_dropout) 
        self.fc1 = nn.Linear(embedding_dim, embedding_dim//2)
        self.fc2 = nn.Linear(embedding_dim//2, 1)
        self.mse_loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean') 

    def forward(self, x):
        x = self.embedding(x)
        if self.attention:
            x = self.transformer_encoder(x) 
        x = x.mean(dim=1) 
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x) 

        return x

    def loss(self, model_output, label):
        mse = self.mse_loss(model_output, label)
        return mse 

    def tokenize_seqs(self, seqs):
        tokenized_seqs = []
        for seq in seqs:
            tokenized_seqs.append(self.tokenize_seq(seq).unsqueeze(0))
        tokenized_seqs = torch.cat(tokenized_seqs)
        
        return tokenized_seqs


    def tokenize_seq(self, seq):
        while len(seq) < self.max_seq_length:
            seq += '-' # pad tokens to get to length 18
        idxs = [self.vocab[char] for char in seq] 
        idxs = torch.tensor(idxs)
        assert idxs.shape[0] == self.max_seq_length

        return idxs 


def start_wandb(args_dict):
    tracker = wandb.init(
        entity=args_dict['wandb_entity'], 
        project=args_dict['wandb_project_name'],
        config=args_dict, 
    ) 
    print('running', wandb.run.name) 
    return tracker 


def train(args):
    lowest_loss = torch.inf 
    tracker = start_wandb(vars(args))
    print("finished data prep ") 
    model_save_path = 'saved_models/' + wandb.run.name + '_model_state.pkl'  

    vocab, max_seq_length, cdr3s, log_r3_r2_rep1 = load_data()
    net =  Net(
        vocab=vocab,
        max_seq_length=max_seq_length,
    )
    X = net.tokenize_seqs(cdr3s) 
    Y = torch.from_numpy(log_r3_r2_rep1).float() 
    N_train = int(X.shape[0]*0.9) 
    train_X = X[0:N_train]
    train_Y = Y[0:N_train]
    val_X = X[N_train:]
    val_Y = Y[N_train:]
    model = Net(
        vocab=vocab,
        max_seq_length=max_seq_length,
        embedding_dim=args.embedding_dim, # 16
        nhead=args.nhead, # 8
        dim_feedforward=args.dim_feedforward, # 256
        num_layers=args.num_layers, # 6,
        enc_dropout=args.enc_dropout, # 0.1,
        extra_dropout=args.extra_dropout, # 0.1 
        attention=args.attention,
    )
    if args.load_ckpt_wandb_name: 
        path_to_state_dict = 'saved_models/' + args.load_ckpt_wandb_name + '_model_state.pkl'  
        state_dict = torch.load(path_to_state_dict) # load state dict 
        model.load_state_dict(state_dict, strict=True) 
    model = model.cuda() 
    model = model.train() 
    optimizer = torch.optim.Adam([ 
        {'params': model.parameters()}, ], 
        lr=args.lr
    ) 
    train_dataset = TensorDataset(train_X.cuda(), train_Y.cuda())
    train_loader = DataLoader(train_dataset, batch_size=args.bsz)
    val_dataset = TensorDataset(val_X.cuda(), val_Y.cuda())
    val_loader = DataLoader(val_dataset, batch_size=args.bsz)
    lowest_loss = torch.inf 
    for epoch in range(args.max_epochs): 
        model = model.train() 
        losses = [] 
        for ix, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(batch_x) 
            loss = model.loss(output, batch_y.unsqueeze(-1))
            losses.append(loss.item() )
            tracker.log({"train_loss_per_batch": loss.item() })
            loss.backward()
            optimizer.step() 
        avg_train_loss = np.array(losses).mean() 
        with torch.no_grad():
            model = model.eval() 
            val_losses = []
            for ix, (batch_x, batch_y) in enumerate(val_loader):
                output = model(batch_x) 
                loss = model.loss(output, batch_y.unsqueeze(-1))
                val_losses.append(loss.item() )
                tracker.log({"val_loss_per_batch": loss.item() })
            avg_val_loss = np.array(val_losses).mean() 
        tracker.log({
            'avg_train_loss':avg_train_loss.item(),
            'avg_val_loss':avg_val_loss.item(),
            'epoch':epoch,
        }) 
        if avg_val_loss < lowest_loss: 
            lowest_loss = avg_val_loss 
            tracker.log({'lowest_avg_val_loss': lowest_loss, 'saved_model_at_epoch': epoch+1 }) 
            if not args.debug:
                torch.save(model.state_dict(), model_save_path) 
    tracker.finish() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--lr', type=float, default=0.0001)  
    # parser.add_argument('--compute_val_freq', type=int, default=5 ) 
    parser.add_argument('--max_epochs', type=int, default=1_000_000_000 ) 
    parser.add_argument('--dim_feedforward', type=int, default=16 )   
    parser.add_argument('--nhead', type=int, default=4 )   
    parser.add_argument('--num_layers', type=int, default=2 )   
    parser.add_argument('--embedding_dim', type=int, default=16 )   
    parser.add_argument('--enc_dropout', type=float, default=0.2 ) 
    parser.add_argument('--extra_dropout', type=float, default=0.2 ) 
    parser.add_argument('--bsz', type=int, default=128 ) 
    parser.add_argument('--debug', type=bool, default=False ) 
    parser.add_argument('--wandb_entity', default="nmaus" )
    parser.add_argument('--wandb_project_name', default="train-binding-affinity-model" )  
    parser.add_argument('--load_ckpt_wandb_name', default="" ) 
    parser.add_argument('--attention', type=bool, default=False ) 
    args = parser.parse_args() 
    # conda activate lolbo_mols
    # tmux attach -t dockmodel
    # CUDA_VISIBLE_DEVICES=0 python3 NN_scoring.py --lr 0.0001 --attention True   
    train(args) 