import numpy as np 
import torch 
import matplotlib.pyplot as plt 
import pandas as pd 

def load_data():
    # RANIBIZUMAB 
    # http://opig.stats.ox.ac.uk/webapps/newsabdab/therasabdab/therasummary/?INN=Ranibizumab
    df = pd.read_csv('rani_display_with_sequences.csv')
    seq_ids = df['Unnamed: 0'].squeeze().values # (96846,)
    h_chains = df['full_sequence'].values # (96846,) 
    l_chain = "DIQLTQSPSSLSASVGDRVTITCSASQDISNYLNWYQQKPGKAPKVLIYFTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSTVPWTFGQGTKVEIK"
    corr1 = df['log(R3/R2) replicate 1'].values # (96846,)
    corr2 = df['log(R3/R2) replicate 2'].values # (96846,)
    return seq_ids, corr1, corr2 


def plot_dfire_dfire2():
    filenm = f"dfire_scores.csv"
    df1 = pd.read_csv(filenm)
    seq_ids_computed1 = df1['seq_id'].values.tolist() 
    scores1 = df1[f'dfire_score'].values 
    filenm = f"dfire2_scores.csv"
    df2 = pd.read_csv(filenm)
    seq_ids_computed2 = df2['seq_id'].values.tolist() 
    scores2 = df2[f'dfire2_score'].values 

    dfire = []
    dfire2 = []
    for ix1, seq_id in enumerate(seq_ids_computed1):
        if seq_id in seq_ids_computed2:
            dfire_s = scores1[ix1]
            dfire2_s = scores2[seq_ids_computed2.index(seq_id)]
            if dfire_s < 200 and dfire2_s > 500:
                dfire.append(dfire_s)
                dfire2.append(dfire2_s) 
    
    # corr1 
    save_path = f"plots/dfire_vs_dfire2_default_pose.png"
    plt.scatter(dfire2, dfire)
    plt.title(F"dfire vs. dfire2 scores")
    plt.xlabel(f"dfire2")
    plt.ylabel(f"dfire")
    plt.savefig(save_path) 
    plt.clf() 







def plotit(args):
    seq_ids, corr1_, corr2_ = load_data() 
    filenm = f"{args.score_f}_scores.csv"
    if args.optimized:
        filenm = "optimized_" + filenm 
    df = pd.read_csv(filenm)
    seq_ids_computed = df['seq_id'].values
    scores = df[f'{args.score_f}_score'].values 
    assert seq_ids_computed.shape == scores.shape 
    if seq_ids_computed[0] == -1:
        seq_ids_computed = seq_ids_computed[1:]
        scores = scores[1:]
    corr1 = corr1_[seq_ids_computed]
    corr2 = corr2_[seq_ids_computed]
    bool1 = np.logical_not(np.isnan(corr1))
    bool2 = np.logical_not(np.isnan(corr2))
    corr1 = corr1[bool1]
    corr2 = corr2[bool2]
    scores1 = scores[bool1]
    scores2 = scores[bool2]

    if args.score_f == "dfire":
        bool1b = scores1 < 200 
        bool2b = scores2 < 200 
    elif args.score_f == "dfire2":
        bool1b = scores1 > 500
        bool2b = scores2 > 500

    corr1 = corr1[bool1b]
    scores1 = scores1[bool1b]
    bool2b = scores2 < 200 
    corr2 = corr2[bool2b]
    scores2 = scores2[bool2b]

    # corr1 
    save_path = args.save_dir + f"corr1_vs_{args.score_f}_default_pose.png"
    if args.optimized:
        save_path = args.save_dir + f"corr1_vs_{args.score_f}_optimized_pose.png"
    plt.scatter(scores1, corr1)
    plt.title(F"log(R3/R2) replicate 1 vs. {args.score_f} Scores")
    plt.xlabel(f"{args.score_f} score")
    plt.ylabel(f"log(R3/R2) replicate 1")
    plt.savefig(save_path) 
    plt.clf() 

    # corr2 
    save_path = args.save_dir + f"corr2_vs_{args.score_f}_default_pose.png"
    if args.optimized:
        save_path = args.save_dir + f"corr2_vs_{args.score_f}_optimized_pose.png"
    plt.scatter(scores2, corr2)
    plt.xlabel(f"{args.score_f} score")
    plt.ylabel(f"log(R3/R2) replicate 2")
    plt.title(F"log(R3/R2) replicate 2 vs. {args.score_f} Scores")
    plt.savefig(save_path) 
    plt.clf() 

if __name__ == "__main__": 
    import argparse 
    parser = argparse.ArgumentParser()  
    parser.add_argument('--save_dir', default="plots/" ) # "dfire, cpydock, dfire2"
    parser.add_argument('--work_dir', default='/home/nmaus/' ) 
    parser.add_argument('--score_f', default='dfire' ) 
    parser.add_argument('--optimized', type=bool, default=False ) 
    parser.add_argument('--dfire_dfire2', type=bool, default=True)
    args = parser.parse_args() 

    if args.dfire_dfire2:
        plot_dfire_dfire2()
    else:
        plotit(args) 

