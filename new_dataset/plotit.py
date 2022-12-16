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


def plot_score_score(score1, score2):
    filenm = f"{score1}_scores.csv"
    df1 = pd.read_csv(filenm)
    seq_ids_computed1 = df1['seq_id'].values.tolist() 
    scores1 = df1[f'{score1}_score'].values 
    filenm = f"{score2}_scores.csv"
    df2 = pd.read_csv(filenm)
    seq_ids_computed2 = df2['seq_id'].values.tolist() 
    scores2 = df2[f'{score2}_score'].values 
    plot_1 = []
    plot_2 = []
    for ix1, seq_id in enumerate(seq_ids_computed1):
        if seq_id in seq_ids_computed2:
            s1 = scores1[ix1]
            s2 = scores2[seq_ids_computed2.index(seq_id)]
            if score1 == "dfire":
                condition1 = s1 < 200 
            elif score1 == "dfire2":
                condition1 = s1 > 500  
            else: 
                condition1 = True
            if score2 == "dfire":
                condition2 = s2 < 200 
            elif score2 == "dfire2":
                condition2 = s2 > 500  
            else:
                condition2 = True
            if condition2 and condition1:
                plot_1.append(s1)
                plot_2.append(s2) 
    save_path = f"plots/{score1}_vs_{score2}_default_pose.png"
    plt.scatter(plot_2, plot_1)
    plt.title(F"{score1} vs. {score2} scores")
    plt.xlabel(f"{score2}")
    plt.ylabel(f"{score1}")
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

    if args.score_f in ["dfire", "dfire2"]:
        if args.score_f == "dfire":
            bool1b = scores1 < 200 
            bool2b = scores2 < 200 
        elif args.score_f == "dfire2":
            bool1b = scores1 > 500
            bool2b = scores2 > 500

        corr1 = corr1[bool1b]
        scores1 = scores1[bool1b]
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
    parser.add_argument('--score_f', default='cpydock' ) 
    parser.add_argument('--score_f2', default='' ) # specify another score to do score-score plot
    parser.add_argument('--optimized', type=bool, default=False ) 
    args = parser.parse_args() 

    if args.score_f2:
        plot_score_score(score1=args.score_f, score2=args.score_f2)
    else:
        plotit(args) 
    # CUDA_VISIBLE_DEVICES=1 python plotit.py --score_f2 dfire

