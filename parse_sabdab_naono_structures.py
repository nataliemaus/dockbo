import glob 
from prody import *

data_dir = "all_sabdab_nano_structures/chothia/*.pdb"
all_files = glob.glob("sabdab_nano_chothia/*.pdb")
save_dir = "parsed_chothia_pdbs"

for file in all_files:
    pdb_id = file.split("/")[-1][0:4] 
    f = open(file, "r")
    data = f.read() 
    data = data.split('\n')
    prev_idx = 0
    pair_idx = 0
    for dat in data:
        if not dat.startswith('REMARK'):
            break 
        try: 
            dat = dat.split()
            if ("HCHAIN=" in dat[3]) and ("AGCHAIN=" in dat[4]) and ("NONE" not in dat[3]) and ("NONE" not in dat[4]) :
                pair_idx += 1 # often multiple pairs in single pdb file 
                hchain = dat[3][-1]
                agchain = dat[4][-1]
                hchain_struct = parsePDB(file, chain=hchain)
                writePDB(f"{save_dir}/{pdb_id}_ex{pair_idx}_ab", hchain_struct)  
                agchain_struct = parsePDB(file, chain=agchain)
                writePDB(f"{save_dir}/{pdb_id}_ex{pair_idx}_ag", agchain_struct) 
        except:
            print(f"Failure on PDB: {pdb_id}")
            pass  
