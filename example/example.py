import sys 
sys.path.append("../")
from dockbo.dockbo import DockBO
import time 

if __name__ == "__main__":
    work_dir = sys.argv[1]
    # work_dir = '/home/nmaus/'

    oracle = DockBO(
        work_dir=work_dir,
        verbose_config_opt=True, # print progress during BO 
        max_n_bo_steps=5, # num bo iterations, increase to get better scores 
        bsz=2, # bo bsz (using thompson sampling)
    )
    # can pass in either an aa seq or a pdb file for antibody, 
    #   antigen must be a pdb file since nanonet, igfold only work well for antibodies 
    antigen_path = work_dir + 'dockbo/example/pdbs/6obd.pdb'
    
    start = time.time() 
    score = oracle(
        path_to_antigen_pdb=antigen_path,
        antibody_aa_seq=None, 
        path_to_antibody_pdb=work_dir + 'dockbo/example/pdbs/example_antibody.pdb', 
    )
    info_string1 = f"Passing in antibody pdb, Time for oracle call:{time.time() - start}, score:{score}"

    # example passing in an aa sequence instead 
    start = time.time() 
    score = oracle(
        path_to_antigen_pdb=antigen_path,
        antibody_aa_seq="AAAAQQQQQQQQVVVVVVVLLLLLLLLLLLLL", 
        path_to_antibody_pdb=None, # option to pass in antibody pdb file direction, otherwise we fold seq
    )
    info_string2 = f"Passing in antigen pdb, Time for oracle call:{time.time() - start}, score:{score}"

    print(info_string1)
    print(info_string2)

    # Expected Output:  (can vary due to BO randomness)
    # Passing in antibody pdb, Time for oracle call:96.25313186645508, score:1117828.0
    # Passing in antigen pdb, Time for oracle call:99.16347241401672, score:2105975.0
