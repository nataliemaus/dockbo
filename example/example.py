import sys 
sys.path.append("../")
from dockbo.dockbo import DockBO
import time 

if __name__ == "__main__":
    work_dir = sys.argv[1]
    # work_dir = '/home/nmaus/'

    oracle = DockBO(
        work_dir=work_dir,
        verbose_config_opt=False, # print progress during BO 
        max_n_bo_steps=20, # num bo iterations, increase to get better scores 
        bsz=1, # bo bsz (using thompson sampling)
        path_to_default_antigen_pdb=work_dir + 'dockbo/example/pdbs/6obd.pdb',
        verbose_timing=True, # print out times it takes to do stuff 
    )

    # example specifying config_x
    print("Passing in dummy config x of all zeros")
    test_config = [0]*27
    start = time.time() 
    score = oracle(
        path_to_antigen_pdb=None, # no antigen path specified --> use default (aviods recomputing default structure)
        antibody_aa_seq="QVQLQESGPGLVRPSQTLSLTCTVSGFTFTDFYMNWVRQPPGRGLEWIGFIRDKAKGYTTEYNPSVKGRVTMLVDTSKNQFSLRLSSVTAADTAVYYCAREGHTAAPFDYWGQGSLVTVSS",
        path_to_antibody_pdb=None, # option to pass in antibody pdb file directly, otherwise we fold seq
        config_x=test_config, # ay len(27) iterable 
    )
    print(f'Score={score}\n')

    # example instead optimizing config_x w/ TuRBO 
    print("Using TuRBO to optimize config")
    score = oracle(
        path_to_antigen_pdb=None, # no antigen path specified --> use default (aviods recomputing default structure)
        antibody_aa_seq="QVQLQESGPGLVRPSQTLSLTCTVSGFTFTDFYMNWVRQPPGRGLEWIGFIRDKAKGYTTEYNPSVKGRVTMLVDTSKNQFSLRLSSVTAADTAVYYCAREGHTAAPFDYWGQGSLVTVSS",
        path_to_antibody_pdb=None, # option to pass in antibody pdb file direction, otherwise we fold seq
        config_x=None,
    )
    print(f'Score={score}\n')

    # example specifying config_x
    print("Directly passing in best config x found by TuRBO on previous run")
    test_config = oracle.previous_best_config
    score = oracle(
        path_to_antigen_pdb=None, # no antigen path specified --> use default (aviods recomputing default structure)
        antibody_aa_seq="QVQLQESGPGLVRPSQTLSLTCTVSGFTFTDFYMNWVRQPPGRGLEWIGFIRDKAKGYTTEYNPSVKGRVTMLVDTSKNQFSLRLSSVTAADTAVYYCAREGHTAAPFDYWGQGSLVTVSS",
        path_to_antibody_pdb=None, # option to pass in antibody pdb file directly, otherwise we fold seq
        config_x=test_config, # ay len(27) iterable 
    )
    print(f'Score={score}\n')

    # example passing in pdb for antibody directoy instead of aa seq 
    #   (saves time by avoiding folding)
    print(f'Using TuRBO to optimize config_x, but with different antibody (from pdb)')
    score = oracle(
        path_to_antigen_pdb=None, # no antigen path specified --> use default (aviods recomputing default structure)
        antibody_aa_seq=None, # no need to pass in an aa seq if we instead give a .pdb file 
        path_to_antibody_pdb=work_dir + 'dockbo/example/pdbs/example_antibody.pdb', 
        config_x=None,
    )
    print(f'Score={score}\n')

    import pdb 
    pdb.set_trace() 

    # Expected Output:  (can vary due to BO randomnes)
    # w/ 20 BO steps: 
    # Passing in aa seq, Time for oracle call:41.79986333847046, score:305244.75
    # Passing in antibody pdb, Time for oracle call:39.276257276535034, score:2180718.75


# Timing breakdown before: 
# time to prep ligand = antigen: 47.57 
# time to prep receptor = antibody: 5.17 
# time to get adapter: 28.66
# time to get indicies: 0.11
# time to get coords: 0.000225
# time to get optimize config: 5.28 (TuRBO)
# __________________________________________________________
# Timing breakdown after w/ precomputing of antigen/ligand stuff: 
# time to prep ligand = 0 (down form 47.7 s)
# time prep receptor = 4.50 
# time get adapter = 3.60   (down from 28 s ) !! 
# time get indicies = 0.00223  (down from 0.11)
# time get coords = 0.00036 
# TOTAL TIME PREP = 8.10 s 


# __________________________________________________________
# __________________________________________________________
# __________________________________________________________
# Time to complete oracle init: 73.74

# Passing in dummy config x of all zeros
# time to fold w/ nanonet: 12.65 
# prep ligthdock time: 4.90
# compute score time: 6.12
# time for full oracle call: 23.66
# Score=2182445.5

# Using TuRBO to optimize config
# time to fold w/ nanonet: 10.776928901672363
# prep ligthdock time: 2.450546979904175
# compute score time: 11.220432996749878
# time for full oracle call: 24.45
# Score=328775.9375

# Directly passing in best config x found by TuRBO on previous run
# time to fold w/ nanonet: 11.057170152664185
# prep ligthdock time: 3.4004077911376953
# compute score time: 0.0509
# time for full oracle call: 14.51
# Score=132.73

# Using TuRBO to optimize config_x, but with different antibody (from pdb)
# prep ligthdock time: 3.7935421466827393
# compute score time: 15.263193368911743
# time for full oracle call: 19.06535768508911
# Score=1551769.88 



# __________________________________________________________
# TIMING SUMMARY: 
# time to fold w/ nanonet: 10-25 s 
# prep ligthdock time: 3-9 s 
# compute score time (with bad given x): 10-12 s 
# compute score time (with good given x): 0.05-1 s 
# compute score time (with TuRBO): 11-16 s (depending on N steps, this assumes 20 steps)
# time for full oracle call: 15-42 s
