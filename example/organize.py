import glob 
import pandas as pd
import os 

def collect_lightdock_best_pdbs(
    file_prefix,
    k,
    save_dir,
):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    antigen_files = glob.glob(file_prefix + '*') 
    for file in antigen_files:
        antigen_id = file[-4:]
        idxz = 0
        ab_files = glob.glob(file + "/*") 
        for ab_file in ab_files: 
            if 'nanonet' in file_prefix:
                antibody_id = file[-16:-12]
                print('nanonet ab id,', antibody_id)
                antibody_files = ['irrelevant']
            else:
                antibody_files = glob.glob(ab_file + "/*_antibody.pdb")
            if len(antibody_files) == 0:
                antibody_files = glob.glob(ab_file + "/*_A.pdb")
            if len(antibody_files) == 0:
                pass # if still fails, must be empty dir due to failed process or newly starting process
            else:
                if 'nanonet' not in file_prefix:
                    antibody_id = antibody_files[0].split("_")[-2][-4:]
                results_files = glob.glob(ab_file + "/*/RESULTS.csv")
                assert len(results_files) <= 1
                if len(results_files) == 0:
                    pass # if no reslts fille yet, run must be ongoing 
                else:
                    results = pd.read_csv(results_files[0])
                    write_dir = save_dir + f"ag{antigen_id}_ab{antibody_id}"
                    if 'nanonet' in file_prefix:
                        write_dir = write_dir + f"_{idxz}" 
                        idxz += 1 
                    if not os.path.exists(write_dir):
                        os.mkdir(write_dir)
                    for i in range(k):
                        swarm = results['swarm'][i]
                        pdb = results['pdb'][i]
                        score = results['score'][i]
                        path_to_pdb = results_files[0][0:-len("RESULTS.csv")] + f'swarm_{swarm}/' +  pdb
                        # print("pdb path: ", path_to_pdb)
                        os.system(f"cp {path_to_pdb} {write_dir}/combined_structure{i}_score{score}.pdb")    


if __name__ == "__main__": 
    collect_lightdock_best_pdbs(
        file_prefix='lightdock_bothrestrained_nanonet_nruns100_ab',
        k=20,
        save_dir='bothrestrained_nanonet_nruns100_lightdock_poses_allpairs/'
    )
