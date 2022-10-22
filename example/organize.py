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
        # print(antigen_id)
        ab_files = glob.glob(file + "/*")
        for ab_file in ab_files:
            antibody_files = glob.glob(ab_file + "/*_antibody.pdb")
            if len(antibody_files) == 0:
                antibody_files = glob.glob(ab_file + "/*_A.pdb")
            if len(antibody_files) == 0:
                pass # if still fails, must be empty dir due to failed process or newly starting process
            else:
                antibody_id = antibody_files[0].split("_")[-2][-4:]
                results_files = glob.glob(ab_file + "/*/RESULTS.csv")
                assert len(results_files) <= 1
                if len(results_files) == 0:
                    pass # if no reslts fille yet, run must be ongoing 
                else:
                    results = pd.read_csv(results_files[0])
                    write_dir = save_dir + f"ag{antigen_id}_ab{antibody_id}"
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
        file_prefix='lightdock_bothrestrainedv3_',
        k=20,
        save_dir='bothrestrainedv3_lightdock_poses/'
    )
