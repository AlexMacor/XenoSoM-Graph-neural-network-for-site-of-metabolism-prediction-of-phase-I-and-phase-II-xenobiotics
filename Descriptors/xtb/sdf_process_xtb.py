import os
import shutil
import numpy as np
import re
import subprocess
import pandas as pd
import rdkit 
from rdkit import Chem 
from rdkit.Chem import AllChem
       
def conformational_analysis(multi_sdfs: str, input_cdk_xtb: str, num_of_conformer: int = 10):

    if not os.path.exists(input_cdk_xtb):
        os.makedirs(input_cdk_xtb)

    mols = []
    mols_h = []
    mol_name = []

    for filename in os.listdir(multi_sdfs):
        if filename.endswith('.sdf'):
            file_path = os.path.join(multi_sdfs, filename)
            mol = Chem.MolFromMolFile(file_path)
            if mol:
                mols.append(mol)
                mol_name.append(os.path.splitext(filename)[0])
                
    for m in mols:
        if m is not None:
            m_h = Chem.rdmolops.AddHs(m, addCoords=True)
            mols_h.append(m_h)

    params = AllChem.ETKDG()
    params.randomSeed = 2
    params.numThreads = 8
    params.useRandomCoords = True

    molecole_cids = []
    for idx, mol in enumerate(mols_h):
        if mol is not None:
            try:
                mol_cid = AllChem.EmbedMultipleConfs(mol,numConfs=num_of_conformer,params=params)
                molecole_cids.append(list(mol_cid))
  
            except Exception as e:
                print(f"Molecola {mol_name[idx]}: errore durante l'Embedding - {e}")
                molecole_cids.append([])  
                
    results_MMFF = []
    for idx, mol in enumerate(mols_h):
        if mol is not None:
            try:
                mol_opt_MMMFF = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=1000)
                results_MMFF.append(list(mol_opt_MMMFF))
                #print(f"Molecola {mol_name[idx]}: Ottimizzazione completata.")
            
            except ValueError as e:
                print(f"Molecola {mol_name[idx]}: Errore durante l'ottimizzazione - {e}")
                results_MMFF.append([])

    for mol_index, (mol, mmff_results) in enumerate(zip(mols_h, results_MMFF)):
        min_energy_index_MMFF = min(
            (conf_id for conf_id, (status, energy) in enumerate(mmff_results) if status == 0),
            key=lambda conf_id: mmff_results[conf_id][1],
            default=None
        )

        if min_energy_index_MMFF is not None:
            original_filename = mol_name[mol_index]
            output_file = os.path.join(input_cdk_xtb, f"{original_filename}.sdf")

            with Chem.SDWriter(output_file) as writer:
                writer.write(Chem.Mol(mol, False, min_energy_index_MMFF))

            # print(f"Molecola {original_filename} salvata in: {output_file}")


## prepare the script to run both the cdk and xtb consider both the idrogen and no idrogen 

def get_mol_charges(input_cdk_xtb: str, charges_csv: str, sh_file_path: str) -> dict:
    
    charges_dict = {}
    for filename in os.listdir(input_cdk_xtb):
        if filename.endswith('.sdf'):
            file_path = os.path.join(input_cdk_xtb, filename)
            mol_h = Chem.MolFromMolFile(file_path, sanitize=False, removeHs=False)
            
            if mol_h is not None:
                charge = Chem.GetFormalCharge(mol_h)
                mol_name = os.path.splitext(filename)[0]
                charges_dict[mol_name] = charge
                
    os.makedirs(os.path.dirname(charges_csv), exist_ok=True)
    df_charges = pd.DataFrame(list(charges_dict.items()), columns=['Molecule', 'Charges'])
    
    df_all_charges = df_charges  
        
    with open(sh_file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    inserted = False

    for line in lines:
        new_lines.append(line)
        if not inserted and line.strip() == "declare -A charges":
            for _, row in df_all_charges.iterrows():
                mol_name = row['Molecule']
                charge = row['Charges']
                new_lines.append(f'charges["{mol_name}"]={charge}\n')
            inserted = True

    with open(sh_file_path, 'w') as f:
        f.writelines(new_lines)

    print(f"File {sh_file_path} aggiornato con le cariche.")
            
    return charges_dict
    
# copiamo tutti i file da analisi conformazionale 

def cp_ac_xtb(input_cdk_xtb: str, in_path: str):

    os.makedirs(in_path, exist_ok=True)

    for f in os.listdir(input_cdk_xtb):
        if f.endswith(".sdf"):  
            src = os.path.join(input_cdk_xtb, f)
            dst = os.path.join(in_path, f)
            shutil.copy(src, dst)
            print(f"Cp sdf file: {f} → {in_path}")

def get_num_file(in_path: str):
    all_sdfs = []
    for f in os.listdir(in_path):
        if f.endswith(".sdf"):
            all_sdfs.append(f)
    return len(all_sdfs)
            

## funzione to copy the ouptut of xtb 

# dopo ottimizzazione di geometria con xtb
# copiamo le strutture in un altra cartella per poter poi lanciare il calcolo single point 

def cp_xtb_opt(temp_path: str, in_path: str):

    os.makedirs(in_path, exist_ok=True)
    for dir_path, dir_name, filen in os.walk(temp_path):
        for file in filen:
            if file.endswith("_xtbopt.sdf"):
                src = os.path.join(dir_path, file)
                new_name = file.replace("_xtbopt", "")
                dst = os.path.join(in_path, new_name)
    
                shutil.copy(src, dst)
                print(f"Cp sdf file: {file} → {dst}")


if __name__ == "__main__":
    multi_sdfs = "/mnt/c/Users/User name/Desktop/Database-test-reazione-equilibrioum/opt/sdf"
    input_cdk_xtb = "/mnt/c/Users/User name/Desktop/Database-test-reazione-equilibrioum/opt/AC-output"
    charges_csv = "/mnt/c/Users/User name/Desktop/Database-test-reazione-equilibrioum/opt/charges.csv"
    in_path = "/mnt/c/Users/User name/Desktop/Database-test-reazione-equilibrioum/opt/xtb/1"
    temp_path = "/mnt/c/Users/User name/Desktop/Database-test-reazione-equilibrioum/opt/xtb/2"
    fin_path = "/mnt/c/Users/User name/Desktop/Database-test-reazione-equilibrioum/opt/xtb/3"
    #opt_sdf_files_xtb_path = "/mnt/c/Users/Alessio Macorano/Desktop/Database-test-reazione-equilibrioum/opt/xtb/OPT"
    sh_file_path = "/mnt/c/Users/User name/Desktop/Database-test-reazione-equilibrioum/opt/xtb/chg-xtb-run-opt.sh"
    SP_bash_Script = "/mnt/c/Users/User name/Desktop/Database-test-reazione-equilibrioum/opt/xtb/chg-xtb-run-SP.sh"

    conformational_analysis(multi_sdfs, input_cdk_xtb, num_of_conformer=10)
    charges = get_mol_charges(input_cdk_xtb, charges_csv, sh_file_path)
    
    cp_ac_xtb(input_cdk_xtb, in_path) 
    num_files = get_num_file(in_path)
    
    # Run the xtb optimization
    optimization_xtb = subprocess.run(["python", "Run_xtb_conv.py", 
        "--n_folders", str(num_files)], capture_output=True, text=True, check=True)
  
    output_xtb = optimization_xtb.stdout
    
    if ("ok" in output_xtb and "Alcuni file sdf devono essere riottimizzati: 0" in output_xtb):
        print("Prossimo step")
    else:
        print("Qualcosa non va, non procedo.")
    
    cp_xtb_opt(temp_path, in_path)
    
    charges = get_mol_charges(in_path, charges_csv, SP_bash_Script)

    shutil.rmtree(temp_path); os.makedirs(temp_path, exist_ok=True)

    # Run the SP optimization
    subprocess.run(["bash", SP_bash_Script], check=True)
    

