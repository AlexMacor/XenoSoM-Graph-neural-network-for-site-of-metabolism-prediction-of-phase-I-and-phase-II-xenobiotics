import os
import numpy as np
import re
import pandas as pd
import rdkit 
from rdkit import Chem 
from rdkit.Chem import AllChem
import subprocess 
import argparse 

def set_sdf_for_calc(base_path: str) -> set:
    sdf_files = set()
    for dirpath, _, filenames in os.walk(base_path):
        for filename in filenames:
            if filename.endswith(".sdf"):
                sdf_files.add(filename)
    return sdf_files

def preprocessing_sdf(sdf_m, multi_sdfs, cln=False):
    """
    Split a multi-molecule SDF file into multiple .sdf files.
    If clean=True, it also corrects the files by adding the missing lines after ‘M V30 END BOND’.
    """
    os.makedirs(multi_sdfs, exist_ok=True)

    with open(sdf_m, "r", encoding="latin1") as f:
        mol, mol_id = [], 1
        for line in f:
            mol.append(line)
            if line.strip() == "$$$$":  
                out_file = os.path.join(multi_sdfs, f"molecola_{mol_id}.sdf")
                with open(out_file, "w", encoding="utf-8") as out:
                    out.writelines(mol)
                mol, mol_id = [], mol_id + 1

    if cln:
        sdf_files = [f for f in os.listdir(multi_sdfs) if f.endswith('.sdf')]
        for sdf_file in sdf_files:
            file_path = os.path.join(multi_sdfs, sdf_file)
            with open(file_path, 'r', encoding='latin1') as file:
                lines = file.readlines()

            c_lines = []
            for line in lines:
                c_lines.append(line)
                if line.strip() == "M  V30 END BOND":
                    c_lines += ["M  V30 END CTAB\n", "M  END\n", "$$$$\n"]
                    break

            with open(file_path, 'w', encoding='latin1') as file:
                file.writelines(c_lines)

    valid_molecules = set_sdf_for_calc(base_path)
    valid_ids = set(int(f.replace("molecola_", "").replace(".sdf", "")) for f in valid_molecules)
    
    sdf_files = [f for f in os.listdir(multi_sdfs) if f.endswith('.sdf')]
    
    for sdf_file in sdf_files:
        mol_id = int(sdf_file.replace("molecola_", "").replace(".sdf", ""))
        if mol_id not in valid_ids:  
            print(f"Rimosso: {sdf_file}")
            os.remove(os.path.join(multi_sdfs, sdf_file))

   
if __name__ == "__main__":
    
    #base_path = r"C:\Users\user name\Desktop\1\Dataset"
    #sdf_m = r"\SubstratesSDF.sdf" # path with the multi sdf file 
    #multi_sdfs = r"all sdf files split" 
    
    parser = argparse.ArgumentParser(description="Split multi-SDF in single sdfs")
    parser.add_argument("--base_path", required=True, help="base path")
    parser.add_argument("--sdf_m", required=True, help="Multi-mol sdf")
    parser.add_argument("--multi_sdfs", required=True, help="folder to save multiple sdfs")
    
    args = parser.parse_args()

    base_path = args.base_path
    sdf_m = args.sdf_m
    multi_sdfs = args.multi_sdfs

    # path to the empty folder to get all the sdf files after the applied set in order to not optimize weird compounds

    set_sdf_for_calc(base_path)
    preprocessing_sdf(sdf_m, multi_sdfs, cln=False)
