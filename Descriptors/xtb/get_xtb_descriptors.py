import os
import shutil
import numpy as np
import re
import pandas as pd
import rdkit 
import csv
from rdkit import Chem 
from rdkit.Chem import AllChem
import argparse

def get_xtb_desc(path_to_xtb_desc):
    
    for root, dirs, files in os.walk(path_to_xtb_desc):
        folder_name = os.path.basename(root)
    
        if folder_name.startswith("molecola_"):
            df_alpha = df_fukui = df_charges = None
    
            for file in files:
                file_path = os.path.join(root, file)
    
                if "charges" in file:
                    charges = []
                    with open(file_path) as f:
                        for line in f:
                            charges.append(line.strip())
                    df_charges = pd.DataFrame({"Mulliken Charges": charges})
    
                elif file.endswith(".out"):
                    fukui_output, alpha_output = [], []
                    in_fukui = in_alpha = False
    
                    with open(file_path, encoding="utf-8") as f:
                        for line in f:
                            # Fukui
                            if line.strip() == "Fukui functions:":
                                in_fukui = True
                                fukui_output.append(line)
                                continue
                            elif line.strip() == "-------------------------------------------------":
                                in_fukui = False
                                fukui_output.append(line)
                                continue
                            elif in_fukui:
                                fukui_output.append(line)
    
                            # Alpha
                            if line.strip() == "#   Z          covCN         q      C6AA      α(0)":
                                in_alpha = True
                                alpha_output.append(line)
                                continue
                            elif "Mol. C6AA /au·bohr" in line.strip():
                                in_alpha = False
                                alpha_output.append(line)
                                continue
                            elif in_alpha:
                                alpha_output.append(line)
    
                    fukui_output = [l for l in fukui_output if '-------------------------------------------------' not in l.strip()]
                    alpha_output = [l for l in alpha_output if 'Mol. C6AA /au·bohr' not in l.strip()]
    
                    fukui_file = file_path + '_FF.csv'
                    with open(fukui_file, 'w', newline='') as out_f:
                        out_f.writelines(fukui_output)
    
                    alpha_file = file_path + '_alpha-pol.csv'
                    with open(alpha_file, 'w', newline='', encoding='utf-8') as out_f:
                        out_f.writelines(alpha_output)
    
                    df_fukui = pd.read_csv(fukui_file, delim_whitespace=True, skiprows=2,
                                        names=["#", "f(+)", "f(-)", "f(0)"])
                    df_alpha = pd.read_csv(alpha_file, delim_whitespace=True, skiprows=1,
                                        names=["#", "Z", "atoms", "covCN", "q", "C6AA", "α(0)"])
    
            if df_charges is not None:
                df_charges.to_csv(os.path.join(root, f"{folder_name}_charges.csv"), sep=";", index=False)
    
            if df_alpha is not None and df_fukui is not None and df_charges is not None:
                df_new = pd.DataFrame({
                    "covCN": df_alpha["covCN"],
                    "C6AA": df_alpha["C6AA"],
                    "α(0)": df_alpha["α(0)"],
                    "Fukui positive": df_fukui["f(+)"],
                    "Fukui negative": df_fukui["f(-)"],
                    "Fukui Radical": df_fukui["f(0)"],
                    "Mulliken Charges": df_charges["Mulliken Charges"]
                })
                output_path = os.path.join(root, f"{folder_name}_merged.csv")
                df_new.to_csv(output_path, sep=";", index=False)


if __name__ == "__main__":
    #path_to_xtb_desc = r'path to xtb desc'    
    #get_xtb_desc(path_to_xtb_desc)
    
    parser = argparse.ArgumentParser(description="Process XTB descriptors and merge CSV outputs")
    parser.add_argument("--path_to_xtb_desc",required=True,help="Folder containing the XTB descriptor output folders")
    
    args = parser.parse_args()
    path_to_xtb_desc = args.path_to_xtb_desc
    
    get_xtb_desc(path_to_xtb_desc)