import os
import numpy as np
import re
import pandas as pd
import rdkit 
from rdkit import Chem 
from rdkit.Chem import AllChem
import subprocess 

def get_atm_bnd_one_hot(mols_path):
    file_sdf = os.listdir(mols_path)
    
    # atomic properties
    chirality_set = set()
    degree_set = set()
    hybridization_set = set()
    atomic_number_set = set()  
    
    # bond properties
    bond_type_set = set()
    stereochem_set = set()

    
    for subdir, _, files in os.walk(mols_path):
        for file in files:
            if file.endswith('.sdf'):
                file_path = os.path.join(subdir, file)
                m = Chem.MolFromMolFile(file_path)
                
                m = Chem.RemoveAllHs(m)
                for atom in m.GetAtoms():
                    atom.SetNoImplicit(True)  
                m.UpdatePropertyCache()                
                    
                for atom in m.GetAtoms():
                    chirality_set.add(str(atom.GetChiralTag()))
                    degree_set.add(str(atom.GetTotalDegree()))
                    hybridization_set.add(str(atom.GetHybridization()))
                    atomic_number_set.add(atom.GetAtomicNum())
    
                for bond in m.GetBonds():
                    bond_type_set.add(str(bond.GetBondType()))
                    stereochem_set.add(str(bond.GetStereo()))
    
    # atomic properties
    CHIRALITY = sorted(list(chirality_set), reverse=True)              
    DEGREE = sorted([int(d) for d in degree_set])                   
    HYBRIDIZATION = sorted(list(hybridization_set))      
    ELEM_LIST = sorted(list(atomic_number_set))     

    # bond properties 
    bond_order_ref = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]    
    BOND_TYPE_STR = [b for b in bond_order_ref if b in bond_type_set]
    STEREO = list(stereochem_set)  

    return {"CHIRALITY": CHIRALITY, "DEGREE": DEGREE, "HYBRIDIZATION": HYBRIDIZATION,
            "ELEM_LIST": ELEM_LIST, "BOND_TYPE_STR": BOND_TYPE_STR, "STEREO": STEREO}


def update_feature_lists(py_file, features):
    with open(py_file, "r", encoding="utf-8") as f:
        content = f.read()

    patterns = {
        "ELEM_LIST": features["ELEM_LIST"],
        "CHIRALITY": features["CHIRALITY"],
        "DEGREE": features["DEGREE"],
        "HYBRIDIZATION": features["HYBRIDIZATION"],
        "BOND_TYPE_STR": features["BOND_TYPE_STR"],
        "STEREO": features["STEREO"],
    }

    for var, new_list in patterns.items():
        regex = rf"{var}\s*=\s*\[.*?\]"
        replacement = f"{var} = {new_list}"
        content = re.sub(regex, replacement, content, flags=re.S)

    with open(py_file, "w", encoding="utf-8") as f:
        f.write(content)

   
if __name__ == "__main__":
    # il path \Database-completo-AM\Dataset-no-H-modelli va bene 
    # poi le lineee seguenti entrano e procedono le classi
    # script di cui modificare dinamicamente il one hot devono essere nella cartella 
    # precedente a quella dei file 
    mols_path = r"path containing the sdf files"
  
    for sub in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, sub)
        if os.path.isdir(subfolder_path):
            features = get_atm_bnd_one_hot(subfolder_path)
            py_file = os.path.join(base_folder, f"{sub}.py")
    
            if os.path.isfile(py_file):
                update_feature_lists(py_file, features)
            else:
                print(f"Script {py_file} non trovato, salto.")