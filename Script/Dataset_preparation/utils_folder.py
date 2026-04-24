import os
import numpy as np
import re
import pandas as pd
import rdkit 
from rdkit import Chem 
from rdkit.Chem import AllChem
import argparse

parser = argparse.ArgumentParser(description="Generate folders data structure.")
parser.add_argument("--csv", required=True, help="Path to input csv file.")
parser.add_argument("--out", required=True, help="Output folder path.")
args = parser.parse_args()

df_all_molecules_db = pd.read_csv(args.csv)
output_path = args.out

rxn_classes = df_all_molecules_db['RXNClass'].unique() 

os.makedirs(output_path, exist_ok=True)

for rxn in rxn_classes:
    folder_name = str(rxn).replace("/", "_").replace("\\", "_").replace(" ", "_")
    folder_path = os.path.join(output_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)


def sanitize(name):
    #name = name.strip()
    name = name.replace("+", "_plus_")
    name = name.replace(":", "_")      
    name = name.replace("/", "_")
    name = name.replace("\\", "_")
    name = name.replace("*", "_")
    name = name.replace("?", "_")
    name = name.replace('"', "_")
    name = name.replace("<", "_")
    name = name.replace(">", "_")
    name = name.replace("|", "_")
    name = name.replace(" ", "_")
    name = re.sub(r'[^A-Za-z0-9_\-]', '_', name) 
    return name

for _, row in df_all_molecules_db.iterrows():
    rxn_class = sanitize(str(row['RXNClass']))
    rxn_name = sanitize(str(row['RXNName']))

    class_path = os.path.join(output_path, rxn_class)
    inner_class_path = os.path.join(class_path, rxn_class)
    os.makedirs(inner_class_path, exist_ok=True)
    
    name_path = os.path.join(class_path, rxn_name)

    os.makedirs(name_path, exist_ok=True)

