import torch
from torch import Tensor
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, one_hot
from typing import Optional
from torch import Tensor
from typing import Any, Dict, List
from rdkit import Chem 
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import pandas as pd
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from typing import Any, Dict, List
import torch_geometric
import os
import pandas as pd
from rdkit import Chem 
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch_geometric.utils import from_rdmol
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import f1_score, confusion_matrix, precision_score, f1_score
from torch_geometric.loader import DataLoader
import random
import numpy as np
from typing import Any,List
from rdkit import Chem
from rdkit.Chem import Mol, MolFromMolFile
from rdkit.Chem.rdchem import Atom, Bond
from sklearn.preprocessing import StandardScaler
from models_pytorch import GINModel_cdk, GCNModel, GATC_cdk, train, weighted_binary_crossentropy, test, accuracy, GINModel
from torch.optim import Adam
from collections import defaultdict
import argparse 

parser = argparse.ArgumentParser(
    description="Inference dealkylation")

parser.add_argument('--root_path', type=str, required=True, help="base path with subclasses")

parser.add_argument('--file_sdf_inf_p', type=str, required=True, help="path to sdf to make prediction")

args = parser.parse_args()
root_path = args.root_path
file_sdf_inf_p = args.file_sdf_inf_p

ELEM_LIST = [6,7,8,9,14,15,16,17,35,53]  # B, C, N, O, F, Al, Si, P, S, Cl,Fe,Ge, As,Se, Br, I, Pt, Au #
CHIRALITY = ['CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW']
DEGREE = [1,2,3,4]
HYBRIDIZATION = ['SP','SP2','SP3']

BOND_TYPE_STR = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
STEREO = ['STEREONONE','STEREOZ','STEREOE']  

def _one_hot_encode(value: Any, valid_set: list[Any]) -> List[float]:
    """One-hot encodes `value` based on `valid_set`, with the last element as default."""
    if value not in valid_set:
        value = valid_set[-1]
    return [float(value == element) for element in valid_set]

def generate_node_features(atom: Atom) -> List[float]:
    """Generates node (atom) features for a given atom."""
    return (
        _one_hot_encode(atom.GetAtomicNum(), ELEM_LIST) +
        _one_hot_encode(str(atom.GetChiralTag()), CHIRALITY) +
        _one_hot_encode(atom.GetTotalDegree(), DEGREE) +
        [float(atom.GetFormalCharge()), float(atom.GetTotalNumHs()), 
         float(atom.GetNumRadicalElectrons())] +
        _one_hot_encode(str(atom.GetHybridization()), HYBRIDIZATION) +
        [float(atom.GetIsAromatic()), float(atom.IsInRing())]
    )

def generate_bond_features(bond: Bond) -> List[float]:
    """Generates bond (edge) features for a given bond."""
    return ( _one_hot_encode(str(bond.GetBondType()), BOND_TYPE_STR) + 
            _one_hot_encode(str(bond.GetStereo()), STEREO)) + [
        float(bond.GetIsConjugated()),
    ]

# Convert the Mol object into pytorch geometric data 


def from_rdmol_one_hot(mol: Chem.Mol) -> Data:
    """Converts an RDKit molecule (Mol) to a PyTorch Geometric Data instance
    using custom one-hot encoding functions.

    Args:
        mol (rdkit.Chem.Mol): The RDKit molecule.

    Returns:
        Data: PyTorch Geometric Data object.
    """
    assert isinstance(mol, Chem.Mol)

    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(generate_node_features(atom))
    
    x = torch.tensor(node_features, dtype=torch.float)  

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_feature = generate_bond_features(bond)

        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.extend([bond_feature, bond_feature])

    edge_index = torch.tensor(edge_indices, dtype=torch.long).T  # Matrice 2xN
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    if edge_index.numel() > 0:
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    

seed = 23  # Usa sempre lo stesso valore per consistenza
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# 
# -----------------------------

root_path = r'\Dealkylation'
subclass_folders = [
    d for d in os.listdir(root_path)
    if os.path.isdir(os.path.join(root_path, d)) and d not in ['y-som', 'Risultati-sottoclassi']
]


# -----------------------------
# 
# -----------------------------

file_sdf_inf_p = r'\file_sdf-1'

file_sdf_inf = [f for f in os.listdir(file_sdf_inf_p) if f.endswith('.sdf')]

nomi_all_dataset = []
mol_rdkit = []

for file in file_sdf_inf:
    file_path = os.path.join(file_sdf_inf_p, file)
    nome_molecola = file[:-4]
    nomi_all_dataset.append(nome_molecola)
    m = Chem.MolFromMolFile(file_path)
    m = Chem.RemoveAllHs(m)
    for atom in m.GetAtoms():
        atom.SetNoImplicit(True)  
    m.UpdatePropertyCache()
    mol_rdkit.append(m)  

data_list = []
    
for mol, nome_molecola in zip(mol_rdkit, nomi_all_dataset):

    g = from_rdmol_one_hot(mol)
    
    data = Data(
        x=g.x,
        edge_index=g.edge_index,
        edge_attr=g.edge_attr,
    )
    data.name = nome_molecola  
    data_list.append(data)
       
batch_size = 1
loader_inf = DataLoader(data_list, batch_size=batch_size, shuffle=False)

for data in data_list:
    node_dim = data.x.size(1)
    edge_dim = data.edge_attr.size(1)
    break

        
all_preds_list = []  
for subclass in subclass_folders:
    print(f"\n Processing subclass: {subclass}")
    
    path_subclass = os.path.join(root_path, subclass)
    file_sdf = os.listdir(path_subclass)
    n_file_sdf_in_folder = len([f for f in file_sdf if f.endswith('.sdf')])
    
    if n_file_sdf_in_folder <= 50:
        print(f"  Skipping '{subclass}' (only {n_file_sdf_in_folder} molecole)")
        continue
        
    save_dir = os.path.join(root_path, 'Risultati-sottoclassi', subclass, 'base')
    os.makedirs(save_dir, exist_ok=True)
       
    args_common = dict(in_channels=node_dim, hidden_channels=64, num_layers=3,dropout=0.5)
    args_gatc = dict(**args_common, edge_dim=edge_dim, out_channels=1)
    args_gin = dict(**args_common, out_channels=1)
    
    # Funzione di caricamento
    def load_model(cls, path, device, **kwargs):
        model = cls(**kwargs).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
    
    # Carica modelli salvati
    gin = load_model(GINModel, os.path.join(save_dir, "GINModel_final_test_model.pt"), device, **args_gin)
        
    @torch.no_grad()
    def predict_atom_probs(model, loader):
        model.eval()
        out_list = []
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).cpu().numpy()
            mol_name = batch.name
            for atom_idx, prob in enumerate(out):
                out_list.append({
                    "subclass": subclass,
                    "molecola": mol_name,
                    "indice_atomo": atom_idx + 1,
                    "probabilita": prob
                })
        return pd.DataFrame(out_list)
    
    preds_df_gin = predict_atom_probs(gin, loader_inf)
    all_preds_list.append(preds_df_gin)
    
    
final_df = pd.concat(all_preds_list, ignore_index=True)    
final_csv_path = os.path.join(file_sdf_inf_p, "Dealkylation_subclasses_rdkit_inf.csv")
final_df.to_csv(final_csv_path, index=False)

print(f"Final prediction saved in: {final_csv_path}")   
    