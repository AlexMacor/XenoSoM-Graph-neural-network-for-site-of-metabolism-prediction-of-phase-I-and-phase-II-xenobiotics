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
from rdkit.Chem import rdmolops
from rdkit.Chem.rdmolops import FastFindRings
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
from torch_geometric.nn import GATConv, GCNConv, GINConv
import random
import numpy as np
from typing import Any,List
from rdkit import Chem
from rdkit.Chem import Mol, MolFromMolFile
from rdkit.Chem.rdchem import Atom, Bond
from sklearn.preprocessing import StandardScaler
from models_pytorch import DimeNetPP, train, weighted_binary_crossentropy, test, accuracy
from torch.optim import Adam
from collections import defaultdict
from models_pytorch import set_class_frequencies
from torch_geometric.utils import to_networkx, from_networkx
import argparse

parser = argparse.ArgumentParser(description="Training with cdk and xtb descriptors ")
parser.add_argument('--root_path', type=str, required=True,help="main path")
parser.add_argument('--cdk_path', type=str, required=True,help="Path of cdk descriptors")
parser.add_argument('--xtb_path', type=str, required=True,help="Path of descriptors xtb")

args = parser.parse_args()
root_path = args.root_path
path_csv_descr_cdk = args.cdk_path
path_csv_descr_xtb = args.xtb_path

use_cdk = True
use_xtb = True

def get_xtb_descriptors(path_file_sdf, path_file_csv_descr):
    
    xtb_desc = []
    file_sdf = os.listdir(path_file_sdf)
    nomi_file_sdf = []
    
    for file in file_sdf:
        if file.endswith('.sdf'):
            file_path = os.path.join(path_file_sdf, file)
            nome_molecola_sdf = file[:-4]
            nomi_file_sdf.append(nome_molecola_sdf)           
            
    for root, dirs, files in os.walk(path_file_csv_descr):
        folder_name = os.path.basename(root)  
        if folder_name.startswith("molecola_"):
            for file in files:
                file_path = os.path.join(root, file)

                if file.endswith('_merged.csv'):
                    merged_file = file_path
                    nome_file_merged = file[:-11] 
                elif file.endswith('_alpha.csv'):
                    alpha_file = file_path
                    
            # controllo sui nomi 
            for nome_sdf in nomi_file_sdf:
                if nome_sdf == nome_file_merged:
                    
                    if merged_file and alpha_file:
                        df_merged = pd.read_csv(merged_file, sep=';', encoding='latin1')
                        df_alpha = pd.read_csv(alpha_file, sep=';', encoding='latin1')
                        if 'atoms' in df_alpha.columns:
                            if len(df_merged) == len(df_alpha):  
                                df_merged['atoms'] = df_alpha['atoms']
                            else:
                                print(f"Attention: {folder_name} has a different number of rows between _merged.csv ({len(df_merged)}) and _alpha.csv ({len(df_alpha)})")
                        #df_merged = df_merged[df_merged['atoms'] != 'H'].reset_index(drop=True) # questa se voglio eliminare gli idrogeni usando xtb. 
                        df_merged = df_merged.drop(columns=['atoms'])
                    
                        features_xtb = torch.from_numpy(df_merged.values.astype(np.float32))  
                        xtb_desc.append(features_xtb)

    return xtb_desc 
    
def get_cdk_descriptors(path_file_sdf, path_file_csv_descr):

    cdk_desc = []
    file_sdf = os.listdir(path_file_sdf)
    nomi_file_sdf = []
    
    for file in file_sdf:
        if file.endswith('.sdf'):
            file_path = os.path.join(path_file_sdf, file)
            nome_molecola_sdf = file[:-4]
            nomi_file_sdf.append(nome_molecola_sdf)
    
    for root, dirs, files in os.walk(path_file_csv_descr):
        folder_name = os.path.basename(root)  
        if folder_name.startswith("molecola_"):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.csv'):
                    nome_molecola_csv = file[:-8]
                    for nome_sdf in nomi_file_sdf:
                        if nome_sdf == nome_molecola_csv:
                            #print(f"Match: {nome_sdf} ↔ {nome_molecola_csv}")
                            df = pd.read_csv(file_path, sep=',', encoding='latin1')
                            cdk_descr = df.iloc[:, 4:].drop(columns=[
                                'StabilizationPlusCharge', 'AtomHybridizationVSEPR',
                                'AtomHybridization', 'AtomValance', 'AtomDegree'
                            ])
                            cdk_descr = torch.from_numpy(cdk_descr.values.astype(np.float32))
                            cdk_desc.append(cdk_descr)
    return cdk_desc
    
def from_rdmol_dimenet(mol: Chem.Mol, node_features: Optional[torch.Tensor] = None) -> Data:
    assert mol.GetNumConformers() > 0, "The molecule must have 3D coordinates."

    z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
    conf = mol.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float)

    if node_features is not None:
        assert node_features.shape[0] == mol.GetNumAtoms(), \
            f"Le feature ({node_features.shape[0]}) non corrispondono al numero di atomi ({mol.GetNumAtoms()})"
        data = Data(z=z, pos=pos, x=node_features)
    else:
        data = Data(z=z, pos=pos)

    return data

    
seed = 23  
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

csv_dir = os.path.join(root_path, 'y-som')

subclass_folders = [
    d for d in os.listdir(root_path)
    if os.path.isdir(os.path.join(root_path, d)) and d not in ['y-som', 'Risultati-sottoclassi','Risultati-sottoclassi_HT']
]

for subclass in subclass_folders:
    print(f"\n Processing subclass: {subclass}")

    path = os.path.join(root_path, subclass)
    file_sdf = os.listdir(path)
    n_file_sdf_in_folder = len([f for f in file_sdf if f.endswith('.sdf')])
    
    if n_file_sdf_in_folder <= 50:
        print(f"  Skipping '{subclass}' (only {n_file_sdf_in_folder} molecole)")
        continue
        
    ## calcolo dei valori per rdkit diversity picker 
    value_1 = round(n_file_sdf_in_folder * 0.2)
    if value_1 % 2 != 0:
        value_1 -= 1

    value_2 = int(value_1 / 2)
    value_3 = int(value_1 / 2)    
    
    path_y = os.path.join(csv_dir, f'y_som_{subclass}_match.csv')

    file_csv = pd.read_csv(path_y, sep=',')

    file_csv['molecole'] = file_csv['molecole'].astype(str).apply(lambda x: f"molecola_{x}")
    valori_match = file_csv.groupby('molecole')
    
    tensori_y = []
    for i in valori_match:
        tensore_y = torch.tensor(i[1]['match'].to_numpy())
        tensori_y.append(tensore_y)
    
    nomi_all_dataset = []
    mol_rdkit = []

    for file in file_sdf:
        if file.endswith('.sdf'):
            file_path = os.path.join(path, file)
            nome_molecola = file[:-4]
            nomi_all_dataset.append(nome_molecola)
            m = Chem.MolFromMolFile(file_path, removeHs=False, sanitize=False)
            #m = Chem.AddHs(m, addCoords=True)
            #m = Chem.RemoveAllHs(m)
            #for atom in m.GetAtoms():
            #    atom.SetNoImplicit(True)  
            #m.UpdatePropertyCache() 
            mol_rdkit.append(m)  
        
    cdk_desc_list = get_cdk_descriptors(path, path_csv_descr_subclass)
    xtb_desc_list = get_xtb_descriptors(path, path_csv_descr_subclass_xtb)    
    
    data_list = []
    
    for mol, nome_molecola, features_cdk, features_xtb in zip(mol_rdkit, nomi_all_dataset, cdk_desc_list, xtb_desc_list):
        g = from_rdmol_dimenet(mol)
        g.cdk_desc = features_cdk
        g.xtb_desc = features_xtb         
        g.name = nome_molecola
        data_list.append(g)
    
    #assert i.z.shape[0] == len(j), f"❌ Mismatch in {i.name}: {i.z.shape[0]} nodi, {len(j)} etichette"

    for i, j in zip(data_list, tensori_y):
        assert i.z.shape[0] == len(j), f"❌ Mismatch in {i.name}: {i.z.shape[0]} nodi, {len(j)} etichette"
        i.y = j
    
    # Fingerprint
    Mfpts, mol_names, molecules_h = [], [], []
    for file in file_sdf:
        if file.endswith('.sdf'):
            file_path = os.path.join(path, file)
            m = Chem.MolFromMolFile(file_path, removeHs=False, sanitize=False)
            #m_h = Chem.AddHs(m, addCoords=True)
            #if m is None: continue
            m_h = Chem.RemoveAllHs(m, sanitize=False)
            for atom in m_h.GetAtoms():
                atom.SetNoImplicit(True)  
            m_h.UpdatePropertyCache(strict=False)
            rdmolops.FastFindRings(m_h)
            molecules_h.append(m_h)
            mgen = AllChem.GetMorganGenerator()
            Morganfg = mgen.GetFingerprint(m_h)
            Mfpts.append(Morganfg)
            mol_names.append(file[:-4])

    # Split
    df_e = pd.DataFrame({'nomi molecole':mol_names, 'oggetto mol':molecules_h, 'morgan fingerprint': Mfpts,})
    
     
    val_test_picker = MaxMinPicker()
    nMfpts = len(Mfpts)  
    val_test_pickIndices = val_test_picker.LazyBitVectorPick(Mfpts,nMfpts,value_1,seed=23)  
    list(val_test_pickIndices)  
    
    # for test val split 
    molname_val_test_picks = [mol_names[x] for x in val_test_pickIndices] 
    molecules_h_val_test_picks = [molecules_h[x] for x in val_test_pickIndices] 
    mol_val_test_picks = [Mfpts[x] for x in val_test_pickIndices] 
    df_val_test = pd.DataFrame({'nomi molecole':molname_val_test_picks, 'oggetto mol':molecules_h_val_test_picks, 'morgan fingerprint': mol_val_test_picks})
    
    # validation splitting 
    val_picker = MaxMinPicker()
    nmol_val_test_picks = len(mol_val_test_picks)  
    val_pickIndices = val_picker.LazyBitVectorPick(mol_val_test_picks,nmol_val_test_picks,value_2,seed=23)  
    list(val_pickIndices) 
    
    molname_val_picks = [molname_val_test_picks[x] for x in val_pickIndices] 
    mol_val_picks = [mol_val_test_picks[x] for x in val_pickIndices] 
    molecules_h_val_picks = [molecules_h_val_test_picks[x] for x in val_pickIndices] 
    
    df_validation = pd.DataFrame({'nomi molecole':molname_val_picks, 'oggetto mol':molecules_h_val_picks, 'morgan fingerprint': mol_val_picks})
    
    # test splitting
     
    test_indices = [i for i in range(len(mol_val_test_picks)) if i not in val_pickIndices]  
    test_picker = MaxMinPicker()
    nmol_test_picks = len(test_indices)
    test_pickIndices = test_picker.LazyBitVectorPick([mol_val_test_picks[i] for i in test_indices], nmol_test_picks,value_3,seed=23)
    
    molname_test_picks = [molname_val_test_picks[test_indices[x]] for x in test_pickIndices]
    mol_test_picks = [mol_val_test_picks[test_indices[x]] for x in test_pickIndices]
    molecules_h_test_picks = [molecules_h_val_test_picks[test_indices[x]] for x in test_pickIndices]
    
    df_test = pd.DataFrame({'nomi molecole': molname_test_picks,'oggetto mol': molecules_h_test_picks,'morgan fingerprint': mol_test_picks})
    
    # training set 
    train_indices = [i for i in range(len(mol_names)) if i not in val_test_pickIndices]
    
    mol_name_train = [mol_names[i] for i in train_indices]
    mol_train = [Mfpts[i] for i in train_indices]
    molecules_h_train = [molecules_h[i] for i in train_indices]
    
    df_train = pd.DataFrame({'nomi molecole': mol_name_train,'oggetto mol': molecules_h_train,'morgan fingerprint': mol_train})

    train_graph = []
    for data in data_list:
        if data.name in mol_name_train:
            train_graph.append(data)
    
    test_graph = []
    for data in data_list:
        if data.name in molname_test_picks:
            test_graph.append(data)
    
    val_graph = []
    for data in data_list:
        if data.name in molname_val_picks:
            val_graph.append(data)
    
    from sklearn.preprocessing import StandardScaler

    scaler_cdk = StandardScaler()
    scaler_xtb = StandardScaler()
    
    train_features_cdk = [data.cdk_desc for data in train_graph] 
    train_features_cdk_all = torch.cat(train_features_cdk, dim=0) 

    train_features_xtb = [data.xtb_desc for data in train_graph] 
    train_features_xtb_all = torch.cat(train_features_xtb, dim=0)  
    
    scaler_cdk.fit(train_features_cdk_all.numpy())
    scaler_xtb.fit(train_features_xtb_all.numpy())    
    
    for data in train_graph:
        data.cdk_desc = torch.from_numpy(scaler_cdk.transform(data.cdk_desc.numpy())).float()
        data.xtb_desc = torch.from_numpy(scaler_xtb.transform(data.xtb_desc.numpy())).float()
    
    for data in val_graph:
        data.cdk_desc = torch.from_numpy(scaler_cdk.transform(data.cdk_desc.numpy())).float()
        data.xtb_desc = torch.from_numpy(scaler_xtb.transform(data.xtb_desc.numpy())).float()
    
    for data in test_graph:
        data.cdk_desc = torch.from_numpy(scaler_cdk.transform(data.cdk_desc.numpy())).float()
        data.xtb_desc = torch.from_numpy(scaler_xtb.transform(data.xtb_desc.numpy())).float()
     
    if n_file_sdf_in_folder >= 500:
        batch_size = 32
        
    elif 100 <= n_file_sdf_in_folder < 500:
        batch_size = 16
        
    else:
        batch_size = 8 
        
    device = torch.device('cpu')
    train_loader = DataLoader(train_graph, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graph, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graph, batch_size=batch_size, shuffle=False)
    
    def calculate_the_frequencies_0_1():
        mask_train = file_csv['molecole'].isin(mol_name_train)
        colonna_match_train = file_csv.loc[mask_train, 'match']
        n_0 = colonna_match_train.value_counts().get(0, 0)
        n_1 = colonna_match_train.value_counts().get(1, 0)
        return n_0, n_1
    
    n_0, n_1 = calculate_the_frequencies_0_1()
    set_class_frequencies(n_0, n_1)
    
    cdk_dim = 9
    xtb_dim = 7
    
    models_to_test = {
        "DimeNetPP": DimeNetPP(
            hidden_channels=64,
            out_channels=1,
            num_blocks=3,
            int_emb_size=32,
            basis_emb_size=32,
            out_emb_channels=64,
            num_spherical=4,
            num_radial=6,
            cutoff=5.0,
            envelope_exponent=5,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3,
            act='swish',
            output_initializer='zeros',
            cdk_dim=cdk_dim, 
            xtb_dim=xtb_dim
        )
    }

    best_model_name = None
    best_val_balanced_acc = 0
    best_model = None

    save_dir = os.path.join(root_path, 'Risultati-sottoclassi', subclass, 'cdk_xtb')
    os.makedirs(save_dir, exist_ok=True)

    for model_name, model in models_to_test.items():
        print(f"\n Training and testing for the model: {model_name}")

        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.01)

        for epoch in range(100):
            trained_model, train_loss, train_acc, train_sensitivity, train_specificity, train_balanced_acc, train_precision, train_fbeta, train_auprc, train_mcc = train(model, train_loader, optimizer)
            val_loss, val_acc, val_sensitivity, val_specificity, val_balanced_acc, val_precision, val_fbeta, val_auprc, val_mcc = test(trained_model, val_loader)

            if val_balanced_acc > best_val_balanced_acc:
                best_val_balanced_acc = val_balanced_acc
                best_model = trained_model
                best_model_name = model_name

            print(f"Epoch {epoch:>3} | Model: {model_name} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                f"Train Balanced Acc: {train_balanced_acc:.4f} | Val Balanced Acc: {val_balanced_acc:.4f}")

        # Test
        test_loss, test_acc, test_sensitivity, test_specificity, test_balanced_acc, test_precision, test_fbeta, test_auprc, test_mcc = test(model, test_loader)

        model_path = os.path.join(save_dir, f"{model_name}_final_test_model.pt")
        torch.save(model.state_dict(), model_path)

        metrics_file = os.path.join(save_dir, f"{model_name}_final_metrics.csv")
        final_metrics = {
            "train_loss": train_loss, "train_acc": train_acc,
            "train_sensitivity": train_sensitivity, "train_specificity": train_specificity,
            "train_balanced_acc": train_balanced_acc, "train_precision": train_precision,
            "train_fbeta": train_fbeta, "train_auprc": train_auprc, "train_mcc": train_mcc,
            "val_loss": val_loss, "val_acc": val_acc,
            "val_sensitivity": val_sensitivity, "val_specificity": val_specificity,
            "val_balanced_acc": val_balanced_acc, "val_precision": val_precision,
            "val_fbeta": val_fbeta, "val_auprc": val_auprc, "val_mcc": val_mcc,
            "test_loss": test_loss, "test_acc": test_acc,
            "test_sensitivity": test_sensitivity, "test_specificity": test_specificity,
            "test_balanced_acc": test_balanced_acc, "test_precision": test_precision,
            "test_fbeta": test_fbeta, "test_auprc": test_auprc, "test_mcc": test_mcc
        }

        pd.DataFrame.from_dict([final_metrics]).to_csv(metrics_file, index=False)
        print(f"\n[*] {model_name} salvato in: {model_path}")
        print(f"[*] Metrics saved  in: {metrics_file}")