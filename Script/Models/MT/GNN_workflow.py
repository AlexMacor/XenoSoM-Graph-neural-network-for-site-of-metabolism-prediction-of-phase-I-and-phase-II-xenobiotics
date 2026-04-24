# --- Standard Library ---
import os
import random
import numpy as np
import copy
from typing import Any, List, Dict, Optional
from collections import defaultdict

# --- PyTorch & PyG Core ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
import torch_geometric.transforms as T
from torch_geometric.utils import degree, one_hot
from torch_geometric.nn import (GATConv, GCNConv, GINConv, GATv2Conv)
from torch_geometric.nn.conv import GINEConv

# --- RDKit (Cheminformatics) ---
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Mol, MolFromMolFile
from rdkit.Chem.rdchem import Atom, Bond
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

# --- Data Handling & Metrics ---
import pandas as pd
from sklearn.metrics import (f1_score, confusion_matrix, precision_score,
                              average_precision_score, fbeta_score, matthews_corrcoef)

# --- Custom Modules ---
from models import (GINModel, GCNModel, GATC, GINEModel, AttentiveFPNode,
                    GATv2Model, get_activation, train, test, accuracy,
                    weighted_binary_crossentropy, set_class_frequencies)

from dataclasses import dataclass
from typing import Union, Tuple
from torch_geometric.nn.models.attentive_fp import GATEConv

import optuna
from optuna.samplers import TPESampler


@dataclass
class FeatureConfig:
    elem_list: list
    chirality: list
    degree: list
    hybridization: list
    bond_types: list
    stereo: list

    def __post_init__(self):
        self.ELEM_LIST = self.elem_list
        self.CHIRALITY = self.chirality
        self.DEGREE = self.degree
        self.HYBRIDIZATION = self.hybridization
        self.BOND_TYPE_STR = self.bond_types
        self.STEREO = self.stereo


# =========================
# GLOBAL CONFIG HOLDER
# =========================

CFG = None


def set_cfg(cfg):
    global CFG
    CFG = cfg


# =========================
# FEATURE ENCODING
# =========================

def _one_hot_encode(value: Any, valid_set: list) -> List[float]:
    if value not in valid_set:
        value = valid_set[-1]
    return [float(value == element) for element in valid_set]


def generate_node_features(atom: Atom) -> List[float]:
    return (
        _one_hot_encode(atom.GetAtomicNum(), CFG.ELEM_LIST) +
        _one_hot_encode(str(atom.GetChiralTag()), CFG.CHIRALITY) +
        _one_hot_encode(atom.GetTotalDegree(), CFG.DEGREE) +
        [
            float(atom.GetFormalCharge()),
            float(atom.GetTotalNumHs()),
            float(atom.GetNumRadicalElectrons())
        ] +
        _one_hot_encode(str(atom.GetHybridization()), CFG.HYBRIDIZATION) +
        [
            float(atom.GetIsAromatic()),
            float(atom.IsInRing())
        ]
    )


def generate_bond_features(bond: Bond) -> List[float]:
    return (
        _one_hot_encode(str(bond.GetBondType()), CFG.BOND_TYPE_STR) +
        _one_hot_encode(str(bond.GetStereo()), CFG.STEREO)
    ) + [
        float(bond.GetIsConjugated())
    ]


# =========================
# GRAPH CONVERSION
# =========================

def from_rdmol_one_hot(mol: Chem.Mol) -> Data:
    assert isinstance(mol, Chem.Mol)

    node_features = [generate_node_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_features, dtype=torch.float)

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feature = generate_bond_features(bond)
        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.extend([bond_feature, bond_feature])

    edge_index = torch.tensor(edge_indices, dtype=torch.long).T
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    if edge_index.numel() > 0:
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# =========================
# DATA LOADING
# =========================

def load_molecules(path):
    mols, names = [], []
    for f in os.listdir(path):
        if f.endswith('.sdf'):
            m = Chem.MolFromMolFile(os.path.join(path, f))
            m = Chem.RemoveAllHs(m)
            for a in m.GetAtoms():
                a.SetNoImplicit(True)
            m.UpdatePropertyCache()
            mols.append(m)
            names.append(f[:-4])
    return mols, names


def load_labels(csv_path, task_names, add_prefix=True):
    df = pd.read_csv(csv_path)
    if add_prefix:
        df['molecole'] = df['molecole'].astype(str).apply(lambda x: f"molecola_{x}")
    else:
        df['molecole'] = df['molecole'].astype(str)
    return df


# =========================
# GRAPH BUILDING
# =========================

def build_graphs(mols, names, labels_group, task_names):
    data_list = []
    for mol, name in zip(mols, names):
        g = from_rdmol_one_hot(mol)

        mol_rows = labels_group[labels_group['molecole'] == name]

        y = torch.tensor(
            mol_rows[task_names].fillna(0).values,
            dtype=torch.float
        )

        mask = torch.tensor(
            mol_rows[task_names].notna().values.astype(float),
            dtype=torch.float
        )

        data = Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr)
        data.name = name
        data.y    = y
        data.mask = mask
        data_list.append(data)

    for d in data_list:
        assert d.x.shape[0] == d.y.shape[0], \
            f"Mismatch in {d.name}: {d.x.shape[0]} nodi, {d.y.shape[0]} etichette"

    return data_list


# =========================
# SPLIT
# =========================

def fingerprint_split(mols, names, value_1, value_2, value_3, seed=23):
    fps = [AllChem.GetMorganGenerator().GetFingerprint(m) for m in mols]

    val_test_picker = MaxMinPicker()
    val_test_pickIndices = list(
        val_test_picker.LazyBitVectorPick(fps, len(fps), value_1, seed=seed)
    )

    fps_val_test = [fps[i] for i in val_test_pickIndices]
    val_picker = MaxMinPicker()
    val_pickIndices = list(
        val_picker.LazyBitVectorPick(fps_val_test, len(fps_val_test), value_2, seed=seed)
    )

    test_candidates = [i for i in range(len(val_test_pickIndices)) if i not in val_pickIndices]
    fps_test_candidates = [fps_val_test[i] for i in test_candidates]
    test_picker = MaxMinPicker()
    test_pickIndices = list(
        test_picker.LazyBitVectorPick(fps_test_candidates, len(fps_test_candidates), value_3, seed=seed)
    )

    val_names   = [names[val_test_pickIndices[i]] for i in val_pickIndices]
    test_names  = [names[val_test_pickIndices[test_candidates[i]]] for i in test_pickIndices]
    train_names = [names[i] for i in range(len(names)) if i not in val_test_pickIndices]

    return train_names, val_names, test_names


# =========================
# ATOM-LEVEL PREDICTIONS
# =========================

@torch.no_grad()
def predict_atom_probs(model, loader, device, task_names):
    model.eval()
    out_list = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch).cpu().numpy()
        batch_indices = batch.batch.cpu().numpy()
        batch_names = batch.name
        atom_counter = defaultdict(int)
        for idx in range(out.shape[0]):
            mol_idx  = batch_indices[idx]
            mol_name = batch_names[mol_idx] if isinstance(batch_names, list) else batch_names
            atom_idx = atom_counter[mol_name]
            row = {
                "molecola": mol_name,
                "indice_atomo": atom_idx + 1,
            }
            for t, task in enumerate(task_names):
                row[f"prob_{task}"] = float(out[idx, t])
                row[f"pred_{task}"] = 1 if out[idx, t] >= 0.5 else 0
            out_list.append(row)
            atom_counter[mol_name] += 1
    return pd.DataFrame(out_list)


# =========================
# TOP-K COMPUTATION (FIXED: only on mols with positive sites)
# =========================

def compute_topk_correct(df_merged, task_names, k_values):
    """
    Top-k calcolata SOLO su molecole con almeno un sito positivo per quel task.
    """
    results = {}
    for task in task_names:
        mols_positive = (
            df_merged.groupby("molecola")[task]
            .sum()
            .loc[lambda s: s > 0]
            .index
        )
        df_task = df_merged[df_merged["molecola"].isin(mols_positive)]
        n_mols = df_task["molecola"].nunique()

        if n_mols == 0:
            for k in k_values:
                results[f"top{k}_{task}"] = float('nan')
            results[f"n_mols_positive_{task}"] = 0
            continue

        topk_counts = {k: 0 for k in k_values}
        for _, group in df_task.groupby("molecola"):
            group_sorted = group.sort_values(f"prob_{task}", ascending=False)
            for k in k_values:
                if group_sorted.head(k)[task].any():
                    topk_counts[k] += 1

        for k in k_values:
            results[f"top{k}_{task}"] = topk_counts[k] / n_mols
        results[f"n_mols_positive_{task}"] = n_mols

    return results


# =========================
# TUNING PIPELINE (Optuna)
# =========================

def run_tuning(subclass, root_path, csv_dir, n_file, cfg, reaction_name, task_names, n_trials=30):

    save_dir = os.path.join(root_path, 'Risultati-sottoclassi', subclass, 'rdkit')
    os.makedirs(save_dir, exist_ok=True)

    def make_objective(fixed_model_name):
        set_cfg(cfg)
        path     = os.path.join(root_path, subclass)
        csv_path = os.path.join(csv_dir, f'y_som_{subclass}_match.csv')

        prefix_needed = False if subclass == 'Hydrolysis_of_esters' else True
        file_csv = load_labels(csv_path, task_names, add_prefix=prefix_needed)

        mols, names = load_molecules(path)
        data_list = build_graphs(mols, names, file_csv, task_names)

        value_1 = round(n_file * 0.2)
        if value_1 % 2 != 0:
            value_1 -= 1
        value_2 = value_1 // 2
        value_3 = value_1 // 2

        train_names, val_names, _ = fingerprint_split(
            mols, names, value_1, value_2, value_3, seed=23
        )

        def objective(trial):
            seed = 23
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            train_graph = [copy.deepcopy(d) for d in data_list if d.name in train_names]
            val_graph   = [copy.deepcopy(d) for d in data_list if d.name in val_names]

            device = torch.device('cuda')

            # ── HP search space ──
            lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
            hidden_dim   = trial.suggest_categorical("hidden_dim", [32, 64, 96, 128, 256])
            drp          = trial.suggest_float("dropout", 0.1, 0.6)
            n_layers     = trial.suggest_int("n_layers", 2, 5)
            batch_size   = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
            if fixed_model_name == "AttentiveFPNode":
                activation = "relu"
            else:
                activation = trial.suggest_categorical("activation", ["relu", "elu", "leaky_relu"])
            if fixed_model_name in ["GATC", "GATv2Model"]:
                heads = trial.suggest_categorical("heads", [1, 2, 4, 8])
            else:
                heads = 4

            train_loader = DataLoader(train_graph, batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(val_graph,   batch_size=batch_size, shuffle=False)

            node_dim = train_graph[0].x.size(1)
            edge_dim = train_graph[0].edge_attr.size(1)

            train_name_set = {d.name for d in train_graph}
            train_df = file_csv[file_csv['molecole'].isin(train_name_set)]
            frequencies = {}
            for t in task_names:
                n0_t = int(train_df[t].eq(0).sum())
                n1_t = int(train_df[t].eq(1).sum())
                frequencies[t] = (n0_t, n1_t)
            set_class_frequencies(frequencies)

            n_tasks = len(task_names)

            if fixed_model_name == "GINModel":
                model = GINModel(node_dim, hidden_dim, n_layers, drp, n_tasks, activation=activation).to(device)
            elif fixed_model_name == "GNN":
                model = GCNModel(node_dim, hidden_dim, n_layers, drp, n_tasks, activation=activation).to(device)
            elif fixed_model_name == "GATC":
                model = GATC(node_dim, edge_dim, hidden_dim, n_layers, drp, n_tasks, heads=heads, activation=activation).to(device)
            elif fixed_model_name == "GINEModel":
                model = GINEModel(node_dim, edge_dim, hidden_dim, n_layers, drp, n_tasks, activation=activation).to(device)
            elif fixed_model_name == "AttentiveFPNode":
                model = AttentiveFPNode(node_dim, edge_dim, hidden_dim, n_layers, drp, n_tasks, activation=activation).to(device)
            elif fixed_model_name == "GATv2Model":
                model = GATv2Model(node_dim, edge_dim, hidden_dim, n_layers, drp, n_tasks, heads=heads, activation=activation).to(device)

            opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            best_val_balanced_acc = -1.0  # FIX B4

            for epoch in range(30):
                model, train_loss, train_metrics = train(model, train_loader, opt, task_names)
                val_loss, val_metrics = test(model, val_loader, task_names)
                val_balanced_acc = np.mean([val_metrics[t]["balanced_acc"] for t in task_names])

                trial.report(val_balanced_acc, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                if val_balanced_acc > best_val_balanced_acc:
                    best_val_balanced_acc = val_balanced_acc

            return best_val_balanced_acc
        return objective

    all_best_params = {}

    for model_name in ["GINModel", "GNN", "GATC", "GINEModel", "AttentiveFPNode", "GATv2Model"]:
        print(f"\n{'='*60}")
        print(f"Tuning {model_name} per {subclass}")
        print(f"{'='*60}")

        sampler = TPESampler(seed=23)
        pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=f"tuning_{subclass}_{model_name}"
        )

        study.optimize(
            make_objective(model_name),
            n_trials=n_trials,
            show_progress_bar=True
        )

        print(f"\nBEST TRIAL {model_name}: Val Balanced Acc = {study.best_value:.4f}")
        print(f"Iperparametri: {study.best_params}")

        best_params = study.best_params
        best_params["model_name"] = model_name
        all_best_params[model_name] = best_params

        study.trials_dataframe().to_csv(
            os.path.join(save_dir, f"{subclass}_{model_name}_optuna_trials.csv"), index=False
        )
        pd.DataFrame([best_params]).to_csv(
            os.path.join(save_dir, f"{subclass}_{model_name}_best_params.csv"), index=False
        )

    return all_best_params


# =========================
# TRAINING PIPELINE
# =========================

def run_training(subclass, root_path, csv_dir, n_file, cfg, reaction_name, task_names, hyperparams=None):

    save_dir = os.path.join(root_path, 'Risultati-sottoclassi', subclass, 'rdkit')
    os.makedirs(save_dir, exist_ok=True)

    set_cfg(cfg)

    seed = 23
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    path = os.path.join(root_path, subclass)
    csv_path = os.path.join(csv_dir, f'y_som_{subclass}_match.csv')

    prefix_needed = False if subclass == 'Hydrolysis_of_esters' else True
    file_csv = load_labels(csv_path, task_names, add_prefix=prefix_needed)
    mols, names = load_molecules(path)
    data_list = build_graphs(mols, names, file_csv, task_names)

    assert len(data_list) == file_csv['molecole'].nunique(), \
        f"Mismatch grafi/etichette in {subclass}"

    value_1 = round(n_file * 0.2)
    if value_1 % 2 != 0:
        value_1 -= 1
    value_2 = value_1 // 2
    value_3 = value_1 // 2

    train_names, val_names, test_names = fingerprint_split(
        mols, names, value_1, value_2, value_3, seed=seed
    )

    pd.DataFrame(train_names, columns=['Molecole_Train']).to_csv(
        os.path.join(save_dir, f"{subclass}_train.csv"), index=False
    )
    pd.DataFrame(val_names, columns=['Molecole_Val']).to_csv(
        os.path.join(save_dir, f"{subclass}_val.csv"), index=False
    )
    pd.DataFrame(test_names, columns=['Molecole_Test']).to_csv(
        os.path.join(save_dir, f"{subclass}_test.csv"), index=False
    )

    train_graph = [d for d in data_list if d.name in train_names]
    val_graph   = [d for d in data_list if d.name in val_names]
    test_graph  = [d for d in data_list if d.name in test_names]

    # Label del test set per top-k
    file_csv['molecole_key'] = file_csv['molecole']
    df_test_labels = file_csv[file_csv['molecole_key'].isin(test_names)].copy()
    df_test_labels = df_test_labels.rename(columns={'molecole': 'molecola'})
    df_test_labels["indice_atomo"] = df_test_labels.groupby("molecola").cumcount() + 1
    df_test_labels = df_test_labels.drop(columns=['molecole_key'])

    device = torch.device('cuda')

    # Class frequencies on TRAIN split
    train_name_set = {d.name for d in train_graph}
    train_df = file_csv[file_csv['molecole'].isin(train_name_set)]
    frequencies = {}
    for t in task_names:
        n0_t = int(train_df[t].eq(0).sum())
        n1_t = int(train_df[t].eq(1).sum())
        frequencies[t] = (n0_t, n1_t)
    set_class_frequencies(frequencies)

    n_tasks = len(task_names)
    node_dim = train_graph[0].x.size(1)
    edge_dim = train_graph[0].edge_attr.size(1)

    model_names_to_run = ["GINModel", "GNN", "GATC", "GINEModel", "AttentiveFPNode", "GATv2Model"]

    # default hyperparams se non tuned
    if not hyperparams:
        if n_file < 100:
            default_hidden_dim = 32
            default_drp = 0.6
        elif n_file < 500:
            default_hidden_dim = 64
            default_drp = 0.5
        else:
            default_hidden_dim = 64
            default_drp = 0.4
        default_lr           = 0.001
        default_weight_decay = 0.01
        default_n_layers     = 3
        default_activation   = "relu"
        default_heads        = 4
        default_batch_size   = 32 if n_file >= 500 else 16 if n_file >= 100 else 8

    results_for_subclass = []

    for model_name in model_names_to_run:
        # ── hyperparam selection ──
        if hyperparams:
            hp = hyperparams[model_name]
            hidden_dim   = int(hp["hidden_dim"])
            drp          = float(hp["dropout"])
            n_layers     = int(hp["n_layers"])
            lr           = float(hp["lr"])
            weight_decay = float(hp["weight_decay"])
            activation   = hp.get("activation", "relu")
            heads        = int(hp.get("heads", 4)) if not pd.isna(hp.get("heads", np.nan)) else 4
            batch_size   = int(hp["batch_size"])
        else:
            hidden_dim   = default_hidden_dim
            drp          = default_drp
            n_layers     = default_n_layers
            lr           = default_lr
            weight_decay = default_weight_decay
            activation   = default_activation
            heads        = default_heads
            batch_size   = default_batch_size

        # ── model instantiation ──
        if model_name == "GINModel":
            model = GINModel(node_dim, hidden_dim, n_layers, drp, n_tasks, activation=activation).to(device)
        elif model_name == "GNN":
            model = GCNModel(node_dim, hidden_dim, n_layers, drp, n_tasks, activation=activation).to(device)
        elif model_name == "GATC":
            model = GATC(node_dim, edge_dim, hidden_dim, n_layers, drp, n_tasks, heads=heads, activation=activation).to(device)
        elif model_name == "GINEModel":
            model = GINEModel(node_dim, edge_dim, hidden_dim, n_layers, drp, n_tasks, activation=activation).to(device)
        elif model_name == "AttentiveFPNode":
            model = AttentiveFPNode(node_dim, edge_dim, hidden_dim, n_layers, drp, n_tasks, activation=activation).to(device)
        elif model_name == "GATv2Model":
            model = GATv2Model(node_dim, edge_dim, hidden_dim, n_layers, drp, n_tasks, heads=heads, activation=activation).to(device)

        train_loader = DataLoader(train_graph, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_graph,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_graph,  batch_size=batch_size, shuffle=False)

        opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # FIX B4
        best_val_balanced_acc = -1.0
        best_model = copy.deepcopy(model)
        # FIX B1
        best_train_metrics = None
        best_val_metrics = None
        best_train_loss = None
        best_val_loss = None
        best_epoch = -1

        for epoch in range(100):
            model, train_loss, train_metrics = train(model, train_loader, opt, task_names)
            val_loss, val_metrics = test(model, val_loader, task_names)
            val_balanced_acc = np.mean([val_metrics[t]["balanced_acc"] for t in task_names])
            train_balanced_acc = np.mean([train_metrics[t]["balanced_acc"] for t in task_names])

            if val_balanced_acc > best_val_balanced_acc:
                best_val_balanced_acc = val_balanced_acc
                best_model = copy.deepcopy(model)
                # FIX B1: salvo snapshot delle metriche di QUESTA epoca
                best_train_metrics = copy.deepcopy(train_metrics)
                best_val_metrics = copy.deepcopy(val_metrics)
                best_train_loss = train_loss
                best_val_loss = val_loss
                best_epoch = epoch

            print(f"Epoch {epoch:>3} | Model: {model_name} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Balanced Acc: {train_balanced_acc:.4f} | Val Balanced Acc: {val_balanced_acc:.4f}")

        # Test finale
        test_loss, test_metrics = test(best_model, test_loader, task_names)

        # Save model
        model_path = os.path.join(save_dir, f"{model_name}_final_test_model.pt")
        torch.save(best_model.state_dict(), model_path)

        # Predictions + merge con labels
        preds = predict_atom_probs(best_model, test_loader, device, task_names)
        df_merged = pd.merge(df_test_labels, preds, on=["molecola", "indice_atomo"], how="inner")

        pred_file_path = os.path.join(save_dir, f"{model_name}_y_pred.csv")
        df_merged.to_csv(pred_file_path, index=False)

        # FIX B2: Top-k corretta
        k_values = [1, 2, 3]
        top_k_acc = compute_topk_correct(df_merged, task_names, k_values)
        for task in task_names:
            n_pos = top_k_acc[f"n_mols_positive_{task}"]
            for k in k_values:
                v = top_k_acc[f"top{k}_{task}"]
                vs = f"{v:.3f}" if not np.isnan(v) else "  NA"
                print(f"  Top-{k} {task} ({model_name}): {vs} (n_mols_positive={n_pos})")

        metrics_file = os.path.join(save_dir, f"{model_name}_final_metrics.csv")

        hp_used = hyperparams[model_name] if hyperparams else None

        final_metrics = {
            "Model"          : model_name,
            "tuned"          : True if hyperparams else False,
            "best_epoch"     : best_epoch,
            "hp_lr"          : hp_used["lr"]           if hp_used else lr,
            "hp_weight_decay": hp_used["weight_decay"] if hp_used else weight_decay,
            "hp_hidden_dim"  : hp_used["hidden_dim"]   if hp_used else hidden_dim,
            "hp_dropout"     : hp_used["dropout"]      if hp_used else drp,
            "hp_n_layers"    : hp_used["n_layers"]     if hp_used else n_layers,
            "hp_activation"  : hp_used.get("activation", "relu") if hp_used else activation,
            "hp_heads"       : hp_used.get("heads", 4) if hp_used else heads,
            "hp_batch_size"  : hp_used.get("batch_size", batch_size) if hp_used else batch_size,
            "train_loss"     : best_train_loss,
            "val_loss"       : best_val_loss,
            "test_loss"      : test_loss,
        }

        # FIX B1: uso best_train_metrics e best_val_metrics (dell'epoca del best model)
        for task in task_names:
            for split, met in [("train", best_train_metrics),
                               ("val",   best_val_metrics),
                               ("test",  test_metrics)]:
                final_metrics[f"{split}_balanced_acc_{task}"] = met[task]["balanced_acc"]
                final_metrics[f"{split}_sensitivity_{task}"]  = met[task]["sensitivity"]
                final_metrics[f"{split}_specificity_{task}"]  = met[task]["specificity"]
                final_metrics[f"{split}_precision_{task}"]    = met[task]["precision"]
                final_metrics[f"{split}_fbeta_{task}"]        = met[task]["fbeta"]
                final_metrics[f"{split}_auprc_{task}"]        = met[task]["auprc"]
                final_metrics[f"{split}_mcc_{task}"]          = met[task]["mcc"]
                final_metrics[f"{split}_acc_{task}"]          = met[task]["acc"]

        final_metrics.update(top_k_acc)

        # Salvataggio CSV singolo per modello
        pd.DataFrame([final_metrics]).to_csv(metrics_file, index=False)

        results_for_subclass.append(final_metrics)

        print(f"\n[*] {model_name} salvato in: {model_path}")
        print(f"[*] Metriche salvate in: {metrics_file}")

    return results_for_subclass
