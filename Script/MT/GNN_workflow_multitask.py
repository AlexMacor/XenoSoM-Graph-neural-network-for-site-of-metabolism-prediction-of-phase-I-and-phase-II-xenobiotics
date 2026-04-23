"""
GNN_workflow_multitask.py
=========================
Workflow di training multi-task ottimizzato.

Differenze rispetto al workflow single-task:
  - Lo split NON viene ricalcolato con MaxMin: viene letto dagli split
    single-task già esistenti e risolto tramite multitask_split_builder.
  - I duplicati cross-task vengono risolti prima del training.
  - La loss è masked e pesata per task.
  - Le metriche sono calcolate separatamente per ogni task.
  - predict_atom_probs usa sigmoid esternamente (modelli restituiscono logits).
"""

import os
import random
import numpy as np
import copy
from collections import defaultdict

import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
import pandas as pd

from models import (
    GINModel, GCNModel, GATC, GINEModel, AttentiveFPNode, GATv2Model,
    train, test, set_class_frequencies
)
from GNN_workflow import (
    set_cfg, FeatureConfig, compute_topk_correct
)
from multitask_split_builder import build_multitask_dataset

import optuna
from optuna.samplers import TPESampler


# =============================================================================
# PREDICT ATOM PROBS
# I modelli restituiscono logits → sigmoid applicata qui per probabilità.
# =============================================================================

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



# =============================================================================
# TUNING PIPELINE (Optuna) — multi-task
# =============================================================================

def run_multitask_tuning(sdf_dir, multitask_csv_path,split_dir, duplicate_csv_path,
    task_names,cfg,save_dir,n_trials=30 ):
    """
    Ottimizzazione iperparametri per il multi-task con Optuna.

    Parameters
    ----------
    sdf_dir : str
        Cartella SDF multi-task (nomi canonici).
    multitask_csv_path : str
        CSV label multi-task.
    split_dir : str
        Cartella con split single-task ({task}_train/val/test.csv).
    duplicate_csv_path : str
        Mappa duplicati (separatore ;).
    task_names : list[str]
    cfg : FeatureConfig
    save_dir : str
        Dove salvare i risultati Optuna.
    n_trials : int

    Returns
    -------
    all_best_params : dict[str, dict]
    """
    os.makedirs(save_dir, exist_ok=True)
    set_cfg(cfg)

    # ── Costruisce il dataset una sola volta ──
    print("\n[Tuning] Costruzione dataset multi-task...")
    train_names, val_names, _, data_list, frequencies = build_multitask_dataset(
        sdf_dir            = sdf_dir,
        multitask_csv_path = multitask_csv_path,
        split_dir          = split_dir,
        duplicate_csv_path = duplicate_csv_path,
        task_names         = task_names,
        cfg                = cfg,
        save_dir           = save_dir
    )
    set_class_frequencies(frequencies)

    train_name_set = set(train_names)
    val_name_set   = set(val_names)
    train_graph    = [d for d in data_list if d.name in train_name_set]
    val_graph      = [d for d in data_list if d.name in val_name_set]

    node_dim = train_graph[0].x.size(1)
    edge_dim = train_graph[0].edge_attr.size(1)
    n_tasks  = len(task_names)
    device   = torch.device('cuda')

    def make_objective(fixed_model_name):
        def objective(trial):
            seed = 23
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

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

            train_loader = DataLoader(
                [copy.deepcopy(d) for d in train_graph],
                batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                [copy.deepcopy(d) for d in val_graph],
                batch_size=batch_size, shuffle=False
            )

            model = _instantiate_model(
                fixed_model_name, node_dim, edge_dim,
                hidden_dim, n_layers, drp, n_tasks, heads, activation, device
            )
            opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            best_val_balanced_acc = -1.0
            for epoch in range(30):
                model, _, train_metrics = train(model, train_loader, opt, task_names)
                _, val_metrics = test(model, val_loader, task_names)
                val_balanced_acc = np.mean([val_metrics[t]["balanced_acc"] for t in task_names])

                trial.report(val_balanced_acc, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                if val_balanced_acc > best_val_balanced_acc:
                    best_val_balanced_acc = val_balanced_acc

            return best_val_balanced_acc
        return objective

    all_best_params = {}
    model_names = ["GINModel", "GNN", "GATC", "GINEModel", "AttentiveFPNode", "GATv2Model"]

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Tuning {model_name} — Multi-Task")
        print(f"{'='*60}")

        study = optuna.create_study(
            direction  = "maximize",
            sampler    = TPESampler(seed=23),
            pruner     = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            study_name = f"multitask_{model_name}"
        )
        study.optimize(make_objective(model_name), n_trials=n_trials, show_progress_bar=True)

        print(f"\nBEST {model_name}: Val Balanced Acc = {study.best_value:.4f}")
        print(f"Params: {study.best_params}")

        best_params = study.best_params
        best_params["model_name"] = model_name
        all_best_params[model_name] = best_params

        study.trials_dataframe().to_csv(
            os.path.join(save_dir, f"multitask_{model_name}_optuna_trials.csv"), index=False
        )
        pd.DataFrame([best_params]).to_csv(
            os.path.join(save_dir, f"multitask_{model_name}_best_params.csv"), index=False
        )

    return all_best_params


# =============================================================================
# TRAINING PIPELINE — multi-task
# =============================================================================

def run_multitask_training(sdf_dir, multitask_csv_path, split_dir,
    duplicate_csv_path,task_names,cfg,save_dir,hyperparams=None):
        
    """
    Training multi-task con split derivato dagli split single-task.

    Parameters
    ----------
    sdf_dir : str
        Cartella SDF multi-task (nomi canonici).
    multitask_csv_path : str
        CSV label multi-task (colonne: molecole, indice_atomi, match_Task1, ...).
    split_dir : str
        Cartella con i CSV di split single-task ({task}_train/val/test.csv).
    duplicate_csv_path : str
        Mappa duplicati cross-task (separatore ;).
    task_names : list[str]
        Lista task nell'ordine delle colonne del CSV label.
    cfg : FeatureConfig
        Configurazione feature condivisa.
    save_dir : str
        Cartella dove salvare modelli, metriche e predizioni.
    hyperparams : dict, optional
        {model_name: {lr, dropout, hidden_dim, ...}} da Optuna.

    Returns
    -------
    results : list[dict]
        Metriche finali per ogni modello.
    """
    os.makedirs(save_dir, exist_ok=True)
    set_cfg(cfg)

    seed = 23
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda')
    print(f"\n[Multi-Task Training] Device: {device}")

    # ── STEP A: Costruisce il dataset multi-task ottimizzato ──
    print("\n" + "="*60)
    print("Costruzione dataset multi-task ottimizzato")
    print("="*60)
    train_names, val_names, test_names, data_list, frequencies = build_multitask_dataset(
        sdf_dir            = sdf_dir,
        multitask_csv_path = multitask_csv_path,
        split_dir          = split_dir,
        duplicate_csv_path = duplicate_csv_path,
        task_names         = task_names,
        cfg                = cfg,
        save_dir           = save_dir    # salva split e report
    )
    set_class_frequencies(frequencies)

    train_name_set = set(train_names)
    val_name_set   = set(val_names)
    test_name_set  = set(test_names)

    train_graph = [d for d in data_list if d.name in train_name_set]
    val_graph   = [d for d in data_list if d.name in val_name_set]
    test_graph  = [d for d in data_list if d.name in test_name_set]

    # ── Label test per top-k ──
    # Ricostruisce df_test_labels dalla y e mask dei grafi di test
    test_label_rows = []
    for d in test_graph:
        for atom_idx in range(d.y.shape[0]):
            row = {"molecola": d.name, "indice_atomo": atom_idx + 1}
            for t, task in enumerate(task_names):
                if d.mask[atom_idx, t].item() == 1:
                    row[task] = int(d.y[atom_idx, t].item())
                else:
                    row[task] = np.nan
            test_label_rows.append(row)
    df_test_labels = pd.DataFrame(test_label_rows)

    node_dim = train_graph[0].x.size(1)
    edge_dim = train_graph[0].edge_attr.size(1)
    n_tasks  = len(task_names)

    # ── Default hyperparams ──
    n_train = len(train_graph)
    if not hyperparams:
        if n_train < 100:
            default_hidden_dim, default_drp = 32, 0.6
        elif n_train < 500:
            default_hidden_dim, default_drp = 64, 0.5
        else:
            default_hidden_dim, default_drp = 64, 0.4
        default_lr           = 0.001
        default_weight_decay = 0.01
        default_n_layers     = 3
        default_activation   = "relu"
        default_heads        = 4
        default_batch_size   = 32 if n_train >= 500 else 16 if n_train >= 100 else 8

    model_names = ["GINModel", "GNN", "GATC", "GINEModel", "AttentiveFPNode", "GATv2Model"]
    results = []

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Modello: {model_name} — Multi-Task")
        print(f"{'='*60}")

        # ── Hyperparams ──
        if hyperparams and model_name in hyperparams:
            hp           = hyperparams[model_name]
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

        model = _instantiate_model(
            model_name, node_dim, edge_dim,
            hidden_dim, n_layers, drp, n_tasks, heads, activation, device
        )

        train_loader = DataLoader(train_graph, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_graph,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_graph,  batch_size=batch_size, shuffle=False)

        opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_balanced_acc = -1.0
        best_model            = copy.deepcopy(model)
        best_train_metrics    = None
        best_val_metrics      = None
        best_train_loss       = None
        best_val_loss         = None
        best_epoch            = -1

        for epoch in range(100):
            model, train_loss, train_metrics = train(model, train_loader, opt, task_names)
            val_loss, val_metrics            = test(model, val_loader, task_names)

            val_balanced_acc   = np.mean([val_metrics[t]["balanced_acc"]   for t in task_names])
            train_balanced_acc = np.mean([train_metrics[t]["balanced_acc"] for t in task_names])

            if val_balanced_acc > best_val_balanced_acc:
                best_val_balanced_acc = val_balanced_acc
                best_model         = copy.deepcopy(model)
                best_train_metrics = copy.deepcopy(train_metrics)
                best_val_metrics   = copy.deepcopy(val_metrics)
                best_train_loss    = train_loss
                best_val_loss      = val_loss
                best_epoch         = epoch

            print(f"Epoch {epoch:>3} | {model_name} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Bal.Acc: {train_balanced_acc:.4f} | "
                  f"Val Bal.Acc: {val_balanced_acc:.4f}")

        # ── Test finale ──
        test_loss, test_metrics = test(best_model, test_loader, task_names)

        model_path = os.path.join(save_dir, f"{model_name}_multitask_model.pt")
        torch.save(best_model.state_dict(), model_path)

        # ── Predizioni sul test set ──
        preds     = predict_atom_probs(best_model, test_loader, device, task_names)
        df_merged = pd.merge(df_test_labels, preds, on=["molecola", "indice_atomo"], how="inner")
        df_merged.to_csv(
            os.path.join(save_dir, f"{model_name}_multitask_y_pred.csv"), index=False
        )

        # ── Top-K ──
        k_values  = [1, 2, 3]
        top_k_acc = compute_topk_correct(df_merged, task_names, k_values)
        for task in task_names:
            n_pos = top_k_acc[f"n_mols_positive_{task}"]
            for k in k_values:
                v  = top_k_acc[f"top{k}_{task}"]
                vs = f"{v:.3f}" if not np.isnan(v) else "  NA"
                print(f"  Top-{k} {task} ({model_name}): {vs}  (n_pos={n_pos})")

        # ── Raccolta metriche ──
        hp_used = hyperparams.get(model_name) if hyperparams else None
        final_metrics = {
            "Model":           model_name,
            "tuned":           bool(hp_used),
            "best_epoch":      best_epoch,
            "hp_lr":           hp_used["lr"]           if hp_used else lr,
            "hp_weight_decay": hp_used["weight_decay"] if hp_used else weight_decay,
            "hp_hidden_dim":   hp_used["hidden_dim"]   if hp_used else hidden_dim,
            "hp_dropout":      hp_used["dropout"]      if hp_used else drp,
            "hp_n_layers":     hp_used["n_layers"]     if hp_used else n_layers,
            "hp_activation":   hp_used.get("activation", "relu") if hp_used else activation,
            "hp_heads":        hp_used.get("heads", 4) if hp_used else heads,
            "hp_batch_size":   hp_used.get("batch_size", batch_size) if hp_used else batch_size,
            "train_loss":      best_train_loss,
            "val_loss":        best_val_loss,
            "test_loss":       test_loss,
            "n_train_mols":    len(train_names),
            "n_val_mols":      len(val_names),
            "n_test_mols":     len(test_names),
        }

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

        metrics_path = os.path.join(save_dir, f"{model_name}_multitask_metrics.csv")
        pd.DataFrame([final_metrics]).to_csv(metrics_path, index=False)
        results.append(final_metrics)

        print(f"\n  Modello salvato: {model_path}")
        print(f"  Metriche salvate: {metrics_path}")

    return results


# =============================================================================
# HELPER — istanziazione modello
# =============================================================================

def _instantiate_model(model_name, node_dim, edge_dim,
                        hidden_dim, n_layers, drp, n_tasks,
                        heads, activation, device):
    if model_name == "GINModel":
        return GINModel(node_dim, hidden_dim, n_layers, drp, n_tasks,
                        activation=activation).to(device)
    elif model_name == "GNN":
        return GCNModel(node_dim, hidden_dim, n_layers, drp, n_tasks,
                        activation=activation).to(device)
    elif model_name == "GATC":
        return GATC(node_dim, edge_dim, hidden_dim, n_layers, drp, n_tasks,
                    heads=heads, activation=activation).to(device)
    elif model_name == "GINEModel":
        return GINEModel(node_dim, edge_dim, hidden_dim, n_layers, drp, n_tasks,
                         activation=activation).to(device)
    elif model_name == "AttentiveFPNode":
        return AttentiveFPNode(node_dim, edge_dim, hidden_dim, n_layers, drp, n_tasks,
                               activation=activation).to(device)
    elif model_name == "GATv2Model":
        return GATv2Model(node_dim, edge_dim, hidden_dim, n_layers, drp, n_tasks,
                          heads=heads, activation=activation).to(device)
    else:
        raise ValueError(f"Modello sconosciuto: {model_name}")


# =============================================================================
# ESEMPIO DI LANCIO
# =============================================================================

if __name__ == "__main__":

    # ── Configura le feature (stessa del single-task) ──
    cfg = FeatureConfig(
        elem_list      = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0],
        chirality      = ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW',
                          'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER'],
        degree         = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        hybridization  = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'OTHER'],
        bond_types     = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'OTHER'],
        stereo         = ['STEREONONE', 'STEREOANY', 'STEREOZ',
                          'STEREOE', 'STEREOCIS', 'STEREOTRANS']
    )

    TASK_NAMES = [
        "Dealkylation",
        "Glucuronidation",
        "GlutathioneConjugation",
        "Hydrolysis",
        "Oxidation",
        "Reduction",
        "Sulfonation"
    ]

    results = run_multitask_training(
        sdf_dir            = "/path/to/multitask_sdf/",
        multitask_csv_path = "/path/to/mappa_multitask.csv",
        split_dir          = "/path/to/singletask_splits/",
        duplicate_csv_path = "/path/to/mappa_duplicati.csv",
        task_names         = TASK_NAMES,
        cfg                = cfg,
        save_dir           = "/path/to/output/multitask/",
        hyperparams        = None   # oppure dict da Optuna
    )
