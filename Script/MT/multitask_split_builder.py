"""
multitask_split_builder.py
==========================
Costruisce il dataset e lo split multi-task a partire da:
  1. Gli split single-task già calcolati (CSV train/val/test per reazione)
  2. La mappa dei duplicati cross-task (InChIKey → nome canonico + alias)
  3. La mappa multi-task delle label (y_som multi-task, con nome canonico)
  4. La cartella SDF multi-task (file nominati col nome canonico)

Output:
  - train_names, val_names, test_names  (liste di nomi canonici)
  - data_list                           (lista di PyG Data con y e mask)
  - conflict_report.csv                 (molecole con conflitti di split)

Logica dello split:
  - Lo split globale di ogni molecola canonica è derivato dagli split single-task
  - I duplicati cross-task vengono risolti con regola conservativa:
      conflitto (es. train in task A, test in task B) → TEST
  - La mask gestisce quali task sono annotati per ogni molecola
"""

import os
import ast
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from torch_geometric.data import Data
from rdkit import Chem

# Importa dal tuo workflow esistente
from GNN_workflow import (
    set_cfg, from_rdmol_one_hot, FeatureConfig
)


# =============================================================================
# STEP 1 — Carica gli split single-task e costruisce il dizionario per task
# =============================================================================

def load_singletask_splits(split_dir, task_names):
    """
    Legge i CSV di split single-task e restituisce un dizionario:
        split_per_task[task][nome_molecola] = "train" | "val" | "test"

    Struttura attesa nella cartella split_dir:
        {split_dir}/{task}_train.csv  → colonna con nomi molecole (senza .sdf)
        {split_dir}/{task}_val.csv
        {split_dir}/{task}_test.csv

    Parameters
    ----------
    split_dir : str
        Cartella dove sono i CSV di split single-task.
    task_names : list[str]
        Lista dei task (es. ["Oxidation", "Dealkylation", ...]).

    Returns
    -------
    split_per_task : dict[str, dict[str, str]]
        split_per_task[task][mol_name] = "train" | "val" | "test"
    """
    split_per_task = {}

    for task in task_names:
        task_filename = task.replace("match_", "")
        task_splits = {}
        for split_label in ["train", "val", "test"]:
            csv_path = os.path.join(split_dir, f"{task_filename}_{split_label}.csv")
            if not os.path.exists(csv_path):
                print(f"  [WARNING] File non trovato: {csv_path} — task {task} ignorato per split {split_label}")
                continue
            df = pd.read_csv(csv_path)
            # Prende la prima colonna qualunque sia il nome
            mol_names = df.iloc[:, 0].astype(str).tolist()
            for name in mol_names:
                name = name.replace("molecola_", "")
                task_splits[name] = split_label

        split_per_task[task] = task_splits
        n_train = sum(1 for v in task_splits.values() if v == "train")
        n_val   = sum(1 for v in task_splits.values() if v == "val")
        n_test  = sum(1 for v in task_splits.values() if v == "test")
        print(f"  {task}: {n_train} train | {n_val} val | {n_test} test")

    return split_per_task


# =============================================================================
# STEP 2 — Carica la mappa dei duplicati e costruisce alias → canonico
# =============================================================================

def load_duplicate_map(duplicate_csv_path):
    """
    Legge la mappa dei duplicati e restituisce due dizionari:

        alias_to_canonical[task_alias][alias_name] = canonical_name
        canonical_tasks[canonical_name]            = task_canonico

    Formato CSV atteso (separatore ;):
        InChIKey;Nome_canonico;Task_canonico;Alias;Task_alias;N_duplicati

    dove Alias può essere "[16222]" o "[7251; 53075]" (lista con ;)
    e Task_alias può essere "['Oxidation']" o "['Oxidation'; 'hydrolysis']"

    Parameters
    ----------
    duplicate_csv_path : str

    Returns
    -------
    alias_to_canonical : dict[str, dict[str, str]]
        alias_to_canonical[task][alias_mol_name] = canonical_mol_name
    canonical_tasks : dict[str, str]
        canonical_tasks[canonical_mol_name] = task_canonico
    """
    df = pd.read_csv(duplicate_csv_path, sep=';')

    alias_to_canonical = defaultdict(dict)   # task → {alias_name: canonical_name}
    canonical_tasks    = {}                  # canonical_name → task_canonico

    for _, row in df.iterrows():
        canonical_name = str(row['Nome_canonico']).strip()
        canonical_task = str(row['Task_canonico']).strip()
        canonical_tasks[canonical_name] = canonical_task

        # Parsa Alias: può essere "[16222]" o "[7251; 53075]"
        alias_raw = str(row['Alias']).strip()
        try:
            # Sostituisce ; con , dentro le liste per renderle parsabili
            alias_clean = alias_raw.replace(';', ',')
            alias_list  = ast.literal_eval(alias_clean)
            if not isinstance(alias_list, list):
                alias_list = [alias_list]
        except Exception:
            alias_list = [alias_raw.strip("[]").strip()]

        # Parsa Task_alias: può essere "['Oxidation']" o "['Oxidation'; 'hydrolysis']"
        task_alias_raw = str(row['Task_alias']).strip()
        try:
            task_alias_clean = task_alias_raw.replace(';', ',')
            task_alias_list  = ast.literal_eval(task_alias_clean)
            if not isinstance(task_alias_list, list):
                task_alias_list = [task_alias_list]
        except Exception:
            task_alias_list = [task_alias_raw.strip("[]'").strip()]

        # Associa ogni alias al suo task e al nome canonico
        for alias_name, alias_task in zip(alias_list, task_alias_list):
            alias_name = str(alias_name).strip()
            alias_task = str(alias_task).strip().strip("'")
            alias_to_canonical[alias_task][alias_name] = canonical_name

    print(f"  Duplicati trovati: {len(df)} righe")
    print(f"  Task con alias: {list(alias_to_canonical.keys())}")

    return alias_to_canonical, canonical_tasks


# =============================================================================
# STEP 3 — Traduce gli split single-task in nomi canonici
# =============================================================================

def translate_splits_to_canonical(split_per_task, alias_to_canonical):
    """
    Traduce i nomi alias degli split single-task nel nome canonico.

    Per ogni task, ogni molecola nello split viene cercata nella mappa alias:
      - Se è un alias → viene sostituita con il nome canonico
      - Se non è un alias → il nome è già canonico

    Parameters
    ----------
    split_per_task : dict[str, dict[str, str]]
        Output di load_singletask_splits.
    alias_to_canonical : dict[str, dict[str, str]]
        Output di load_duplicate_map.

    Returns
    -------
    canonical_split_per_task : dict[str, dict[str, str]]
        canonical_split_per_task[task][canonical_mol_name] = "train"|"val"|"test"
    translation_log : list[dict]
        Log di ogni traduzione alias→canonico effettuata.
    """
    canonical_split_per_task = {}
    translation_log = []

    for task, mol_splits in split_per_task.items():

        task_clean = task.replace("match_", "")
        task_alias_map = alias_to_canonical.get(task_clean, {})
        
        canonical_splits = {}

        for mol_name, split_label in mol_splits.items():
            if mol_name in task_alias_map:
                # Questo nome è un alias: lo traduciamo
                canonical_name = task_alias_map[mol_name]
                canonical_splits[canonical_name] = split_label
                translation_log.append({
                    "task":           task,
                    "alias":          mol_name,
                    "canonical":      canonical_name,
                    "split":          split_label
                })
            else:
                # Nome già canonico
                canonical_splits[mol_name] = split_label

        canonical_split_per_task[task] = canonical_splits

    print(f"  Traduzioni alias→canonico effettuate: {len(translation_log)}")
    return canonical_split_per_task, translation_log


# =============================================================================
# STEP 4 — Risolve i conflitti e assegna lo split globale
# =============================================================================

# Priorità: test > val > train
# Una molecola con conflitto viene messa nel set più restrittivo.
SPLIT_PRIORITY = {"test": 2, "val": 1, "train": 0}


def resolve_global_split(canonical_split_per_task, task_names):
    """
    Per ogni molecola canonica, raccoglie gli split assegnati in tutti i task
    e risolve i conflitti con la regola conservativa:
        conflitto → split con priorità più alta (test > val > train)

    Parameters
    ----------
    canonical_split_per_task : dict[str, dict[str, str]]
    task_names : list[str]

    Returns
    -------
    global_split : dict[str, str]
        global_split[canonical_mol_name] = "train" | "val" | "test"
    conflict_report : list[dict]
        Lista di molecole con conflitto di split tra task diversi.
    """
    # Raccoglie tutti gli split per ogni molecola canonica
    mol_splits_across_tasks = defaultdict(dict)
    for task in task_names:
        for mol_name, split_label in canonical_split_per_task.get(task, {}).items():
            mol_splits_across_tasks[mol_name][task] = split_label

    global_split    = {}
    conflict_report = []

    for mol_name, task_split_map in mol_splits_across_tasks.items():
        unique_splits = set(task_split_map.values())

        if len(unique_splits) == 1:
            # Nessun conflitto
            global_split[mol_name] = unique_splits.pop()
        else:
            # Conflitto: prende lo split con priorità più alta
            resolved = max(unique_splits, key=lambda s: SPLIT_PRIORITY.get(s, -1))
            global_split[mol_name] = resolved
            conflict_report.append({
                "molecola_canonica": mol_name,
                "split_per_task":    str(task_split_map),
                "split_risolto":     resolved,
                "n_task_coinvolti":  len(task_split_map)
            })

    n_train = sum(1 for v in global_split.values() if v == "train")
    n_val   = sum(1 for v in global_split.values() if v == "val")
    n_test  = sum(1 for v in global_split.values() if v == "test")
    print(f"  Split globale: {n_train} train | {n_val} val | {n_test} test")
    print(f"  Conflitti risolti: {len(conflict_report)}")

    return global_split, conflict_report


# =============================================================================
# STEP 5 — Carica molecole SDF dalla cartella multi-task
# =============================================================================

def load_multitask_molecules(sdf_dir):
    """
    Carica tutti i file SDF dalla cartella multi-task.
    I nomi dei file devono essere i nomi canonici (senza .sdf).

    Parameters
    ----------
    sdf_dir : str
        Cartella contenente i file SDF multi-task.

    Returns
    -------
    mols  : list[Chem.Mol]
    names : list[str]   (nome canonico senza .sdf)
    """
    mols, names = [], []
    for f in sorted(os.listdir(sdf_dir)):
        if not f.endswith('.sdf'):
            continue
        mol_path = os.path.join(sdf_dir, f)
        m = Chem.MolFromMolFile(mol_path)
        if m is None:
            print(f"  [WARNING] Impossibile leggere: {f}")
            continue
        m = Chem.RemoveAllHs(m)
        for a in m.GetAtoms():
            a.SetNoImplicit(True)
        m.UpdatePropertyCache()
        mols.append(m)
        name = f[:-4]
        if name.startswith("molecola_"):
            name = name.replace("molecola_", "") 
        names.append(name)    # rimuove .sdf

    print(f"  Molecole caricate dalla cartella multi-task: {len(mols)}")
    return mols, names


# =============================================================================
# STEP 6 — Costruisce i grafi PyG con y e mask
# =============================================================================

def build_multitask_graphs(mols, names, multitask_label_df, task_names):
    """
    Costruisce i grafi PyG con:
        data.y    = label [n_atomi, n_tasks]  (0 dove NaN)
        data.mask = mask  [n_atomi, n_tasks]  (1 se annotato, 0 se NaN)
        data.name = nome canonico

    Parameters
    ----------
    mols : list[Chem.Mol]
    names : list[str]
        Nomi canonici (devono matchare la colonna 'molecole' nel CSV label).
    multitask_label_df : pd.DataFrame
        DataFrame con colonne: molecole, indice_atomi, match_Task1, match_Task2, ...
        I NaN indicano task non annotati per quella molecola.
    task_names : list[str]

    Returns
    -------
    data_list : list[Data]
    """
    data_list = []
    missing   = []

    for mol, name in zip(mols, names):
        g = from_rdmol_one_hot(mol)

        mol_rows = multitask_label_df[multitask_label_df['molecole'] == name]

        if mol_rows.empty:
            missing.append(name)
            continue

        # Controlla coerenza numero atomi
        if len(mol_rows) != g.x.shape[0]:
            print(f"  [WARNING] {name}: {g.x.shape[0]} atomi nel SDF, "
                  f"{len(mol_rows)} righe nel CSV — molecola saltata")
            continue

        label_vals = mol_rows[task_names].values.astype(float)
        mask_vals  = mol_rows[task_names].notna().values.astype(float)

        # Dove mask=0, forza y=0 (non entra nella loss comunque)
        y_safe = np.where(mask_vals.astype(bool), label_vals, 0.0)

        data      = Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr)
        data.name = name
        data.y    = torch.tensor(y_safe,    dtype=torch.float)
        data.mask = torch.tensor(mask_vals, dtype=torch.float)
        data_list.append(data)

    if missing:
        print(f"  [WARNING] {len(missing)} molecole senza label nel CSV: {missing[:5]}{'...' if len(missing)>5 else ''}")

    print(f"  Grafi costruiti: {len(data_list)}")
    return data_list


# =============================================================================
# STEP 7 — Calcola le frequenze di classe (solo su atomi annotati nel train)
# =============================================================================

def compute_multitask_frequencies(train_names, multitask_label_df, task_names):
    """
    Calcola n0 e n1 per ogni task usando solo gli atomi annotati
    (mask=1, cioè non NaN) nel training set.

    Parameters
    ----------
    train_names : list[str]
    multitask_label_df : pd.DataFrame
    task_names : list[str]

    Returns
    -------
    frequencies : dict[str, tuple[int, int]]
        frequencies[task] = (n0, n1)
    """
    train_df   = multitask_label_df[multitask_label_df['molecole'].isin(set(train_names))]
    frequencies = {}

    for t in task_names:
        col  = train_df[t].dropna()      # solo atomi effettivamente annotati
        n0_t = int((col == 0).sum())
        n1_t = int((col == 1).sum())
        frequencies[t] = (n0_t, n1_t)
        print(f"    {t}: n0={n0_t}, n1={n1_t}")

    return frequencies


# =============================================================================
# FUNZIONE PRINCIPALE
# =============================================================================

def build_multitask_dataset(
    sdf_dir,
    multitask_csv_path,
    split_dir,
    duplicate_csv_path,
    task_names,
    cfg,
    save_dir=None
):
    """
    Pipeline completa per costruire il dataset multi-task ottimizzato.

    Parameters
    ----------
    sdf_dir : str
        Cartella con i file SDF multi-task (nomi canonici).
    multitask_csv_path : str
        CSV con le label multi-task (colonna 'molecole' = nome canonico).
    split_dir : str
        Cartella con i CSV di split single-task ({task}_train/val/test.csv).
    duplicate_csv_path : str
        CSV della mappa dei duplicati cross-task (separatore ;).
    task_names : list[str]
        Lista dei task nell'ordine usato nel CSV label
        (es. ["Dealkylation", "Glucuronidation", "Oxidation", ...]).
    cfg : FeatureConfig
        Configurazione delle feature (condivisa con GNN_workflow).
    save_dir : str, optional
        Se specificato, salva i report e gli split in questa cartella.

    Returns
    -------
    train_names : list[str]
    val_names   : list[str]
    test_names  : list[str]
    data_list   : list[Data]
    frequencies : dict[str, tuple[int, int]]
    """
    set_cfg(cfg)

    print("\n" + "="*60)
    print("STEP 1 — Caricamento split single-task")
    print("="*60)
    split_per_task = load_singletask_splits(split_dir, task_names)

    print("\n" + "="*60)
    print("STEP 2 — Caricamento mappa duplicati")
    print("="*60)
    alias_to_canonical, canonical_tasks = load_duplicate_map(duplicate_csv_path)

    print("\n" + "="*60)
    print("STEP 3 — Traduzione alias → nomi canonici")
    print("="*60)
    canonical_split_per_task, translation_log = translate_splits_to_canonical(
        split_per_task, alias_to_canonical
    )

    print("\n" + "="*60)
    print("STEP 4 — Risoluzione conflitti e split globale")
    print("="*60)
    global_split, conflict_report = resolve_global_split(
        canonical_split_per_task, task_names
    )

    print("\n" + "="*60)
    print("STEP 5 — Caricamento molecole SDF multi-task")
    print("="*60)
    mols, names = load_multitask_molecules(sdf_dir)

    # Verifica che tutte le molecole SDF abbiano uno split assegnato
    no_split = [n for n in names if n not in global_split]
    if no_split:
        print(f"  [WARNING] {len(no_split)} molecole SDF senza split assegnato "
              f"(non presenti in nessun split single-task): {no_split[:5]}")
        print(f"  Queste molecole verranno escluse.")
        # Filtra mols e names
        mols  = [m for m, n in zip(mols, names) if n in global_split]
        names = [n for n in names if n in global_split]

    print("\n" + "="*60)
    print("STEP 6 — Costruzione label multi-task")
    print("="*60)
    # Carica il CSV delle label multi-task
    # Atteso: colonne = molecole, indice_atomi, match_Task1, match_Task2, ...
    multitask_label_df = pd.read_csv(multitask_csv_path)
    multitask_label_df['molecole'] = multitask_label_df['molecole'].astype(str)

    # Rinomina le colonne match_* → nomi task se necessario
    # (adatta qui se i tuoi nomi colonna sono diversi da task_names)
    col_map = {}
    for col in multitask_label_df.columns:
        for task in task_names:
            if task.lower() in col.lower() and col != task:
                col_map[col] = task
                break
    if col_map:
        multitask_label_df = multitask_label_df.rename(columns=col_map)
        print(f"  Rinominazione colonne: {col_map}")

    print("\n" + "="*60)
    print("STEP 6b — Costruzione grafi PyG")
    print("="*60)
    data_list = build_multitask_graphs(mols, names, multitask_label_df, task_names)

    # Sincronizza: tieni solo i grafi con split assegnato
    data_list = [d for d in data_list if d.name in global_split]

    # Costruisci le liste di nomi per split
    train_names = [d.name for d in data_list if global_split[d.name] == "train"]
    val_names   = [d.name for d in data_list if global_split[d.name] == "val"]
    test_names  = [d.name for d in data_list if global_split[d.name] == "test"]

    print(f"\n  Split finale sui grafi costruiti:")
    print(f"    train: {len(train_names)} molecole")
    print(f"    val:   {len(val_names)} molecole")
    print(f"    test:  {len(test_names)} molecole")

    print("\n" + "="*60)
    print("STEP 7 — Calcolo frequenze di classe (solo training)")
    print("="*60)
    frequencies = compute_multitask_frequencies(
        train_names, multitask_label_df, task_names
    )

    # ── Salvataggio report ──
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        pd.DataFrame(train_names, columns=['molecola']).to_csv(
            os.path.join(save_dir, "multitask_train.csv"), index=False)
        pd.DataFrame(val_names,   columns=['molecola']).to_csv(
            os.path.join(save_dir, "multitask_val.csv"),   index=False)
        pd.DataFrame(test_names,  columns=['molecola']).to_csv(
            os.path.join(save_dir, "multitask_test.csv"),  index=False)

        if conflict_report:
            pd.DataFrame(conflict_report).to_csv(
                os.path.join(save_dir, "conflict_report.csv"), index=False)
            print(f"\n  conflict_report.csv salvato in {save_dir}")

        if translation_log:
            pd.DataFrame(translation_log).to_csv(
                os.path.join(save_dir, "translation_log.csv"), index=False)
            print(f"  translation_log.csv salvato in {save_dir}")

        # Split globale completo
        pd.DataFrame(
            [{"molecola": k, "split": v} for k, v in global_split.items()]
        ).to_csv(os.path.join(save_dir, "global_split.csv"), index=False)

    print("\n" + "="*60)
    print("Dataset multi-task costruito correttamente.")
    print("="*60 + "\n")

    return train_names, val_names, test_names, data_list, frequencies
