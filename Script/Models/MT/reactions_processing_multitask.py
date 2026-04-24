import os
import pandas as pd
from GNN_workflow_multitask import run_multitask_training, run_multitask_tuning
from GNN_workflow import FeatureConfig

ROOT_PATH   = r'/home/altair/Dati_A/Dati_Ricerca/xalessio/Multitask'
SDF_DIR     = r'/home/altair/Dati_A/Dati_Ricerca/xalessio/Multitask/Multitask'        # cartella SDF canonici
CSV_PATH    = r'/home/altair/Dati_A/Dati_Ricerca/xalessio/Multitask/y-som/y_som_Multitask_match.csv' # y_som multitask
SPLIT_DIR   = r'/home/altair/Dati_A/Dati_Ricerca/xalessio/Multitask/'     # {task}_train/val/test.csv
DUPL_PATH   = r'/home/altair/Dati_A/Dati_Ricerca/xalessio/Multitask/mappa_multitask.csv'   # mappa duplicati

def run_all():
    all_metrics = []

    GLOBAL_CONFIG = FeatureConfig(
        elem_list     = [6, 7, 8, 9, 14, 15, 16, 17, 35, 53],
        chirality     = ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW'],
        degree        = [1, 2, 3, 4],
        hybridization = ['SP', 'SP2', 'SP3'],
        bond_types    = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
        stereo        = ['STEREONONE', 'STEREOZ', 'STEREOE']
    )

    task_names = [
        'match_Dealkylation',
        'match_Glucuronidation',
        'match_GlutathioneConjugation',
        'match_Hydrolysis',
        'match_Oxidation',
        'match_Reduction',
        'match_Sulfonation'
    ]

    save_dir = os.path.join(ROOT_PATH, 'Risultati-sottoclassi', 'Multitask', 'rdkit')

    best_params  = run_multitask_tuning(
        sdf_dir            = SDF_DIR,
        multitask_csv_path = CSV_PATH,
        split_dir          = SPLIT_DIR,
        duplicate_csv_path = DUPL_PATH,
        task_names         = task_names,
        cfg                = GLOBAL_CONFIG,
        save_dir           = save_dir,
        n_trials           = 30
    )

    metrics_list = run_multitask_training(
        sdf_dir            = SDF_DIR,
        multitask_csv_path = CSV_PATH,
        split_dir          = SPLIT_DIR,
        duplicate_csv_path = DUPL_PATH,
        task_names         = task_names,
        cfg                = GLOBAL_CONFIG,
        save_dir           = save_dir,
        hyperparams        = best_params
    )

    if metrics_list:
        df = pd.DataFrame(metrics_list)
        df.to_csv(os.path.join(ROOT_PATH, "GNN_multitask_metrics_rdkit.csv"), index=False)
        print("\nReport globale salvato.")

if __name__ == "__main__":
    run_all()