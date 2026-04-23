import os 
import multiprocessing as mp 
from GNN_workflow import run_training, run_tuning, FeatureConfig

from configs import (
    Dealkylation_CONFIG,
    Sulfonation_CONFIG,
    Oxidation_CONFIG,
    Reduction_CONFIG,
    Hydrolysis_CONFIG,
    Glucuronidation_CONFIG,
    GlutathioneConjugation_CONFIG
)

ROOT_PATH = r'C:\Users\Alessio Macorano\Desktop\Database-completo-AM\Dataset-no-H-modelli-AM-new-classes'

N_WORKERS = 4
N_THREADS = 4

REACTION_CONFIG_MAP = {
    "Dealkylation":           Dealkylation_CONFIG,
    "Sulfonation":            Sulfonation_CONFIG,
    "Oxidation":              Oxidation_CONFIG,
    "Reduction":              Reduction_CONFIG,
    "Hydrolysis":             Hydrolysis_CONFIG,
    "Glucuronidation":        Glucuronidation_CONFIG,
    "GlutathioneConjugation": GlutathioneConjugation_CONFIG,
}

def init_worker():
    os.environ["OMP_NUM_THREADS"]     = str(N_THREADS)
    os.environ["MKL_NUM_THREADS"]     = str(N_THREADS)
    os.environ["NUMEXPR_NUM_THREADS"] = str(N_THREADS)

def run_reaction(args):
    reaction, config = args
    reaction_path = os.path.join(ROOT_PATH, reaction)
    if not os.path.exists(reaction_path):
        return

    csv_dir = os.path.join(reaction_path, 'y-som')
    subclasses = [d for d in os.listdir(reaction_path)
                  if os.path.isdir(os.path.join(reaction_path, d))
                  and d not in ['y-som', 'Risultati-sottoclassi']]

    for subclass in subclasses:
        sub_path = os.path.join(reaction_path, subclass)
        n_file = len([f for f in os.listdir(sub_path) if f.endswith('.sdf')])

        if n_file <= 50:
            print(f"Skipping {subclass} ({n_file} mols)")
            continue

        current_config = config

        if reaction == "Hydrolysis":
            if subclass == 'Hydrolysis_of_esters':
                current_config = FeatureConfig(
                    elem_list=[6,7,8,9,15,16,17,35],
                    chirality=['CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW'],
                    degree=[1,2,3,4],
                    hybridization=['SP','SP2','SP3'],
                    bond_types=["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
                    stereo=['STEREONONE','STEREOZ','STEREOE']
                )
            elif subclass == 'Hydrolysis_of_all_imines':
                current_config = FeatureConfig(
                    elem_list=[6,7,8,9,14,15,16,17,35,53],
                    chirality=['CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW'],
                    degree=[1,2,3,4],
                    hybridization=['SP','SP2','SP3'],
                    bond_types=["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
                    stereo=['STEREONONE','STEREOZ','STEREOE']
                )

        print(f"Training: {reaction} -> {subclass} (pid={os.getpid()})")
        best_params = run_tuning(subclass, reaction_path, csv_dir, n_file, current_config, reaction, n_trials=30)
        run_training(subclass, reaction_path, csv_dir, n_file, current_config, reaction, hyperparams=best_params)

def run_all():
    tasks = [(r, c) for r, c in REACTION_CONFIG_MAP.items()
             if os.path.exists(os.path.join(ROOT_PATH, r))]

    with mp.Pool(processes=N_WORKERS, initializer=init_worker) as pool:
        pool.map(run_reaction, tasks)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    run_all()