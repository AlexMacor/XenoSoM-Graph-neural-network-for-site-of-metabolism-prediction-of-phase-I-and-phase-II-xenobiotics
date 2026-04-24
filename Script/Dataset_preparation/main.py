import os
import numpy as np
import re
import pandas as pd
import rdkit 
from rdkit import Chem 
from rdkit.Chem import AllChem
import shutil
import argparse

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


def find_metals(df_no_smiles): 
    smiles_column = df_no_smiles['SubstrateSMILES']
    name_column = df_no_smiles['Name']

    mol_smi = []
    for smile in smiles_column:
        mol_smi.append(Chem.MolFromSmiles(smile))

    smart_patterns = [
        '[Li]', '[Be]', '[B]', '[Na]', '[Mg]', '[Al]', '[Si]', '[K]', '[Ca]',
        '[Sc]', '[Ti]', '[V]', '[Cr]', '[Mn]', '[Fe]', '[Co]', '[Ni]', '[Cu]',
        '[Zn]', '[Ga]', '[Ge]', '[As]', '[Se]', '[Rb]', '[Sr]', '[Y]', '[Zr]',
        '[Nb]', '[Mo]', '[Tc]', '[Ru]', '[Rh]', '[Pd]', '[Ag]', '[Cd]', '[In]',
        '[Sn]', '[Sb]', '[Te]', '[Cs]', '[Ba]', '[La]', '[Ce]', '[Pr]', '[Nd]',
        '[Pm]', '[Sm]', '[Eu]', '[Gd]', '[Tb]', '[Dy]', '[Ho]', '[Er]', '[Tm]',
        '[Yb]', '[Lu]', '[Hf]', '[Ta]', '[W]', '[Re]', '[Os]', '[Ir]', '[Pt]',
        '[Au]', '[Hg]', '[Tl]', '[Pb]', '[Bi]', '[Th]', '[U]', '[Np]', '[Pu]',
        '[Fr]', '[Ra]', '[Rf]', '[Db]', '[Sg]', '[Bh]', '[Hs]', '[Mt]', '[Ds]',
        '[Rg]', '[Cn]', '[Fl]', '[Lv]'
    ]

    exotic_atoms = {}
    index = 0

    for mol in mol_smi:
        if mol is None:
            index += 1
            continue

        all_matches = {}
        for atom in smart_patterns:
            atomo_exo = Chem.MolFromSmarts(atom)
            substr_match = mol.GetSubstructMatches(atomo_exo)
            if substr_match:
                all_matches[atom] = substr_match

        if all_matches:
            name = name_column[index]
            exotic_atoms[name] = ', '.join(all_matches.keys())

        index += 1

    df_metals = pd.DataFrame.from_dict(exotic_atoms, orient='index', columns=['MetalliPresenti'])
    df_metals.reset_index(inplace=True)
    df_metals.rename(columns={'index': 'Name'}, inplace=True)
    # df_metals.to_csv(output_csv_path, index=False) eventually save 
 
    return df_metals

# generate filtered csv for both rxn name and rnx class
def split_reactions_by_RXN_CLASS(df_final, base_path: str):

    reaction_classes = df_final['RXNClass'].unique()

    for rxn in reaction_classes:
        df_rxn = df_final[df_final['RXNClass'] == rxn]

        if 'SubstrateInChIKey' in df_rxn.columns:
            df_rxn = df_rxn.drop_duplicates(subset=['SubstrateInChIKey'], keep='first')

        folder_path = os.path.join(base_path, rxn)
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f'{rxn}.csv')

        df_rxn.to_csv(file_path, index=False)

    print(f"Saved: {base_path}")

def split_reactions_by_RXN_CLASS_Name(df_final, base_path: str, sanitize):

    for rxn_class in df_final['RXNClass'].unique():
        class_folder = os.path.join(base_path, sanitize(str(rxn_class)))
        os.makedirs(class_folder, exist_ok=True)

        df_class = df_final[df_final['RXNClass'] == rxn_class]

        for rxn_name in df_class['RXNName'].unique():
            safe_name = sanitize(str(rxn_name))
            file_path = os.path.join(class_folder, f"{safe_name}.csv")

            df_sub = df_class[df_class['RXNName'] == rxn_name]
            if 'SubstrateInChIKey' in df_sub.columns:
                df_sub = df_sub.drop_duplicates(subset=['SubstrateInChIKey'], keep='first')

            df_sub.to_csv(file_path, index=False)
            print(f"Saved: {file_path}")

 
def check_match_csv_folders(base_path):
    cartelle = [nome for nome in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, nome))]
    file_csv = [os.path.splitext(nome)[0] for nome in os.listdir(base_path) if nome.endswith('.csv')]

    match = []
    no_match = []

    for cartella in cartelle:
        if cartella in file_csv:
            match.append(cartella)
        else:
            no_match.append(cartella)

    return match, no_match


def copy_sdf_to_folders(base_path: str, sdf_source: str):
    for filename in os.listdir(base_path):
        if filename.endswith('.csv'):
            csv_name = os.path.splitext(filename)[0]
            csv_path = os.path.join(base_path, filename)
            target_folder = os.path.join(base_path, csv_name)

            if os.path.isdir(target_folder) and os.path.basename(target_folder) == csv_name:
                df = pd.read_csv(csv_path)

                if 'Name' in df.columns:
                    names = [str(n).strip() for n in df['Name']]

                    for name in names:
                        sdf_filename = f"molecola_{name}.sdf"
                        source_path = os.path.join(sdf_source, sdf_filename)
                        destination_path = os.path.join(target_folder, sdf_filename)

                        if os.path.exists(source_path):
                            shutil.copy(source_path, destination_path)
                        else:
                            print(f"not found: {sdf_filename}")
            else:
                print(f"no match csv - folder: {csv_name}")

def set_sdf_for_calc(base_path: str) -> set:
    sdf_files = set()
    for dirpath, _, filenames in os.walk(base_path):
        for filename in filenames:
            if filename.endswith(".sdf"):
                sdf_files.add(filename)
    return sdf_files

##
## Y SOM PREPARATION 
## 


def get_atm_indices_from_sdf(base_path: str):
    valori_da_sdf = {}

    for dirp, dirn, filen in os.walk(base_path):
        valori_da_sdf = {}
        for file in filen:
            if file.endswith('.sdf'):
                file_path = os.path.join(dirp, file)
                m = Chem.MolFromMolFile(file_path) # to use H atoms for 3D graph, change 
                m_h = Chem.RemoveAllHs(m)          # with m = Chem.MolFromMolFile(file_path, removeHs=True, sanitize=True)
                for atom in m_h.GetAtoms():
                    atom.SetNoImplicit(True)
                    m_h.UpdatePropertyCache()
                atom_labels = []
                for atom in m_h.GetAtoms():
                    atom_label = atom.GetIdx() + 1
                    atom_labels.append(atom_label)
                valori_da_sdf[file] = atom_labels

        if valori_da_sdf:
            df = pd.Series(valori_da_sdf).rename_axis('molecole').reset_index(name='indice_atomi')
            df = df.explode('indice_atomi').reset_index(drop=True)

            nome_cartella = os.path.basename(dirp)
            parent_dir = os.path.dirname(dirp)
            output_path = os.path.join(parent_dir, f'{nome_cartella}_indice_atomi_da_sdf.csv')

            df.to_csv(output_path, sep=',', index=False)

# the following function will works only for RXN Name 
def up_reactive_atoms_RXNname(base_path, df_final):

    for dirpath, dirnames, filenames in os.walk(base_path):
        for filename in filenames:
            if filename.endswith('.csv') and not filename.endswith('_indice_atomi_da_sdf.csv'):
                file_path = os.path.join(dirpath, filename)
                df = pd.read_csv(file_path, sep=',')

                som_dict = {}
                
                set_file_csv = set(df[['RXNClass','RXNName', 'SubstrateInChIKey', 'SubstrateReactiveAtoms']].itertuples(index=False, name=None))
                set_file_all_db = set(df_final[['RXNClass', 'RXNName', 'SubstrateInChIKey', 'SubstrateReactiveAtoms']].itertuples(index=False, name=None))

                unmatched = set_file_all_db - set_file_csv

                df_unmatched = pd.DataFrame(unmatched, columns=['RXNClass','RXNName', 'SubstrateInChIKey', 'SubstrateReactiveAtoms'])

                for _, row in df_unmatched.iterrows():
                    key = (row['RXNClass'], row['RXNName'], row['SubstrateInChIKey'])
                    som_dict.setdefault(key, []).append(row['SubstrateReactiveAtoms'])

                df['SubstrateReactiveAtoms'] = df.apply(
                    lambda row: ', '.join(dict.fromkeys(
                        re.split(r'[,\s\n]+', f"{row['SubstrateReactiveAtoms']}, {', '.join(som_dict[(row['RXNClass'], row['RXNName'], row['SubstrateInChIKey'])])}")
                    )) if (row['RXNClass'], row['RXNName'], row['SubstrateInChIKey']) in som_dict else row['SubstrateReactiveAtoms'],
                    axis=1
                )

                atomi_reattivi_cn = df['SubstrateReactiveAtoms'].apply(
                    lambda x: list(dict.fromkeys(
                        [int(num) for num in re.findall(r'\d+', str(x))] 
                    )) if pd.notna(x) else []
                )

                df['SubstrateReactiveAtoms'] = atomi_reattivi_cn
                output_dir = os.path.join(dirpath)
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, f'{base_name}_1.csv')

                df.to_csv(output_path, sep=',', index=False)
                
                # CSV files for classes must be removed and 
                # regenerated with the following function

                name_to_rmv = ['Dealkylation', 'Equilibrium', 'Glucuronidation', 'GlutathioneConjugation',
                 'Hydrolysis', 'Oxidation', 'Reduction', 'Sulfonation']
                if base_name in name_to_rmv and os.path.exists(output_path):
                    os.remove(output_path)
                    print(f"removed classes file: {output_path}")

def up_reactive_atoms_RXNClass(base_path, df_final):
    
    class_folders = ['Dealkylation', 'Equilibrium', 'Glucuronidation', 'GlutathioneConjugation',
                     'Hydrolysis', 'Oxidation', 'Reduction', 'Sulfonation']


    for dirpath, dirnames, filenames in os.walk(base_path):
        for filename in filenames:
            if (filename.endswith('.csv') and not filename.endswith('_indice_atomi_da_sdf.csv') 
                and not filename.endswith('_1.csv') and os.path.splitext(filename)[0] in class_folders):
                print(filename)

                file_path = os.path.join(dirpath, filename)
                df = pd.read_csv(file_path, sep=',')

                som_dict = {}
                
                set_file_csv = set(df[['RXNClass', 'SubstrateInChIKey', 'SubstrateReactiveAtoms']].itertuples(index=False, name=None))
                set_file_all_db = set(df_final[['RXNClass', 'SubstrateInChIKey', 'SubstrateReactiveAtoms']].itertuples(index=False, name=None))

                unmatched = set_file_all_db - set_file_csv

                df_unmatched = pd.DataFrame(unmatched, columns=['RXNClass', 'SubstrateInChIKey', 'SubstrateReactiveAtoms'])

                for _, row in df_unmatched.iterrows():
                    key = (row['RXNClass'],  row['SubstrateInChIKey'])
                    som_dict.setdefault(key, []).append(row['SubstrateReactiveAtoms'])

                df['SubstrateReactiveAtoms'] = df.apply(
                    lambda row: list(dict.fromkeys(
                        [int(n) for n in re.findall(r'\d+', f"{row['SubstrateReactiveAtoms']}, {', '.join(som_dict[(row['RXNClass'], row['SubstrateInChIKey'])])}")]
                    )) if (row['RXNClass'], row['SubstrateInChIKey']) in som_dict else 
                    [int(n) for n in re.findall(r'\d+', str(row['SubstrateReactiveAtoms']))],
                    axis=1
                )

                atomi_reattivi_cn = df['SubstrateReactiveAtoms'].apply(lambda x: list(dict.fromkeys([int(num) for num in re.findall(r'\d+', str(x))])) 
                                                                       if isinstance(x, (str, list)) else [])
                
                df['SubstrateReactiveAtoms'] = atomi_reattivi_cn
                output_dir = os.path.join(dirpath)
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, f'{base_name}_1.csv')

                df.to_csv(output_path, sep=',', index=False)


def create_y_som(base_path, out_soms):
    os.makedirs(out_soms, exist_ok=True)

    def check_match(atomi_reattivi, indice_atomo):
        if atomi_reattivi is None or not isinstance(atomi_reattivi, (list, np.ndarray)):
            return 0
        return 1 if indice_atomo in list(atomi_reattivi) else 0

    for dirpath, dirnames, filenames in os.walk(base_path):
        som_files = {}
        indici_files = {}

        for filename in filenames:
            base_name = None
            file_path = os.path.join(dirpath, filename)

            if filename.endswith('_1.csv') and not filename.endswith('_indice_atomi_da_sdf.csv'):
                base_name = filename.replace('_1.csv', '')
                som_files[base_name] = file_path

            elif filename.endswith('_indice_atomi_da_sdf.csv'):
                base_name = filename.replace('_indice_atomi_da_sdf.csv', '')
                indici_files[base_name] = file_path

        for base_name in som_files:
            if base_name in indici_files:
                file_som = som_files[base_name]
                file_idx = indici_files[base_name]

                #print(f"Match tra: {file_som} e {file_idx}")

                df_siti = pd.read_csv(file_som, sep=',', converters={'SubstrateReactiveAtoms': eval})
                df_siti['Name'] = df_siti['Name'].astype(int)

                df_indici = pd.read_csv(file_idx, sep=',')
                df_indici['molecole'] = df_indici['molecole'].str.extract(r'molecola_(\d+)\.sdf')[0]
                df_indici['molecole'] = df_indici['molecole'].astype(int)

                match_results = []

                for _, row in df_indici.iterrows():
                    molecola = row['molecole']
                    indice_atomo = row['indice_atomi']
                    matching_siti = df_siti[df_siti['Name'] == molecola]

                    if not matching_siti.empty:
                        atomi_reattivi = matching_siti.iloc[0]['SubstrateReactiveAtoms']
                        match_results.append(check_match(atomi_reattivi, indice_atomo))
                    else:
                        match_results.append(0)

                df_indici['match'] = match_results

                output_file = os.path.join(out_soms, f'y_som_{base_name}_match.csv')
                df_indici.to_csv(output_file, sep=',', index=False)

                print(f"saved: {output_file}")

# create results folder 
def results_folder(base_path):
    dest_root = os.path.join(base_path, 'Risultati-sottoclassi')
    os.makedirs(dest_root, exist_ok=True)
    for folder in os.listdir(base_path):
        full_folder_path = os.path.join(base_path, folder)
        if os.path.isdir(full_folder_path) and folder != 'Risultati-sottoclassi':
            new_folder = os.path.join(dest_root, folder)
            os.makedirs(new_folder, exist_ok=True)
            print(f"Creata: {new_folder}")

            for subfolder_name in ['rdkit', 'cdk']:
                subfolder_path = os.path.join(new_folder, subfolder_name)
                os.makedirs(subfolder_path, exist_ok=True)
                print(f"Creata sottocartella: {subfolder_path}")


if __name__ == "__main__":
    
    # example of used path 
    #base_path = r'C:\Users\User name\Desktop\1\Dataset'
    #sdf_source = r'C:\Users\User name\Desktop\1\all-sdf-files'
    #out_soms = r'C:\Users\User name\Desktop\1\Dataset\Equilibrium\y_som'
    #df = pd.read_csv(r'C:\Users\User name\Desktop\1\Equilibrium.csv', sep=',')
    
    parser = argparse.ArgumentParser(description="data structure")
    parser.add_argument("--csv", required=True, help="input csv containing the classes and subclasses")
    parser.add_argument("--base", required=True, help="base path")
    parser.add_argument("--sdf", required=True, help="folder with all sdfs files")
    parser.add_argument("--out_som", required=True, help="folder to save y_som")

    args = parser.parse_args()

    csv_path = args.csv
    base_path = args.base
    sdf_source = args.sdf
    out_soms = args.out_som    
    df = pd.read_csv(csv_path, sep=',')
    
    unique_inchi = df.drop_duplicates(subset=['SubstrateInChIKey'], keep='first')

    smiles_column = df['SubstrateSMILES']
    name_column = df['Name']
    smiles_invalid = {}

    for name, smile in zip(name_column, smiles_column):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            smiles_invalid[name] = smile

    df_bad_smiles = pd.DataFrame.from_dict(smiles_invalid, orient='index', columns=['SubstrateSMILES'])
    df_bad_smiles.reset_index(inplace=True)
    df_bad_smiles.rename(columns={'index': 'Name'}, inplace=True)
    names_to_remove = df_bad_smiles['Name']

    df_no_smiles = df[~df['Name'].isin(names_to_remove)]

    df_metals = find_metals(df_no_smiles)  # opzionale salvataggio
    mol_with_metals = df_metals['Name'].tolist()
    df_final = df_no_smiles[~df_no_smiles['Name'].isin(mol_with_metals)]
    
    split_reactions_by_RXN_CLASS(df_final, base_path)
    split_reactions_by_RXN_CLASS_Name(df_final, base_path, sanitize)
    
    for sub in os.listdir(base_path):
        subfolder_path = os.path.join(base_path, sub)
        if os.path.isdir(subfolder_path):
            match, no_match = check_match_csv_folders(subfolder_path)
            copy_sdf_to_folders(subfolder_path, sdf_source)

    set_all_sdf = set_sdf_for_calc(base_path)
    get_atm_indices_from_sdf(base_path)
    up_reactive_atoms_RXNname(base_path, df_final)
    up_reactive_atoms_RXNClass(base_path, df_final)
    
    create_y_som(base_path, out_soms)
   
    for sub in os.listdir(base_path):
        subfolder_path = os.path.join(base_path, sub)
        if os.path.isdir(subfolder_path):
            results_folder(subfolder_path)

    for dirpath, dirnames, filenames in os.walk(base_path):
        for filename in filenames:
            if filename.endswith('.csv') and not filename.endswith('_match.csv'):
                file_path = os.path.join(dirpath, filename)
                os.remove(file_path)
                print(f"Removed: {file_path}")
