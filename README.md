# XenoSoM: Graph neural networks for site of metabolism prediction of xenobiotics

Requirements
* Python 3.13.0
* Numpy 2.3.1
* pandas 2.3.0
* Matplotlib 3.10.3
* rdkit 2025.3.3
* scikit-learn 1.7.0
* torch 2.7.1+cu118
* torch-geometric 2.6.1

## 1. Create and activate the virtual environment

Run the following command in your project directory:
```bash
python3 -m venv .GNN_SoM_env
```

## 2. Activate the virtual environment

On Linux: source .GNN_SoM_env/bin/activate  
On Windows: .GNN_SoM_env\Scripts\activate


## 3 Dataset preparation
Dataset preparation involves several steps. Starting from the CSV file, the entire data structure is created in order to perform both descriptor calculations and GNN training/inference.

Run the following command in your project directory:
```bash
Dataset_prep.cmd
```
which contains the following three .py scripts

utils_folder.py: create folders  
```bash
python utils_folder.py --csv \path_to_csv_file\ --out \path_to_generate_folder\
```

main.py: data preprocessing and prepares the dependent variable y 
```bash
python main.py --csv path_to_csv_file --base main_path_to_build_the_dataset --sdf path_containing_all_the_sdf_files --out_som path_output_for_y_som
```

utils_sdf.py: gets a set of the sdf molecules for the next step
```bash
python utils_sdf.py --base_path \main_path_to_build_the_dataset\ --sdf_m \path_to_sdf_file_multisdfs\ --multi_sdfs \output_folder_for_multisdfs\
```
    
## 3 GNNs training 
To perform the GNN training sinply change the path in the reaction_processing.py file 


```bash
python Glucuronidation.py --root_path "main folder" --file_sdf_inf_p "sdf folder inference"
```

## Usage
Please read the notes.txt file for more information about this work. 

